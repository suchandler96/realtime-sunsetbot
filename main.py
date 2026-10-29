import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone, date

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from solar import sun_event, last_light_delay, grazing_height
from weather_data import DataSource, WeatherData
from cities import get_city, custom_city, CITIES
from satellite import get_source, SOURCES
import fy4b
import aerosol
import terrain
import fy4nsmc
import radar


def parse_args() -> argparse.Namespace:
    """Parse command-line query options."""
    parser = argparse.ArgumentParser(
        description="Plot an atmosphere cross-section near sunrise/sunset."
    )
    parser.add_argument("--city", default=None, choices=[c.name for c in CITIES],
                        help="city name from cities.py (default: Shanghai, unless --lat/--lon are given)")
    parser.add_argument("--lat", type=float, default=None,
                        help="observer latitude in degrees north (-90..90); use with --lon instead of --city")
    parser.add_argument("--lon", type=float, default=None,
                        help="observer longitude in degrees east (-180..180, or 0..360)")
    parser.add_argument("--place-name", default=None,
                        help="label for the figure title / file name when using --lat/--lon "
                             "(default: the coordinates)")
    parser.add_argument("--timezone", default=None,
                        help="IANA time zone for --lat/--lon, e.g. Asia/Shanghai "
                             "(default: looked up from the coordinates)")
    parser.add_argument("--date", default=date.today().strftime('%Y-%m-%d'), help="local date YYYY-MM-DD")
    parser.add_argument("--event", default="sunset", choices=["sunrise", "sunset"],
                        help="which solar event to plot")
    parser.add_argument("--data-source", default=DataSource.GFS.name,
                        choices=[s.name for s in DataSource], help="weather model: GFS, ECMWF, JMA_MSM, CMA_GFS, HRRR")
    parser.add_argument("--cloud-base", type=float, default=-1,
                        help="manual cloud base height in km")
    parser.add_argument("--satellite", action=argparse.BooleanOptionalAction, default=False,
                        help="include a satellite cloud-top layer in the figure")
    parser.add_argument("--satellite-source", default="fy4b",
                        choices=list(SOURCES.keys()),
                        help="satellite source: himawari -> cloud-top temperature, fy4b -> cloud-top height")
    parser.add_argument("--satellite-file", default="",
                        help="optional local satellite .nc file (skip download/auto-detect if provided); "
                             "with --satellite-source fy4b and no file, the newest FY-4B CTH file is "
                             "downloaded from the NSMC portal (see fy4nsmc.py)")
    parser.add_argument("--cpd-file", default="",
                        help="optional local FY-4B CPD .nc file (cloud optical thickness); "
                             "with --satellite-source fy4b and --satellite-file this adds the cloud-products transect figure")
    parser.add_argument("--cth-method", default="weighted", choices=fy4b.CTH_METHODS,
                        help="how a transect point's FY-4B cloud-top height is taken from the 3x3 pixels around "
                             "its nearest pixel (valid pixels only): nearest = that pixel; max / median / mean; "
                             "weighted = mean weighted by the retrieval-quality flag")
    parser.add_argument("--auto-cpd", action=argparse.BooleanOptionalAction, default=False,
                        help="with the automatic FY-4B download, also fetch the newest CPD file "
                             "(cloud optical thickness, daytime only) for the cloud-products transect figure")
    parser.add_argument("--radar", action=argparse.BooleanOptionalAction, default=False,
                        help="add an NMC radar-reflectivity strip along the transect")
    parser.add_argument("--radar-file", default="",
                        help="optional local NMC radar mosaic PNG (skip region selection/download; region taken from file name)")
    parser.add_argument("--aerosol", action=argparse.BooleanOptionalAction, default=False,
                        help="draw iso-lines of the CAMS 3D aerosol extinction coefficient "
                             "(needs an ADS account: ~/.cdsapirc and the dataset licence accepted)")
    parser.add_argument("--aerosol-wavelength", type=int, default=532, choices=aerosol.WAVELENGTHS_NM,
                        help="wavelength (nm) of the aerosol extinction coefficient")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False,
                        help="save the figure to a PNG file")
    parser.add_argument("--silent", action=argparse.BooleanOptionalAction, default=False,
                        help="suppress plt.show for server run")
    return parser.parse_args()


# ==================== 1. 核心参数配置 ====================
args = parse_args()
if (args.lat is None) != (args.lon is None):
    sys.exit("--lat and --lon must be given together")
if args.lat is not None:
    if args.city:
        sys.exit("use either --city or --lat/--lon, not both")
    try:
        city = custom_city(args.lat, args.lon, args.place_name, args.timezone)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Observer: {city.name} ({city.latitude:.4f}, {city.longitude:.4f}), time zone {city.timezone}")
else:
    city = get_city(args.city or "Shanghai")
R_EARTH_EQUATOR = 6378.1370
R_EARTH_POLAR = 6356.7523

a_cos_phi_sq = (R_EARTH_EQUATOR * np.cos(np.radians(city.latitude))) ** 2
b_sin_phi_sq = (R_EARTH_POLAR * np.sin(np.radians(city.latitude))) ** 2
R_EARTH = np.sqrt((R_EARTH_EQUATOR**2 * a_cos_phi_sq + R_EARTH_POLAR**2 * b_sin_phi_sq) /
                  (a_cos_phi_sq + b_sin_phi_sq))
INCLUDE_TERRAIN = True    # 绘制地形 (AWS terrarium DEM)，并把观测者放在本地海拔
HA_KM = 1.0               # 气溶胶最低透光高度 (km)，可调整

# 云底高度模式：True 为自动识别，False 为手动指定 MANUAL_CBH_KM
AUTO_CLOUD_BASE = args.cloud_base < 0
MANUAL_CBH_KM = args.cloud_base

# 观测日期与事件
DATE = args.date
EVENT = args.event
TIMEZONE = city.timezone

# 数据源
DATA_SOURCE = DataSource.__members__[args.data_source]

# 卫星云顶层开关与数据源
INCLUDE_SATELLITE = args.satellite
SATELLITE_SOURCE = args.satellite_source
SATELLITE_FILE = args.satellite_file
CPD_FILE = args.cpd_file
AUTO_CPD = args.auto_cpd
CTH_METHOD = args.cth_method

# 雷达反射率开关
INCLUDE_RADAR = args.radar
RADAR_FILE = args.radar_file

# 气溶胶消光系数开关
INCLUDE_AEROSOL = args.aerosol
AEROSOL_WAVELENGTH = args.aerosol_wavelength
AEROSOL_LEVELS_KM = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]  # iso-line values of the extinction coefficient (km^-1)

# 是否保存图片
SILENT_MODE = args.silent
SAVE_FIGURE = args.save

# 模型数据更新时间估算 (UTC 起报时刻)。
# Open-Meteo 标准 forecast 接口不直接返回起报时刻，这里按当前时间与模型循环周期估算。
MODEL_CYCLE_HOURS = {DataSource.GFS: 6, DataSource.ECMWF: 12, DataSource.JMA_MSM: 3,
                     DataSource.CMA_GFS: 6, DataSource.HRRR: 1}
MODEL_RUN_DELAY_HOURS = {DataSource.GFS: 5, DataSource.ECMWF: 9, DataSource.JMA_MSM: 2,
                         DataSource.CMA_GFS: 6, DataSource.HRRR: 2}


def latest_model_run(source):
    """估算当前可用的最新模型起报时刻 (UTC)。"""
    available = datetime.now(timezone.utc) - timedelta(hours=MODEL_RUN_DELAY_HOURS[source])
    cycle = MODEL_CYCLE_HOURS[source]
    run_hour = available.hour - (available.hour % cycle)
    return available.replace(hour=run_hour, minute=0, second=0, microsecond=0)


DATA_UPDATE_STR = latest_model_run(DATA_SOURCE).strftime("%Y%m%d%H") + "z"

# 计算日出/日落时间、太阳方位角与仰角
sun = sun_event(city.latitude, city.longitude, DATE, EVENT, TIMEZONE)
AZIMUTH = sun.position.azimuth
SUN_ELEVATION = sun.position.geometric_elevation  # 负值表示太阳在地平线以下

# 空间跨度配置：从 -100km 到 900km
X_MIN, X_MAX = -100, 900
STEP_KM = 20
RADAR_STEP_KM = 1  # radar pixels are ~1.4-2.4 km, so sample the radar much finer than the model grid

distances = np.arange(X_MIN, X_MAX + STEP_KM, STEP_KM)
radar_distances = np.arange(X_MIN, X_MAX + RADAR_STEP_KM, RADAR_STEP_KM)

# 垂直气压层与对应的大致海拔高度 (米)
PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]
# Approximate heights; only a fallback when the model gives no geopotential height.
ALTITUDES_KM = [0.1, 0.2, 0.4, 0.6, 0.8, 1.5, 2.0, 3.0, 4.2, 5.6, 7.2, 9.2, 10.4, 11.8, 13.6, 16.2]

# 相对湿度色标 (与 ExampleProfile.png 一致)
RH_BOUNDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
RH_COLORS = [
    "#E6D2C7", "#ECDDD6", "#F2EAE7", "#F9F5F2", "#FFFFFF",
    "#E6F0E7", "#D3E6D3", "#C3DCC9", "#97BEAB", "#9FD6DD",
    "#8BC1DD", "#6DA3D2",
]

# ==================== 2. 采样点地理坐标计算 ====================
def get_point_at_distance(lat, lon, dist_km, azimuth):
    R = R_EARTH
    bearing = math.radians(azimuth)
    d_r = dist_km / R
    lat1, lon1 = math.radians(lat), math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_r) * math.cos(lat1), math.cos(d_r) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

lats, lons = [], []
for d in distances:
    pt_lat, pt_lon = get_point_at_distance(city.latitude, city.longitude, d, AZIMUTH)
    lats.append(pt_lat)
    lons.append(pt_lon)

radar_lats, radar_lons = zip(*(
    get_point_at_distance(city.latitude, city.longitude, d, AZIMUTH) for d in radar_distances))

# ==================== 3. 请求气象数据 (相对湿度与云量) ====================
print(
    f"正在获取 {DATE} {EVENT} 的相对湿度与云量剖面 "
    f"({DATA_SOURCE.value}, az={AZIMUTH:.1f}°, elev={SUN_ELEVATION:.2f}°) ..."
)
weather = (
    WeatherData.source(DATA_SOURCE)
    .coordinates(lats, lons)
    .pressure_levels(PRESSURE_LEVELS)
    .relative_humidity()
    .cloud_cover()
    .temperature()
    .geopotential_height()
    .timezone(TIMEZONE)
    .on(DATE)
    .fetch()
)

print(f"Computed {EVENT} time: {sun.time:%Y-%m-%d %H:%M:%S %Z}")

# Some models lack some pressure levels (e.g. ECMWF: 975/950/900/800 hPa); fill them by interpolation.
filled_levels = weather.fill_missing_levels()
if filled_levels:
    print(f"Interpolated pressure levels missing in {DATA_SOURCE.value}: "
          + ", ".join(f"{p} hPa" for p in filled_levels))

# 选择距离日出/日落时刻最近的小时
time_idx = weather.nearest_index(sun.time)

# 每个剖面点、每个气压层的真实位势高度 (km)；缺失时退回到 ALTITUDES_KM 近似表
height_km = weather.values("geopotential_height")[:, :, time_idx] / 1000.0
height_km = np.where(np.isnan(height_km), np.asarray(ALTITUDES_KM)[:, None], height_km)

# 3.0 地形：沿剖面 (1 km 间隔) 的 DEM 海拔 (km)；观测者位于 x=0 处的本地海拔
TERRAIN_STEP_KM = 3
terrain_distances = np.arange(X_MIN, X_MAX + TERRAIN_STEP_KM, TERRAIN_STEP_KM)
terrain_lats, terrain_lons = zip(*(
    get_point_at_distance(city.latitude, city.longitude, d, AZIMUTH) for d in terrain_distances))
terrain_km = None
observer_km = 0.0
if INCLUDE_TERRAIN:
    try:
        terrain_km = terrain.read_elevation(terrain_lats, terrain_lons)
    except Exception as exc:  # network problems must not kill the weather figure
        print(f"Terrain download failed ({exc}); continuing without terrain")
    if terrain_km is not None:
        observer_km = float(np.interp(0.0, terrain_distances, terrain_km))
        print(f"Terrain: observer at {observer_km * 1000:.0f} m; profile {terrain_km.min() * 1000:.0f}"
              f"-{terrain_km.max() * 1000:.0f} m")
# 模式自身的 (平滑) 地形高度 (km)；低于它的气压层数据是外推值。
# 图中不用 NaN 去掩盖 (会在地形附近留下白色区域)，而是让地面填充一直盖到 DEM 与模式地形中较高者；
# 识别本地云底时则直接排除这些层。
model_terrain_km = np.nan_to_num(weather.model_elevation_km, nan=0.0)
underground = height_km < model_terrain_km[None, :]
ground_top_km = None
if terrain_km is not None:
    ground_top_km = np.maximum(terrain_km, np.interp(terrain_distances, distances, model_terrain_km))
    print(f"Model terrain: {model_terrain_km.min() * 1000:.0f}-{model_terrain_km.max() * 1000:.0f} m "
          f"(observer cell {model_terrain_km[distances == 0][0] * 1000:.0f} m)")

# 3.1 本地(x=0)云底高度
if AUTO_CLOUD_BASE:
    # 自动识别：取首个相对湿度饱和或云量明显的层结
    obs_idx = np.where(distances == 0)[0][0]
    rh_at_observer = np.where(underground[:, obs_idx], np.nan, weather.values("relative_humidity")[:, obs_idx, time_idx])
    cloud_at_observer = np.where(underground[:, obs_idx], np.nan, weather.values("cloud_cover")[:, obs_idx, time_idx])

    cloud_layer_mask = (rh_at_observer >= 99) | (cloud_at_observer >= 50)
    if np.any(cloud_layer_mask):
        local_cbh_km = float(height_km[int(np.argmax(cloud_layer_mask)), obs_idx])
    else:
        local_cbh_km = 6.0  # 晴空无云时，默认设定一个高空火烧云幕布高度
else:
    local_cbh_km = MANUAL_CBH_KM

# 3.2 相对湿度、云量与温度矩阵 [高度层, 距离点]，取距离日出/日落最近时次
rh_matrix = weather.values("relative_humidity")[:, :, time_idx]
cloud_matrix = weather.values("cloud_cover")[:, :, time_idx]
temp_matrix = weather.values("temperature")[:, :, time_idx]

# 3.3 卫星云顶信息：通过 SATELLITE_SOURCE 选择数据源，并沿剖面采样 (可选)
cloud_top_temp_c = None
cloud_top_alt_km = None
if INCLUDE_SATELLITE:
    if SATELLITE_SOURCE == "fy4b" and not SATELLITE_FILE:
        # No file given: fetch the newest FY-4B file(s) from the NSMC portal (login session is cached).
        wanted = ["CTH"] + (["CPD"] if AUTO_CPD and not CPD_FILE else [])
        try:
            downloaded, _ = fy4nsmc.fetch_newest(wanted)
        except Exception as exc:  # network / login problems must not kill the weather figure
            print(f"FY-4B download failed ({exc}); continuing without it")
            downloaded = {}
        SATELLITE_FILE = (downloaded.get("CTH") or [""])[0]
        CPD_FILE = CPD_FILE or (downloaded.get("CPD") or [""])[0]
    satellite_source = get_source(SATELLITE_SOURCE, CTH_METHOD)
    cloud_top = satellite_source.read_cloud_top(sun.time, lats, lons, local_file=SATELLITE_FILE)
    cloud_top_temp_c = cloud_top.temperature_c
    cloud_top_alt_km = cloud_top.altitude_km

# 3.4 NMC 雷达组合反射率：沿剖面采样 (可选，雷达回波强 -> 云多)
radar_dbz = None
if INCLUDE_RADAR:
    radar_result = radar.read_reflectivity(sun.time, radar_lats, radar_lons, local_file=RADAR_FILE,
                                           half_window=0)
    radar_dbz = radar_result.dbz
    if radar_dbz is None:
        print("Radar: no picture found, strip skipped.")
    else:
        radar_times = ", ".join(f"{k} {t:%Y-%m-%d %H:%M}" for k, t in radar_result.obs_times_utc.items())
        print(f"Radar pictures (UTC): {radar_times}")

# 3.5 CAMS 三维气溶胶消光系数：沿剖面采样 (可选)
aerosol_profile = None
aerosol_height_km = None
if INCLUDE_AEROSOL:
    try:
        aerosol_profile = aerosol.read_extinction(sun.time, lats, lons, AEROSOL_WAVELENGTH)
    except Exception as exc:  # credentials / network problems must not kill the weather figure
        print(f"CAMS aerosol failed ({exc}); continuing without it")
    if aerosol_profile is None:
        print("Aerosol: no CAMS data available, iso-lines skipped.")
    else:
        aerosol_height_km = aerosol.level_heights_km(aerosol_profile.levels_hpa, PRESSURE_LEVELS, height_km)
        print(f"CAMS aerosol: run {aerosol_profile.run:%Y-%m-%d %Hz}, valid {aerosol_profile.valid:%Y-%m-%d %H:%M} UTC, "
              f"{AEROSOL_WAVELENGTH} nm extinction {np.nanmin(aerosol_profile.extinction_per_km):.3g}"
              f"-{np.nanmax(aerosol_profile.extinction_per_km):.3g} km^-1")

# ==================== 4. 绘制剖面图与模拟光路 ====================
if radar_dbz is not None:
    fig, (ax, ax_radar) = plt.subplots(
        2, 1, figsize=(14, 6.8), dpi=120, sharex=True,
        gridspec_kw={"height_ratios": [12, 1], "hspace": 0.12},
    )
else:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)

# 4.1 绘制相对湿度背景色 (与 ExampleProfile.png 色标一致)
X = np.broadcast_to(distances, height_km.shape)
Y = height_km  # (levels, points): true height of each pressure level at each point
rh_cmap = ListedColormap(RH_COLORS)
rh_norm = BoundaryNorm(RH_BOUNDS, rh_cmap.N)
contour = ax.contourf(X, Y, rh_matrix, levels=RH_BOUNDS, cmap=rh_cmap, norm=rh_norm)
cbar = fig.colorbar(contour, ax=ax, ticks=RH_BOUNDS)
cbar.set_label('Relative Humidity (%)', fontsize=11)

# 4.1b 叠加云量：云量 >= 50% 的区域以斜线阴影表示，低云也能显示出来
ax.contourf(X, Y, cloud_matrix, levels=[50, 101], colors='none', hatches=['///'])
ax.contour(X, Y, cloud_matrix, levels=[50], colors='dimgray', linewidths=0.6, alpha=0.7, zorder=1.2)

# 4.1c 叠加等温线 (每隔 5°C)
temp_levels = np.arange(-80, 31, 5)
temp_contour = ax.contour(X, Y, temp_matrix, levels=temp_levels, colors='black', linewidths=0.7)
for label in ax.clabel(temp_contour, inline=True, fontsize=8, fmt='%d°C'):
    label.set_zorder(2.2)  # 与等值线一起位于地形之下

# 4.1c2 气溶胶消光系数等值线 (CAMS, km^-1)
aerosol_handle = None
if aerosol_profile is not None:
    aerosol_x = np.broadcast_to(distances, aerosol_height_km.shape)
    aerosol_contour = ax.contour(aerosol_x, aerosol_height_km, aerosol_profile.extinction_per_km,
                                 levels=AEROSOL_LEVELS_KM, colors='purple', linewidths=1.0, linestyles='--')
    for label in ax.clabel(aerosol_contour, inline=True, fontsize=8, fmt='%g'):
        label.set_zorder(2.2)
    aerosol_handle = Line2D([0], [0], color='purple', linestyle='--', linewidth=1.0)

# 4.1d 卫星云顶温度代理 (Himawari tbb_14) 沿剖面
cloud_top_line = None
if cloud_top_temp_c is not None:
    ax2 = ax.twinx()
    (cloud_top_line,) = ax2.plot(
        distances, cloud_top_temp_c, color='navy', marker='o', markersize=2.5,
        linewidth=1.2, label='Cloud-top temp proxy (°C)',
    )
    ax2.set_ylabel('Cloud-top temp proxy (°C)', fontsize=11)
    ax2.set_ylim(-90, 40)

# 4.1e FY-4B 云顶高度沿剖面
if cloud_top_alt_km is not None:
    ax.plot(distances, cloud_top_alt_km, color='darkgreen',
            marker='s', markersize=3, linewidth=1.5,
            label='FY-4B cloud-top height (km)')

# 4.1f 雷达反射率条带 (沿剖面，色标与 NMC 图例一致；白色=无回波，灰色=雷达范围外)
if radar_dbz is not None:
    radar_cmap = ListedColormap(radar.LEGEND_RGB / 255.0)
    radar_cmap.set_under("white")
    radar_cmap.set_bad("lightgray")
    radar_edges = np.r_[5.0 + 5.0 * np.arange(len(radar.LEGEND_RGB)), 1e9]
    radar_norm = BoundaryNorm(radar_edges, radar_cmap.N)
    half = RADAR_STEP_KM / 2
    ax_radar.imshow(
        np.where(radar_dbz > 0, radar_dbz, np.where(np.isnan(radar_dbz), np.nan, -1.0))[None, :], cmap=radar_cmap, norm=radar_norm,
        aspect="auto", extent=(radar_distances[0] - half, radar_distances[-1] + half, 0, 1),
        interpolation="nearest",
    )
    # Legend of the reflectivity classes; placed under the RH colorbar after layout.
    n_leg = len(radar.LEGEND_RGB)
    radar_handles = [
        Patch(facecolor=radar.LEGEND_RGB[i] / 255.0, edgecolor="0.4", linewidth=0.3,
              label=f"{5 + 5 * i}+" if i == n_leg - 1 else f"{5 + 5 * i}-{10 + 5 * i}")
        for i in range(n_leg)
    ]
    ax_radar.set_yticks([])
    ax_radar.set_ylabel("Radar", rotation=0, ha="right", va="center")
    ax_radar.set_xlabel("Distance from Observer (km) [Towards Sun direction]", fontsize=12)
    ax_radar.set_title(
        f"NMC composite reflectivity {radar_times} UTC "
        "(NMC colours, 5-70+ dBZ; white = no echo, gray = outside mosaic)",
        fontsize=9, loc="left")

# 4.2 计算并绘制两条边界光路 (光线追踪)
# 展平坐标系：x 为沿地表朝向太阳的距离 (km)，y 为海拔高度 (km)。
# 两条光线都经过观测者头顶的云底 (x=0, y=h)，分别代表最早和最晚还能照到云底的光线。
sun_side_x = distances[distances >= 0]  # 只绘制朝向太阳的一侧 (x >= 0)


def earliest_light_y(x, h):
    """最早能照到云底的光线：经过 (0, h)，平行于与地表 (半径 R)于(0,0)相切的光线。"""
    s = x / R_EARTH
    return h + 0.5 * (R_EARTH + h) * s**2 * (1.0 + 5.0 * s**2 / 12.0)


def latest_light_y(x, h, ha=None):
    """最晚还能照到云底的光线：与气溶胶层顶 (半径 R+ha) 相切，并经过 (0, h)。

    ha 为光线最低点的高度，缺省为气溶胶层顶；中间光线传入介于 0 与层顶之间的值。"""
    if ha is None:
        ha = min(HA_KM, h)  # 若云底低于气溶胶层顶，则最晚光线退化为与地表相切
    theta = math.acos((R_EARTH + ha) / (R_EARTH + h))
    s = theta - x / R_EARTH
    return ha + (R_EARTH + ha) * 0.5 * s**2 * (1.0 + 5.0 * s**2 / 12.0)


# 两条光线对应的时刻：贴地光线 = 本地日出/日落时刻；掠过气溶胶层顶的光线与之相差 last_light_delay
# (日落后延迟 / 日出前提前)。日落时贴地光线最早，日出时则最晚。
ground_ray_time = sun.time
aerosol_ray_delay = last_light_delay(city.latitude, sun.position.declination, R_EARTH,
                                     local_cbh_km, min(HA_KM, local_cbh_km))
aerosol_ray_limit = "aerosol top"


def terrain_limited_delay(delay):
    """地形遮挡修正：气溶胶光线若落到 地形 + 气溶胶层顶 之下，则每次提前 1 分钟迭代，直到不再被遮挡。

    只在上一条光线被遮挡的 x 位置上检验新光线 (光线族逐条抬高，被遮挡的 x 只会越来越少)；
    返回 (新的延迟, 初始光线被遮挡的 x 范围 (km) 或 None, 是否在延迟 0 (最早光线) 处仍被遮挡)。"""
    x_check = terrain_distances[terrain_distances >= 0]
    # 气溶胶层顶不会高于观测者头顶的云底
    ha_above_ground = min(HA_KM, max(local_cbh_km - observer_km, 0.0))
    required = np.interp(x_check, terrain_distances, terrain_km) + ha_above_ground
    blocked = np.where(latest_light_y(x_check, local_cbh_km) < required - 1e-9)[0]
    if blocked.size == 0:
        return delay, None, False
    blocked_range = (float(x_check[blocked[0]]), float(x_check[blocked[-1]]))
    step_back = timedelta(minutes=1)
    while blocked.size and delay > timedelta(0):
        delay = max(delay - step_back, timedelta(0))
        y = latest_light_y(x_check[blocked], local_cbh_km, grazing_height(
            city.latitude, sun.position.declination, R_EARTH, local_cbh_km, delay))
        blocked = blocked[y < required[blocked] - 1e-9]
    return delay, blocked_range, bool(blocked.size)


if terrain_km is not None:
    unblocked_delay = aerosol_ray_delay
    aerosol_ray_delay, blocked_range, still_blocked = terrain_limited_delay(aerosol_ray_delay)
    if blocked_range is not None:
        aerosol_ray_limit = "terrain"
        print(f"Aerosol-top-grazing ray ({unblocked_delay.total_seconds() / 60:.1f} min) is blocked by terrain at "
              f"x = {blocked_range[0]:.0f}-{blocked_range[1]:.0f} km; stepping back 1 min at a time -> "
              f"{aerosol_ray_delay.total_seconds() / 60:.1f} min")
        if still_blocked:
            print("Warning: even the earliest ray (at the ground-grazing time) is blocked by terrain.")
aerosol_ray_time = ground_ray_time + (aerosol_ray_delay if EVENT == "sunset" else -aerosol_ray_delay)
ground_ray_name, aerosol_ray_name = (("Earliest", "Latest") if EVENT == "sunset" else ("Latest", "Earliest"))
print(f"{ground_ray_name} light to cloud base (ground-grazing ray): {ground_ray_time:%H:%M:%S %Z}; "
      f"{aerosol_ray_name.lower()} ({aerosol_ray_limit}-grazing ray): {aerosol_ray_time:%H:%M:%S %Z} "
      f"({aerosol_ray_delay.total_seconds() / 60:.1f} min {'after sunset' if EVENT == 'sunset' else 'before sunrise'})")

ground_ray_y = earliest_light_y(sun_side_x, local_cbh_km)
aerosol_ray_y = latest_light_y(
    sun_side_x, local_cbh_km,
    None if aerosol_ray_limit == "aerosol top" else grazing_height(
        city.latitude, sun.position.declination, R_EARTH, local_cbh_km, aerosol_ray_delay))
ax.plot(sun_side_x, ground_ray_y, color='darkorange', linestyle='--', linewidth=2, zorder=3,
        label=f'{ground_ray_name} light to cloud base')
ax.plot(sun_side_x, aerosol_ray_y, color='crimson', linestyle='--', linewidth=2, zorder=3,
        label=f'{aerosol_ray_name} light to cloud base')


INTERMEDIATE_RAY_STEP_MIN = 5   # 中间光线的时间间隔 (从日出/日落时刻起算)
INTERMEDIATE_RAY_GAP_MIN = 2    # 最后一条中间光线与气溶胶光线至少相隔的分钟数
step = timedelta(minutes=INTERMEDIATE_RAY_STEP_MIN)
direction = 1 if EVENT == "sunset" else -1
intermediate_rays = []  # (time, ray height)
delay = step
while delay <= aerosol_ray_delay - timedelta(minutes=INTERMEDIATE_RAY_GAP_MIN):
    grazing = grazing_height(city.latitude, sun.position.declination, R_EARTH, local_cbh_km, delay)
    intermediate_rays.append((ground_ray_time + direction * delay, latest_light_y(sun_side_x, local_cbh_km, grazing)))
    delay += step
for i, (when, y) in enumerate(intermediate_rays):
    ax.plot(sun_side_x, y, color='darkgoldenrod', linestyle=':', linewidth=1.2, zorder=3,
            label=f'Intermediate light (every {INTERMEDIATE_RAY_STEP_MIN} min)' if i == 0 else None)


def label_ray_time(x, y, when, color, side, top_km=15.3, **text):
    """Write the ray's clock time beside it near the top of the figure (side: 'left' or 'right' of the ray)."""
    inside = np.where(y <= top_km)[0]
    if len(inside) == 0:
        return
    i = inside[-1]  # far end of the ray still inside the axes
    ax.annotate(f"{when:%H:%M}", (x[i], y[i]), xytext=(6 if side == "right" else -6, 0),
                textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                color=color, **{"fontsize": 11, "fontweight": "bold", **text})


label_ray_time(sun_side_x, ground_ray_y, ground_ray_time, 'darkorange', "left")
label_ray_time(sun_side_x, aerosol_ray_y, aerosol_ray_time, 'crimson', "right")
for when, y in intermediate_rays:
    label_ray_time(sun_side_x, y, when, 'darkgoldenrod', "right", fontsize=8, fontweight="normal")

# 4.3 标记核心元素
ax.plot(0, observer_km, 'ro', markersize=8, zorder=6)
ax.plot(0, local_cbh_km, 'gX', markersize=10, zorder=3, label=f'Local Cloud Base ({local_cbh_km:.2f} km)') # 本地云底

# 绘制地形 (无 DEM 时退回为 0 km 以下的灰色地面)
if terrain_km is not None:
    # 不透明，盖住 RH、云量、等温线与气溶胶等值线 (含低于模式地形的外推数据)，但位于光线与标记之下
    ax.fill_between(terrain_distances, -1, ground_top_km, color='#d9d9d9', zorder=2.5, label='Ground')
else:
    ax.fill_between(distances, -1, 0, color='gray', alpha=0.3, label='Ground')

# ==================== 5. 图表美化 ====================
title = f"{DATE.replace('-', '')}_{city.name.replace(' ', '')}_{EVENT}_{DATA_SOURCE.name}_{DATA_UPDATE_STR}"
ax.set_title(title, fontsize=14, fontweight='bold')
if radar_dbz is None:
    ax.set_xlabel("Distance from Observer (km) [Towards Sun direction]", fontsize=12)
ax.set_ylabel("Altitude (km)", fontsize=12)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(0, 16) # 对流层高度
ax.grid(True, linestyle=':', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor='none', edgecolor='dimgray', hatch='///', label='Cloud cover ≥ 50%'))
labels.append('Cloud cover ≥ 50%')
if cloud_top_line is not None:
    handles.append(cloud_top_line)
    labels.append('Cloud-top temp proxy (°C)')
if aerosol_handle is not None:
    handles.append(aerosol_handle)
    labels.append(f'Aerosol extinction {AEROSOL_WAVELENGTH} nm (km⁻¹)')
ax.legend(handles=handles, labels=labels, loc='lower right')

if radar_dbz is None:
    plt.tight_layout()
else:
    # The RH colorbar narrows the main axes; give the radar strip the same x extent
    # so the distance axes line up.
    fig.canvas.draw()
    main_pos, radar_pos = ax.get_position(), ax_radar.get_position()
    ax_radar.set_position([main_pos.x0, radar_pos.y0, main_pos.width, radar_pos.height])
    # Radar legend: one horizontal row centred under the x-axis title.
    xlabel_bb = ax_radar.xaxis.label.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    fig.legend(handles=radar_handles, title="Radar reflectivity (dBZ)", loc="upper center",
               ncol=len(radar_handles), bbox_to_anchor=(main_pos.x0 + main_pos.width / 2, xlabel_bb.y0 - 0.01),
               fontsize=9, title_fontsize=9, handlelength=1.0, handleheight=0.7,
               columnspacing=0.8, handletextpad=0.4, borderpad=0.4, framealpha=0.85)
if SAVE_FIGURE:
    out_path = f"{title}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

# ==================== 6. FY-4B 云参数剖面图 (native resolution) ====================
# 卫星源选为 fy4b 且同时给出 CTH 与 CPD 文件时，额外绘制云顶高度/光学厚度剖面图。
if SATELLITE_SOURCE == "fy4b" and SATELLITE_FILE and CPD_FILE:
    if os.path.exists(SATELLITE_FILE) and os.path.exists(CPD_FILE):
        companion_path = f"{title}_cloud_products.png" if SAVE_FIGURE else None
        fy4b.plot_transect(
            SATELLITE_FILE, CPD_FILE,
            city.latitude, city.longitude, AZIMUTH,
            x_min=X_MIN, x_max=X_MAX,
            save_path=companion_path, show=False,
        )
        print(f"Cloud-products transect drawn from {SATELLITE_FILE} + {CPD_FILE}")
        if companion_path:
            print(f"Cloud-products figure saved to {companion_path}")
    else:
        print("Cloud-products transect skipped: satellite file(s) not found.")

if not SILENT_MODE:
    plt.show()
