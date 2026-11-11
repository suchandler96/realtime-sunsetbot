import argparse
import math
import os
import sys
from datetime import datetime, timedelta, timezone, date

for _stream in (sys.stdout, sys.stderr):  # Chinese city names must not crash print() on a legacy (GBK/cp1252) console
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from solar import sun_event, last_light_delay, grazing_height
from weather_data import DataSource, WeatherData
from cities import resolve_city, custom_city, CityLookupError
from satellite import get_source, SOURCES
import fy4b
import gfs_aws
import aerosol
import raytrace
import terrain
import fy4nsmc
import radar
import sunset_core as core
from sunset_core import (PRESSURE_LEVELS, HA_KM, X_MIN, X_MAX, STEP_KM, TERRAIN_STEP_KM,
                         INTERMEDIATE_RAY_STEP_MIN)


def parse_args() -> argparse.Namespace:
    """Parse command-line query options."""
    parser = argparse.ArgumentParser(
        description="Plot an atmosphere cross-section near sunrise/sunset."
    )
    parser.add_argument("--city", default=None,
                        help="city name, English (any case) or Chinese: cities.py first, then looked up online "
                             "(Chinese cities; an English name shared by several cities must be given in Chinese). "
                             "Default: Shanghai, unless --lat/--lon are given")
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
    parser.add_argument("--data-source", default=DataSource.GFS.name, type=str.upper,
                        choices=[s.name for s in DataSource],
                        help="weather model (case-insensitive): GFS = Open-Meteo gfs_seamless blend; "
                             "GFS_AWS = plain GFS from NOAA's public AWS bucket, with cloud liquid/ice water so "
                             "the figure shows cloud phase (falls back to GFS if the download fails); "
                             "ECMWF, JMA_MSM, CMA_GFS, HRRR via Open-Meteo")
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
    parser.add_argument("--clm-file", default="",
                        help="optional local FY-4B CLM .nc file (cloud mask) for the cloud strip; "
                             "without it the newest CLM file is downloaded (like the CTH file)")
    parser.add_argument("--cloud-strip", action=argparse.BooleanOptionalAction, default=True,
                        help="with --satellite-source fy4b, add a strip of the FY-4B cloud mask and cloud optical "
                             "thickness under the figure (downloads the newest CLM and CPD files) and break the "
                             "cloud-top line where there is no cloud")
    parser.add_argument("--patient-download", action=argparse.BooleanOptionalAction, default=False,
                        help="automatic FY-4B download: keep downloading the newest file however slow it is "
                             "(only a minute without any data counts as failed) instead of moving on to another "
                             "file when the download is too slow")
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
    parser.add_argument("--transmittance", action=argparse.BooleanOptionalAction, default=True,
                        help="integrate the extinction (cloud, aerosol, Rayleigh) along each light ray and show its "
                             "direct transmittance next to the ray time")
    parser.add_argument("--ray-min-transmittance", type=float, default=0.001,
                        help="threshold for reporting direct transmission; cloud checks continue over the full ray")
    parser.add_argument("--cloud-map", action=argparse.BooleanOptionalAction, default=False,
                        help="also draw Himawari visible + infrared satellite maps (NASA GIBS) around the "
                             "light path, with the path drawn on them")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False,
                        help="save the figure to a PNG file")
    parser.add_argument("--silent", action=argparse.BooleanOptionalAction, default=False,
                        help="suppress plt.show for server run")
    return parser.parse_args()


# Latin glyphs stay DejaVu Sans; Chinese characters (city names in titles) use the first installed CJK font.
# Only installed fonts are listed: matplotlib logs a "findfont" warning for every missing family in the list.
from matplotlib import font_manager
_installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
CJK_FONT = next((name for name in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC",
                                   "WenQuanYi Micro Hei", "Source Han Sans SC", "Noto Sans SC", "Heiti SC",
                                   "Noto Sans CJK JP", "Noto Sans CJK TC") if name in _installed_fonts), None)
if CJK_FONT:
    plt.rcParams["font.family"] = ["DejaVu Sans", CJK_FONT]

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
    try:
        city = resolve_city(args.city or "Shanghai")
    except CityLookupError as exc:
        sys.exit(f"Error: {exc}")
    print(f"Observer: {city.name} ({city.latitude:.4f}, {city.longitude:.4f}), time zone {city.timezone}")
INCLUDE_TERRAIN = True    # 绘制地形 (AWS terrarium DEM)，并把观测者放在本地海拔

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
CLM_FILE = args.clm_file
CLOUD_STRIP = args.cloud_strip
COT_FILE = CPD_FILE  # the file the cloud strip takes COT from (CPD_FILE itself only enables the companion figure)
CTH_METHOD = args.cth_method

# 雷达反射率开关
INCLUDE_RADAR = args.radar
RADAR_FILE = args.radar_file

# 气溶胶消光系数开关
INCLUDE_AEROSOL = args.aerosol
RAY_MIN_TRANSMITTANCE = args.ray_min_transmittance
AEROSOL_WAVELENGTH = args.aerosol_wavelength
AEROSOL_LEVELS_KM = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]  # iso-line values of the extinction coefficient (km^-1)

# 是否保存图片
SILENT_MODE = args.silent
SAVE_FIGURE = args.save

# 模型数据更新时间 (UTC 起报时刻)：先取发布/估算值，取到真实数据后以其实际起报时刻为准
DATA_UPDATE_STR = core.model_run_label(DATA_SOURCE)

# 计算日出/日落时间、太阳方位角与仰角，以及沿太阳方向的采样点 (-100 km 到 900 km)
transect = core.make_transect(city, DATE, EVENT)
sun = transect.sun
AZIMUTH = transect.azimuth
SUN_ELEVATION = transect.sun_elevation  # 负值表示太阳在地平线以下
RADAR_STEP_KM = 1  # radar pixels are ~1.4-2.4 km, so sample the radar much finer than the model grid
FY4B_STEP_KM = 4   # FY-4B L2 pixels are 4 km, so sample the satellite at its native resolution

distances = transect.distances
radar_distances = np.arange(X_MIN, X_MAX + RADAR_STEP_KM, RADAR_STEP_KM)
lats, lons = transect.lats, transect.lons

# 卫星采样点：FY-4B 用其原生分辨率，其余数据源保持模式网格间距 (Himawari 的采样逐点扫描整幅图，较慢)
satellite_step_km = FY4B_STEP_KM if SATELLITE_SOURCE == "fy4b" else STEP_KM
satellite_distances = np.arange(X_MIN, X_MAX + satellite_step_km, satellite_step_km)
satellite_lats, satellite_lons = core.points_along(transect, satellite_distances)
radar_lats, radar_lons = core.points_along(transect, radar_distances)

# Cloud phase (only with data carrying cloud liquid/ice water): ice share of the cloud-water mass
CLOUD_WATER_MIN = 1e-8                 # kg/kg; below this the cell is treated as having no cloud water
PHASE_CLASSES = [("Liquid cloud", 0.0, 0.2, "///", "tab:blue"),
                 ("Mixed-phase cloud", 0.2, 0.8, "xxx", "tab:green"),
                 ("Ice cloud", 0.8, 1.0001, "...", "tab:purple")]   # (label, ice fraction from, to, hatch, colour)

# 相对湿度色标 (与 ExampleProfile.png 一致)
RH_BOUNDS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]
RH_COLORS = [
    "#E6D2C7", "#ECDDD6", "#F2EAE7", "#F9F5F2", "#FFFFFF",
    "#E6F0E7", "#D3E6D3", "#C3DCC9", "#97BEAB", "#9FD6DD",
    "#8BC1DD", "#6DA3D2",
]

# ==================== 3. 请求气象数据 (相对湿度与云量) ====================
print(
    f"正在获取 {DATE} {EVENT} 的相对湿度与云量剖面 "
    f"({DATA_SOURCE.value}, az={AZIMUTH:.1f}°, elev={SUN_ELEVATION:.2f}°) ..."
)
# 计算部分见 sunset_core.py (同样可一次批量取多个剖面)；这里只取一个剖面
(weather,), DATA_SOURCE = core.fetch_weather_batch([transect], DATA_SOURCE)
HAS_CLOUD_PHASE = "cloud_ice_water" in weather.variables   # only the native-model (AWS GFS) data carries it
if weather.model_run is not None:   # the run really used, not an estimate
    DATA_UPDATE_STR = core.model_run_label(DATA_SOURCE, weather)

print(f"Computed {EVENT} time: {sun.time:%Y-%m-%d %H:%M:%S %Z}")

# 剖面数组：真实高度、地形、云底、RH/云量/温度 (见 core.Profile)
profile = core.build_profile(transect, weather, cloud_base_km=args.cloud_base, include_terrain=INCLUDE_TERRAIN)
time_idx = profile.time_idx
height_km = profile.height_km
terrain_distances = profile.terrain_distances
terrain_km = profile.terrain_km
observer_km = profile.observer_km
model_terrain_km = profile.model_terrain_km
underground = profile.underground
ground_top_km = profile.ground_top_km
local_cbh_km = profile.local_cbh_km
rh_matrix = profile.rh
cloud_matrix = profile.cloud
temp_matrix = profile.temp
cloud_liquid_matrix = profile.cloud_liquid
cloud_ice_matrix = profile.cloud_ice
ice_fraction = profile.ice_fraction

# 3.3 卫星云顶信息：通过 SATELLITE_SOURCE 选择数据源，并沿剖面采样 (可选)
cloud_top_temp_c = None
cloud_top_alt_km = None
cloud_state = None
if INCLUDE_SATELLITE:
    if SATELLITE_SOURCE == "fy4b" and not SATELLITE_FILE:
        # No file given: fetch the newest FY-4B file(s) from the NSMC portal (login session is cached).
        downloaded = {}
        try:
            portal = fy4nsmc.Portal()  # shared, so the login is done once
            if not SATELLITE_FILE:
                downloaded.update(fy4nsmc.fetch_newest(["CTH"], portal=portal, patient=args.patient_download)[0])
                SATELLITE_FILE = (downloaded.get("CTH") or [""])[0]
            # CLM / CPD: the files closest in time to the CTH file, so that the three products describe the same clouds
            companions = (["CLM"] if CLOUD_STRIP and not CLM_FILE else []) \
                         + (["CPD"] if (AUTO_CPD or CLOUD_STRIP) and not CPD_FILE else [])
            if companions and SATELLITE_FILE:
                downloaded.update(fy4nsmc.fetch_newest(
                    companions, portal=portal, reference_time=fy4nsmc.file_start_time(SATELLITE_FILE),
                    patient=args.patient_download)[0])
        except Exception as exc:  # network / login problems must not kill the weather figure
            print(f"FY-4B download failed ({exc}); continuing without it")
        CLM_FILE = CLM_FILE or (downloaded.get("CLM") or [""])[0]
        COT_FILE = CPD_FILE or (downloaded.get("CPD") or [""])[0]
        if AUTO_CPD:
            CPD_FILE = COT_FILE
    if SATELLITE_SOURCE == "fy4b" and SATELLITE_FILE:
        cth_time = fy4nsmc.file_start_time(SATELLITE_FILE)
        for label, path in (("CLM", CLM_FILE), ("CPD", COT_FILE)):
            other = fy4nsmc.file_start_time(path) if path else None
            if cth_time and other and abs((other - cth_time).total_seconds()) > 15 * 60:
                print(f"WARNING: the {label} file is {abs((other - cth_time).total_seconds()) / 60:.0f} min "
                      f"{'after' if other > cth_time else 'before'} the CTH file ({other:%H:%M} vs {cth_time:%H:%M} UTC); "
                      f"clouds may have moved, so the strip and the cloud-top line can disagree at cloud edges")
    satellite_source = get_source(SATELLITE_SOURCE, CTH_METHOD)
    cloud_top = satellite_source.read_cloud_top(sun.time, satellite_lats, satellite_lons, local_file=SATELLITE_FILE)
    cloud_top_temp_c = cloud_top.temperature_c
    cloud_top_alt_km = cloud_top.altitude_km
    if CLOUD_STRIP and SATELLITE_SOURCE == "fy4b" and SATELLITE_FILE and os.path.exists(SATELLITE_FILE):
        try:
            cloud_state = fy4b.cloud_strip(SATELLITE_FILE, CLM_FILE, COT_FILE, satellite_lats, satellite_lons)
        except Exception as exc:  # a bad file must not kill the weather figure
            print(f"FY-4B cloud strip failed ({exc}); continuing without it")
        else:
            print(f"FY-4B cloud strip: mask from {cloud_state.mask_source}, "
                  + ("COT from " + os.path.basename(COT_FILE) if np.isfinite(cloud_state.cot).any() else "no COT values"))
            if cloud_top_alt_km is not None:  # no cloud -> no cloud top: break the line there
                cloud_top_alt_km = np.where(cloud_state.state == 0, np.nan, cloud_top_alt_km)

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
# Strips under the main axes (same x axis): satellite cloud mask / COT first, then radar.
strip_names = (["cloud"] if cloud_state is not None else []) + (["radar"] if radar_dbz is not None else [])
if strip_names:
    fig, (ax, *strip_axes) = plt.subplots(
        1 + len(strip_names), 1, figsize=(14, 6 + 0.48 * len(strip_names)), dpi=120, sharex=True,
        gridspec_kw={"height_ratios": [12] + [0.6] * len(strip_names), "hspace": 0.12},
        )
    strips = dict(zip(strip_names, strip_axes))
else:
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    strip_axes, strips = [], {}

# 4.1 绘制相对湿度背景色 (与 ExampleProfile.png 色标一致)
X = np.broadcast_to(distances, height_km.shape)
Y = height_km  # (levels, points): true height of each pressure level at each point
rh_cmap = ListedColormap(RH_COLORS)
rh_norm = BoundaryNorm(RH_BOUNDS, rh_cmap.N)
contour = ax.contourf(X, Y, rh_matrix, levels=RH_BOUNDS, cmap=rh_cmap, norm=rh_norm)
cbar = fig.colorbar(contour, ax=ax, ticks=RH_BOUNDS)
cbar.set_label('Relative Humidity (%)', fontsize=11)

# 4.1b 叠加云量：云量 >= 50% 的区域以斜线阴影表示，低云也能显示出来
if ice_fraction is None:
    ax.contourf(X, Y, cloud_matrix, levels=[50, 101], colors='none', hatches=['///'])
else:   # one hatch per cloud phase; each class is contoured from the cloud cover itself (like the single hatch),
    # because masking cells without cloud water would erase layers only one pressure level thick
    for _label, low, high, hatch, colour in PHASE_CLASSES:
        in_class = (cloud_matrix >= 50) & (ice_fraction >= low) & (ice_fraction < high)
        if in_class.any():
            with plt.rc_context({'hatch.color': colour}):
                ax.contourf(X, Y, np.where(in_class, cloud_matrix, 0.0), levels=[50, 101], colors='none', hatches=[hatch])
    no_water = (cloud_matrix >= 50) & np.isnan(ice_fraction)   # cloudy by the cover field, but no cloud water in the model
    if no_water.any():
        with plt.rc_context({'hatch.color': 'dimgray'}):
            ax.contourf(X, Y, np.where(no_water, cloud_matrix, 0.0), levels=[50, 101], colors='none', hatches=['\\\\'])
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
        satellite_distances, cloud_top_temp_c, color='navy', marker='o', markersize=2.5,
        linewidth=1.2, label='Cloud-top temp proxy (°C)',
    )
    ax2.set_ylabel('Cloud-top temp proxy (°C)', fontsize=11)
    ax2.set_ylim(-90, 40)

# 4.1e FY-4B 云顶高度沿剖面
if cloud_top_alt_km is not None:
    ax.plot(satellite_distances, cloud_top_alt_km, color='darkgreen',
            marker='s', markersize=2, linewidth=1.0,
            label='FY-4B cloud-top height (km)')

# 4.1f 雷达反射率条带 (沿剖面，色标与 NMC 图例一致；白色=无回波，灰色=雷达范围外)
if radar_dbz is not None:
    ax_radar = strips["radar"]
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
    ax_radar.set_title(
        f"NMC composite reflectivity {radar_times} UTC "
        "(NMC colors, 5-70+ dBZ; white: no echo, gray: not covered)",
        fontsize=9, loc="left", pad=2)

# 4.1g FY-4B 云检测 (CLM) 与云光学厚度 (COT) 条带：白色 = 无云；COT 越小颜色越浅；
#      深灰 = 有云但无 COT 反演 (如夜间/终止线附近)；浅灰 = 无数据
cloud_cot_cmap, cloud_cot_norm = None, None
if cloud_state is not None:
    ax_cloud = strips["cloud"]
    COT_MIN, COT_MAX = 1, 100.0
    cloud_cot_cmap = ListedColormap(plt.cm.Blues(np.linspace(0.01, 1.0, 256)))
    cloud_cot_norm = LogNorm(COT_MIN, COT_MAX, clip=True)
    colours = np.empty((1, len(satellite_distances), 3))
    for k, (state, cot) in enumerate(zip(cloud_state.state, cloud_state.cot)):
        if np.isnan(state):
            colours[0, k] = (0.85, 0.85, 0.85)
        elif state == 0:
            colours[0, k] = (1.0, 1.0, 1.0)
        elif np.isnan(cot):
            colours[0, k] = (0.45, 0.45, 0.45)
        else:
            colours[0, k] = cloud_cot_cmap(cloud_cot_norm(cot))[:3]
    half = satellite_step_km / 2
    ax_cloud.imshow(colours, aspect="auto", interpolation="nearest",
                    extent=(satellite_distances[0] - half, satellite_distances[-1] + half, 0, 1))
    ax_cloud.set_yticks([])
    ax_cloud.set_ylabel("FY-4B", rotation=0, ha="right", va="center")
    ax_cloud.set_title(f"FY-4B cloud mask ({cloud_state.mask_source}) and cloud optical thickness "
                       "(white: no cloud, lighter: thinner; dark gray: cloud without COT, light gray: no data)",
                       fontsize=9, loc="left", pad=2)
if strip_axes:
    strip_axes[-1].set_xlabel("Distance from Observer (km) [Towards Sun direction]", fontsize=12)

# 4.2 计算并绘制两条边界光路 (光线追踪，计算见 core.trace_rays)
# 展平坐标系：x 为沿地表朝向太阳的距离 (km)，y 为海拔高度 (km)。
# 两条光线都经过观测者头顶的云底 (x=0, y=h)，分别代表最早和最晚还能照到云底的光线。
sun_side_x = distances[distances >= 0]  # 只绘制朝向太阳的一侧 (x >= 0)

rays = core.trace_rays(profile, aerosol_profile, aerosol_height_km, AEROSOL_WAVELENGTH,
                       transmittance=args.transmittance, ray_min_transmittance=RAY_MIN_TRANSMITTANCE)
extinction_model = None
if rays is not None:
    extinction_model = rays.extinction_model
    ground_ray_time, ground_ray_name = rays.ground_ray_time, rays.ground_ray_name
    aerosol_ray_time, aerosol_ray_name = rays.aerosol_ray_time, rays.aerosol_ray_name
    scan_delays, scan_results = rays.scan_delays, rays.scan_results
    intermediate_rays = rays.intermediate_rays
    ray_y_at, T_text = rays.ray_y_at, rays.t_text


    def label_ray_time(x, y, when, color, side, result=None, top_km=15.3, **text):
        """Write the ray's clock time beside it near the top of the figure (side: 'left' or 'right' of the ray)."""
        inside = np.where(y <= top_km)[0]
        if len(inside) == 0:
            return
        i = inside[-1]  # far end of the ray still inside the axes
        if result is not None:  # swatch = color of the sunlight that reaches the cloud along this ray
            ax.plot(x[i], y[i], 's', markersize=8, markerfacecolor=result.normalized_light_colour, markeredgecolor=color,
                    markeredgewidth=0.8, zorder=4)
        ax.annotate(f"{when:%H:%M}{T_text(result)}", (x[i], y[i]), xytext=(9 if side == "right" else -9, 0),
                    textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                    color=color, **{"fontsize": 11, "fontweight": "bold", **text})


    last = len(scan_delays) - 1
    # 第一条 (贴地) 与最后一条 (气溶胶 / 地形限制) 光线始终绘出，不受透过率阈值影响
    ground_ray_y = ray_y_at(sun_side_x, scan_delays[0])
    ax.plot(sun_side_x, ground_ray_y, color='darkorange', linestyle='--', linewidth=2, zorder=3,
            label=f'{ground_ray_name} light to cloud base')
    label_ray_time(sun_side_x, ground_ray_y, ground_ray_time, 'darkorange', "left", scan_results[0])
    if last > 0:
        aerosol_ray_y = ray_y_at(sun_side_x, scan_delays[last])
        ax.plot(sun_side_x, aerosol_ray_y, color='crimson', linestyle='--', linewidth=2, zorder=3,
                label=f'{aerosol_ray_name} light to cloud base')
        label_ray_time(sun_side_x, aerosol_ray_y, aerosol_ray_time, 'crimson', "right", scan_results[last])
    for i, (when, y, result) in enumerate(intermediate_rays):
        ax.plot(sun_side_x, y, color='darkgoldenrod', linestyle=':', linewidth=1.2, zorder=3,
                label=f'Intermediate light (every {INTERMEDIATE_RAY_STEP_MIN} min)' if i == 0 else None)
        label_ray_time(sun_side_x, y, when, 'darkgoldenrod', "right", result, fontsize=8, fontweight="normal")

# 4.3 标记核心元素
ax.plot(0, observer_km, 'ro', markersize=8, zorder=6)
if local_cbh_km is not None:
    ax.plot(0, local_cbh_km, 'gX', markersize=10, zorder=3, label=f'Local Cloud Base ({local_cbh_km:.2f} km)') # 本地云底

# 绘制地形 (无 DEM 时退回为 0 km 以下的灰色地面)
if terrain_km is not None:
    # 不透明，盖住 RH、云量、等温线与气溶胶等值线 (含低于模式地形的外推数据)，但位于光线与标记之下
    ax.fill_between(terrain_distances, -1, ground_top_km, color='#d9d9d9', zorder=2.5, label='Ground')
else:
    ax.fill_between(distances, -1, 0, color='gray', alpha=0.3, label='Ground')

# ==================== 5. 图表美化 ====================
sun_direction = round(AZIMUTH % 360, 1) % 360  # 0-359.9 degrees clockwise from north
name_prefix = f"{DATE.replace('-', '')}_{city.name.replace(' ', '')}_{EVENT}"
name_suffix = f"{DATA_SOURCE.name}_{DATA_UPDATE_STR}"
file_stem = f"{name_prefix}_{name_suffix}"
title = f"{DATE.replace('-', '')}_{(city.title_name if CJK_FONT else city.name).replace(' ', '')}_{EVENT}({sun_direction:.1f}°)_{name_suffix}"  # the file name stays without the direction
ax.set_title(title, fontsize=14, fontweight='bold')
if not strip_axes:
    ax.set_xlabel("Distance from Observer (km) [Towards Sun direction]", fontsize=12)
ax.set_ylabel("Altitude (km)", fontsize=12)
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(0, 16) # 对流层高度
ax.grid(True, linestyle=':', alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
if ice_fraction is None:
    handles.append(Patch(facecolor='none', edgecolor='dimgray', hatch='///'))
    labels.append('Cloud cover ≥ 50%')
else:
    for label, _low, _high, hatch, colour in PHASE_CLASSES:
        handles.append(Patch(facecolor='none', edgecolor=colour, hatch=hatch))
        labels.append(f'{label} (cover ≥ 50%)')
    if no_water.any():
        handles.append(Patch(facecolor='none', edgecolor='dimgray', hatch='\\\\'))
        labels.append('Cloud cover ≥ 50%, no cloud water')
if extinction_model is not None:
    handles.append(Line2D([0], [0], marker='s', linestyle='none', markersize=8, markerfacecolor='#ff9a5a',
                          markeredgecolor='dimgray'))
    labels.append('Color of the cloud-casting sunlight')
if cloud_top_line is not None:
    handles.append(cloud_top_line)
    labels.append('Cloud-top temp proxy (°C)')
if aerosol_handle is not None:
    handles.append(aerosol_handle)
    labels.append(f'Aerosol extinction {AEROSOL_WAVELENGTH} nm (km⁻¹)')
ax.legend(handles=handles, labels=labels, loc='lower right')

if not strip_axes:
    plt.tight_layout()
else:
    # The RH colorbar narrows the main axes; give the strips the same x extent
    # so the distance axes line up.
    fig.canvas.draw()
    main_pos = ax.get_position()
    for strip_ax in strip_axes:
        strip_pos = strip_ax.get_position()
        strip_ax.set_position([main_pos.x0, strip_pos.y0, main_pos.width, strip_pos.height])
    if cloud_cot_cmap is not None:
        # COT color bar in the free margin beside the cloud strip
        cloud_pos = ax_cloud.get_position()
        cot_cax = fig.add_axes([main_pos.x1 + 0.02, cloud_pos.y0 + 0.3 * cloud_pos.height,
                                0.09, 0.4 * cloud_pos.height])
        cot_bar = fig.colorbar(plt.cm.ScalarMappable(norm=cloud_cot_norm, cmap=cloud_cot_cmap), cax=cot_cax,
                               orientation="horizontal", ticks=[0.1, 1, 10, 100])
        cot_bar.ax.set_xticklabels(["0.1", "1", "10", "100"])
        cot_bar.ax.tick_params(labelsize=8)
        cot_bar.set_label("COT", fontsize=8, labelpad=1)
if radar_dbz is not None:
    # Radar legend: one horizontal row centred under the x-axis title.
    xlabel_bb = strip_axes[-1].xaxis.label.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted())
    fig.legend(handles=radar_handles, title="Radar reflectivity (dBZ)", loc="upper center",
               ncol=len(radar_handles), bbox_to_anchor=(main_pos.x0 + main_pos.width / 2, xlabel_bb.y0 - 0.01),
               fontsize=9, title_fontsize=9, handlelength=1.0, handleheight=0.7,
               columnspacing=0.8, handletextpad=0.4, borderpad=0.4, framealpha=0.85)
if SAVE_FIGURE:
    out_path = f"{file_stem}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Figure saved to {out_path}")

# ==================== 6. FY-4B 云参数剖面图 (native resolution) ====================
# 卫星源选为 fy4b 且同时给出 CTH 与 CPD 文件时，额外绘制云顶高度/光学厚度剖面图。
if SATELLITE_SOURCE == "fy4b" and SATELLITE_FILE and CPD_FILE:
    if os.path.exists(SATELLITE_FILE) and os.path.exists(CPD_FILE):
        companion_path = f"{file_stem}_cloud_products.png" if SAVE_FIGURE else None
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

# ==================== 7. Himawari 可见光/红外云图 (NASA GIBS) ====================
if args.cloud_map:
    import cloudmap
    try:
        cloudmap.draw_cloud_map(
            city.latitude, city.longitude, AZIMUTH, sun.time_utc,
            title=f"{city.name} {DATE} {EVENT}", x_min=0, x_max=X_MAX,
            save_path=f"{file_stem}_cloudmap.png" if SAVE_FIGURE else None)
    except Exception as exc:  # network / server problems must not kill the main figure
        print(f"Cloud map skipped: {exc}")

if not SILENT_MODE:
    plt.show()
