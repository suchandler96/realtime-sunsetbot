"""Compute side of the sunrise/sunset cross-section: no figure code, no command line.

main.py draws one profile; other callers (for example a vividness map over many points) can reuse the very same
steps on many transects:

    transect = make_transect(city, "2026-11-10", "sunset")
    (weather,), source = fetch_weather_batch([transect], DataSource.GFS)       # one profile ...
    weathers, source = fetch_weather_batch(transects, DataSource.GFS_AWS)      # ... or a whole batch in one go
    profile = build_profile(transect, weathers[0])
    rays = trace_rays(profile)

`fetch_weather_batch` joins the sampling points of all transects into one request (Open-Meteo: chunked GETs;
AWS GFS: one set of GRIB fields sampled at every point), then hands each transect its own `WeatherResult.subset`.
A batch of one is exactly the old single fetch, and every later step only sees one transect's `WeatherResult`.
Pass `log=lambda *a, **k: None` to the functions below to silence their progress output.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

import gfs_aws
import raytrace
import terrain
from solar import SunEvent, grazing_height, last_light_delay, sun_event
from weather_data import DataSource, WeatherData, WeatherResult

# ==================== constants ====================
R_EARTH_EQUATOR = 6378.1370
R_EARTH_POLAR = 6356.7523
HA_KM = 1.0               # lowest height (km) at which the aerosol layer lets light through
X_MIN, X_MAX = -100, 900  # transect extent (km) from the observer; positive = towards the sun
STEP_KM = 20              # model sampling step along the transect
TERRAIN_STEP_KM = 3       # DEM sampling step along the transect

# Vertical pressure levels and the heights (km) used only as a fallback when the model gives no geopotential height
PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200, 150, 100]
ALTITUDES_KM = [0.1, 0.2, 0.4, 0.6, 0.8, 1.5, 2.0, 3.0, 4.2, 5.6, 7.2, 9.2, 10.4, 11.8, 13.6, 16.2]

CLOUD_WATER_MIN = 1e-8    # kg/kg; below this a cell counts as having no cloud water

# Model initialisation time estimate (UTC cycle) when the real run time is unknown.
MODEL_CYCLE_HOURS = {DataSource.GFS: 6, DataSource.GFS_AWS: 6, DataSource.ECMWF: 12, DataSource.JMA_MSM: 3,
                     DataSource.CMA_GFS: 6, DataSource.HRRR: 1}
MODEL_RUN_DELAY_HOURS = {DataSource.GFS: 5, DataSource.GFS_AWS: 4, DataSource.ECMWF: 9, DataSource.JMA_MSM: 2,
                         DataSource.CMA_GFS: 6, DataSource.HRRR: 2}
# Open-Meteo publishes the real run time per dataset (gfs_seamless is a blend; use the global GFS).
META_DATASET = {DataSource.GFS: "ncep_gfs013", DataSource.GFS_AWS: "ncep_gfs013"}


def _quiet(*_args, **_kwargs):
    return None


# ==================== geometry ====================
def earth_radius_km(latitude_deg: float) -> float:
    """Geocentric radius of the reference ellipsoid at a latitude."""
    a_cos_phi_sq = (R_EARTH_EQUATOR * np.cos(np.radians(latitude_deg))) ** 2
    b_sin_phi_sq = (R_EARTH_POLAR * np.sin(np.radians(latitude_deg))) ** 2
    return float(np.sqrt((R_EARTH_EQUATOR**2 * a_cos_phi_sq + R_EARTH_POLAR**2 * b_sin_phi_sq) /
                         (a_cos_phi_sq + b_sin_phi_sq)))


def get_point_at_distance(lat, lon, dist_km, azimuth, r_earth):
    """Point `dist_km` from (lat, lon) along the great circle that leaves at `azimuth` degrees (clockwise from north)."""
    bearing = math.radians(azimuth)
    d_r = dist_km / r_earth
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(d_r) + math.cos(lat1) * math.sin(d_r) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d_r) * math.cos(lat1),
                             math.cos(d_r) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def points_along(transect: "Transect", distances) -> tuple[list[float], list[float]]:
    """Coordinates of the given distances (km) along the transect (radar / satellite use their own step)."""
    pts = [get_point_at_distance(transect.city.latitude, transect.city.longitude, d, transect.azimuth,
                                 transect.r_earth) for d in distances]
    return [p[0] for p in pts], [p[1] for p in pts]


@dataclass
class Transect:
    """One observer, one solar event, and the line of sampling points towards the sun."""
    city: object              # cities.City-like: latitude, longitude, timezone, name
    date: str                 # local date YYYY-MM-DD
    event: str                # "sunrise" or "sunset"
    sun: SunEvent
    r_earth: float
    distances: np.ndarray     # km from the observer, sun side positive
    lats: list
    lons: list
    x_min: float
    x_max: float
    step_km: float

    @property
    def timezone(self) -> str:
        return self.city.timezone

    @property
    def azimuth(self) -> float:
        return self.sun.position.azimuth

    @property
    def sun_elevation(self) -> float:
        return self.sun.position.geometric_elevation   # negative: the sun is below the horizon

    @property
    def observer_index(self) -> int:
        return int(np.where(self.distances == 0)[0][0])


def make_transect(city, date: str, event: str, x_min: float = X_MIN, x_max: float = X_MAX,
                  step_km: float = STEP_KM) -> Transect:
    sun = sun_event(city.latitude, city.longitude, date, event, city.timezone)
    r_earth = earth_radius_km(city.latitude)
    distances = np.arange(x_min, x_max + step_km, step_km)
    lats, lons = zip(*(get_point_at_distance(city.latitude, city.longitude, d, sun.position.azimuth, r_earth)
                       for d in distances))
    return Transect(city=city, date=date, event=event, sun=sun, r_earth=r_earth, distances=distances,
                    lats=list(lats), lons=list(lons), x_min=x_min, x_max=x_max, step_km=step_km)


# ==================== model run time ====================
def latest_model_run(source: DataSource) -> datetime:
    """Estimated newest available model run (UTC)."""
    available = datetime.now(timezone.utc) - timedelta(hours=MODEL_RUN_DELAY_HOURS[source])
    cycle = MODEL_CYCLE_HOURS[source]
    run_hour = available.hour - (available.hour % cycle)
    return available.replace(hour=run_hour, minute=0, second=0, microsecond=0)


def published_model_run(source: DataSource) -> datetime | None:
    """Latest run initialisation time (UTC) reported by Open-Meteo, or None if unavailable."""
    import requests
    dataset = META_DATASET.get(source, source.value)
    try:
        r = requests.get(f"https://api.open-meteo.com/data/{dataset}/static/meta.json", timeout=5)
        r.raise_for_status()
        return datetime.fromtimestamp(r.json()["last_run_initialisation_time"], timezone.utc)
    except Exception:
        return None


def model_run_label(source: DataSource, weather: WeatherResult | None = None) -> str:
    """'YYYYMMDDHHz' of the run behind the data: the real one if the backend knows it, else the published/estimated."""
    if weather is not None and weather.model_run is not None:
        return weather.model_run.strftime("%Y%m%d%H") + "z"
    run = published_model_run(source) or latest_model_run(source)
    return run.strftime("%Y%m%d%H") + "z"


# ==================== weather ====================
def request_profile(source: DataSource, lats, lons, date: str, tz: str, valid_at, backend=None,
                    cloud_water: bool = False) -> WeatherResult:
    """One pressure-level request for any number of points (a single transect or a batch of them)."""
    request = (
        WeatherData.source(source, backend=backend)
        .coordinates(lats, lons)
        .pressure_levels(PRESSURE_LEVELS)
        .relative_humidity()
        .cloud_cover()
        .temperature()
        .geopotential_height()
        .timezone(tz)
        .on(date)
        .valid_at(valid_at)
    )
    return (request.cloud_water() if cloud_water else request).fetch()


def fetch_weather_batch(transects: list[Transect], source: DataSource, valid_at=None,
                        log=print) -> tuple[list[WeatherResult], DataSource]:
    """Fetch the profiles of one or many transects with a single batched request; one `WeatherResult` per transect.

    All transects must share the local date and time zone (one request has one of each).  `valid_at` is the moment
    single-time backends (AWS GFS: one forecast hour per request) are asked for; by default the middle of the
    transects' event times, which for one transect is simply its own event time.  Open-Meteo returns the whole
    day, and `build_profile` picks each transect's own hour.  `GFS_AWS` falls back to Open-Meteo's GFS for the
    whole batch when the AWS download fails.  Returns the results and the data source actually used."""
    if not transects:
        raise ValueError("at least one transect is required")
    if len({(t.date, t.timezone) for t in transects}) != 1:
        raise ValueError("a batched fetch needs one local date and one time zone for all transects")
    first = transects[0]
    lats = [lat for t in transects for lat in t.lats]
    lons = [lon for t in transects for lon in t.lons]
    if valid_at is None:
        valid_at = sorted(t.sun.time for t in transects)[len(transects) // 2]

    weather = None
    if source is DataSource.GFS_AWS:
        try:
            weather = request_profile(source, lats, lons, first.date, first.timezone, valid_at,
                                      backend=gfs_aws.GfsAwsBackend(), cloud_water=True)
        except Exception as exc:  # network / bucket problems must not kill the figure: use Open-Meteo's GFS instead
            log(f"GFS download from AWS failed ({exc}); using Open-Meteo GFS instead (no cloud phase)")
            source = DataSource.GFS
    if weather is None:
        weather = request_profile(source, lats, lons, first.date, first.timezone, valid_at)
    if weather.model_run is not None:   # the run really used, not an estimate
        log(f"GFS from AWS: run {weather.model_run:%Y-%m-%d %Hz} (plain global GFS, with cloud liquid/ice water)")

    # Some models lack some pressure levels (e.g. ECMWF: 975/950/900/800 hPa); fill them by interpolation.
    filled_levels = weather.fill_missing_levels()
    if filled_levels:
        log(f"Interpolated pressure levels missing in {source.value}: "
            + ", ".join(f"{p} hPa" for p in filled_levels))

    results, start = [], 0
    for t in transects:
        results.append(weather.subset(slice(start, start + len(t.lats))))
        start += len(t.lats)
    return results, source


# ==================== profile ====================
@dataclass
class Profile:
    """Everything the ray tracer and the figure need about one transect's atmosphere."""
    transect: Transect
    weather: WeatherResult
    time_idx: int
    height_km: np.ndarray            # (levels, points): true height of each pressure level
    underground: np.ndarray          # (levels, points): level lies below the model's own terrain
    model_terrain_km: np.ndarray     # (points,)
    terrain_distances: np.ndarray
    terrain_km: np.ndarray | None    # DEM heights along terrain_distances (None: no terrain)
    observer_km: float
    ground_top_km: np.ndarray | None
    rh: np.ndarray                   # (levels, points) at the event hour
    cloud: np.ndarray
    temp: np.ndarray
    cloud_liquid: np.ndarray | None  # only data carrying cloud water (AWS GFS)
    cloud_ice: np.ndarray | None
    ice_fraction: np.ndarray | None
    local_cbh_km: float | None       # None: no cloud above the observer, so no rays
    has_cloud_phase: bool = False


def load_terrain(transect: Transect, log=print):
    """DEM heights (km) along the transect, or None if the download fails."""
    terrain_distances = np.arange(transect.x_min, transect.x_max + TERRAIN_STEP_KM, TERRAIN_STEP_KM)
    t_lats, t_lons = points_along(transect, terrain_distances)
    try:
        return terrain.read_elevation(t_lats, t_lons)
    except Exception as exc:  # network problems must not kill the weather figure
        log(f"Terrain download failed ({exc}); continuing without terrain")
        return None


def detect_cloud_base(distances, height_km, underground, rh, cloud, observer_index, log=print):
    """First level above the observer with saturated air or sizeable cloud; None when the sky is clear."""
    obs = observer_index
    rh_at_observer = np.where(underground[:, obs], np.nan, rh[:, obs])
    cloud_at_observer = np.where(underground[:, obs], np.nan, cloud[:, obs])
    cloud_layer_mask = (rh_at_observer >= 99) | (cloud_at_observer >= 50)
    if not np.any(cloud_layer_mask):
        return None  # clear sky: no default cloud base, no rays and no integration
    k = int(np.argmax(cloud_layer_mask))
    local_cbh_km = float(height_km[k, obs])
    # The base lies between the first cloudy level and the clear level below it: interpolate where the thresholds
    # are crossed (consistent with the cloud-region edge in the figure)
    if k > 0 and not np.isnan(rh_at_observer[k - 1]):
        crossings = []
        for below, here, threshold in ((rh_at_observer[k - 1], rh_at_observer[k], 99.0),
                                       (cloud_at_observer[k - 1], cloud_at_observer[k], 50.0)):
            if here >= threshold > below:
                crossings.append(height_km[k - 1, obs]
                                 + (threshold - below) / (here - below) * (height_km[k, obs] - height_km[k - 1, obs]))
        if crossings:
            local_cbh_km = float(min(crossings))
            log(f"Local cloud base: {local_cbh_km:.2f} km (interpolated; first cloudy level at "
                f"{height_km[k, obs]:.2f} km)")
    return local_cbh_km


def build_profile(transect: Transect, weather: WeatherResult, cloud_base_km: float = -1.0,
                  include_terrain: bool = True, terrain_km=None, log=print) -> Profile:
    """Turn one transect's `WeatherResult` into the arrays used for ray tracing and plotting.

    `cloud_base_km` < 0 detects the cloud base above the observer; otherwise it is taken as given.  Terrain is
    downloaded unless `terrain_km` (DEM heights along the transect at TERRAIN_STEP_KM) is supplied or
    `include_terrain` is False."""
    distances = transect.distances
    time_idx = weather.nearest_index(transect.sun.time)   # the hour closest to the event

    # True geopotential height of every level at every point; fall back to the approximate table where missing
    height_km = weather.values("geopotential_height")[:, :, time_idx] / 1000.0
    height_km = np.where(np.isnan(height_km), np.asarray(ALTITUDES_KM)[:, None], height_km)

    # Terrain along the profile; the observer stands at the local elevation at x = 0
    terrain_distances = np.arange(transect.x_min, transect.x_max + TERRAIN_STEP_KM, TERRAIN_STEP_KM)
    observer_km = 0.0
    if include_terrain and terrain_km is None:
        terrain_km = load_terrain(transect, log)
    elif not include_terrain:
        terrain_km = None
    if terrain_km is not None:
        observer_km = float(np.interp(0.0, terrain_distances, terrain_km))
        log(f"Terrain: observer at {observer_km * 1000:.0f} m; profile {terrain_km.min() * 1000:.0f}"
            f"-{terrain_km.max() * 1000:.0f} m")
    # The model's own (smooth) terrain; levels below it are extrapolated data.  The figure covers them with the
    # ground fill up to the higher of DEM and model terrain instead of masking them with NaN (which would leave
    # white areas near the terrain); the cloud-base detection excludes them directly.
    model_terrain_km = np.nan_to_num(weather.model_elevation_km, nan=0.0)
    underground = height_km < model_terrain_km[None, :]
    ground_top_km = None
    if terrain_km is not None:
        ground_top_km = np.maximum(terrain_km, np.interp(terrain_distances, distances, model_terrain_km))
        log(f"Model terrain: {model_terrain_km.min() * 1000:.0f}-{model_terrain_km.max() * 1000:.0f} m "
            f"(observer cell {model_terrain_km[distances == 0][0] * 1000:.0f} m)")

    # Relative humidity, cloud cover and temperature [level, point] at the event hour
    rh = weather.values("relative_humidity")[:, :, time_idx]
    cloud = weather.values("cloud_cover")[:, :, time_idx]
    temp = weather.values("temperature")[:, :, time_idx]
    has_cloud_phase = "cloud_ice_water" in weather.variables   # only native-model (AWS GFS) data carries it
    cloud_liquid = cloud_ice = ice_fraction = None
    if has_cloud_phase:
        cloud_liquid = weather.values("cloud_liquid_water")[:, :, time_idx]
        cloud_ice = weather.values("cloud_ice_water")[:, :, time_idx]
        condensate = np.nan_to_num(cloud_liquid) + np.nan_to_num(cloud_ice)
        # ice share of the cloud water mass; NaN where the model holds no cloud water
        ice_fraction = np.where(condensate > CLOUD_WATER_MIN,
                                np.nan_to_num(cloud_ice) / np.maximum(condensate, 1e-30), np.nan)

    if cloud_base_km < 0:
        local_cbh_km = detect_cloud_base(distances, height_km, underground, rh, cloud,
                                         transect.observer_index, log)
    else:
        local_cbh_km = cloud_base_km

    return Profile(transect=transect, weather=weather, time_idx=time_idx, height_km=height_km,
                   underground=underground, model_terrain_km=model_terrain_km, terrain_distances=terrain_distances,
                   terrain_km=terrain_km, observer_km=observer_km, ground_top_km=ground_top_km, rh=rh,
                   cloud=cloud, temp=temp, cloud_liquid=cloud_liquid, cloud_ice=cloud_ice,
                   ice_fraction=ice_fraction, local_cbh_km=local_cbh_km, has_cloud_phase=has_cloud_phase)


# ==================== light rays ====================
def earliest_light_y(x, h, r_earth):
    """Earliest light reaching the cloud base: passes (0, h), parallel to the ray tangent to the ground (radius R) at (0, 0)."""
    s = x / r_earth
    return h + 0.5 * (r_earth + h) * s**2 * (1.0 + 5.0 * s**2 / 12.0)


def latest_light_y(x, h, r_earth, ha=None):
    """Latest light still reaching the cloud base: tangent to the aerosol-layer top (radius R + ha) and passing (0, h).

    `ha` is the height of the ray's lowest point (default: the aerosol top); intermediate rays pass a value
    between 0 and the layer top."""
    if ha is None:
        ha = min(HA_KM, h)  # a cloud base below the aerosol top turns the latest ray into the ground-grazing one
    theta = math.acos((r_earth + ha) / (r_earth + h))
    s = theta - x / r_earth
    return ha + (r_earth + ha) * 0.5 * s**2 * (1.0 + 5.0 * s**2 / 12.0)


def percent(t):
    return f"{t:.0%}" if t >= 0.1 else f"{t:.1%}"


@dataclass
class RayTrace:
    """The family of rays to the cloud base above the observer and their transmittance."""
    profile: Profile
    local_cbh_km: float
    event: str
    ground_ray_time: datetime
    aerosol_ray_delay: timedelta
    aerosol_ray_time: datetime
    aerosol_ray_limit: str                # "aerosol top" or "terrain"
    ground_ray_name: str
    aerosol_ray_name: str
    extinction_model: raytrace.ProfileExtinction | None
    ray_min_transmittance: float
    scan_delays: list
    scan_results: list                    # RayTransmittance per scan delay (None without extinction_model)
    direction: int                        # +1 sunset (rays come after sunset), -1 sunrise
    intermediate_rays: list = field(default_factory=list)   # (time, ray height along sun_side_x, result)

    def ray_y_at(self, x, delay):
        """Height of the ray reaching the local cloud base `delay` after sunset (before sunrise)."""
        t = self.profile.transect
        cbh = self.local_cbh_km
        if delay <= timedelta(0):
            return earliest_light_y(x, cbh, t.r_earth)
        if delay >= self.aerosol_ray_delay:
            return latest_light_y(x, cbh, t.r_earth, None if self.aerosol_ray_limit == "aerosol top" else grazing_height(
                t.city.latitude, t.sun.position.declination, t.r_earth, cbh, self.aerosol_ray_delay))
        return latest_light_y(x, cbh, t.r_earth, grazing_height(
            t.city.latitude, t.sun.position.declination, t.r_earth, cbh, delay))

    def t_text(self, result):
        """Describe cloud blockage explicitly and otherwise apply the direct-beam display threshold."""
        if result is None:
            return ""
        if result.blocked_by_cloud:
            return " blocked"
        return (f" <{percent(self.ray_min_transmittance)}" if result.stopped_at_km is not None
                else f" {percent(result.transmittance)}")


RAY_SCAN_STEP = timedelta(minutes=0.5)   # time step of the scanned rays (same as the terrain iteration)
INTERMEDIATE_RAY_STEP_MIN = 5            # step of the intermediate rays (counted from the event)
INTERMEDIATE_RAY_GAP_MIN = 2             # minimum gap between the last intermediate ray and the aerosol ray


def terrain_limited_delay(profile: Profile, delay: timedelta):
    """Terrain shading: a latest ray that falls below terrain + aerosol top is moved 0.5 min earlier until clear.

    Only the x positions blocked by the previous ray are rechecked (rays rise one by one, so fewer positions stay
    blocked).  Returns (new delay, x range (km) blocked for the initial ray or None, still blocked at delay 0)."""
    t = profile.transect
    cbh = profile.local_cbh_km
    x_check = profile.terrain_distances[profile.terrain_distances >= 0]
    # The aerosol top never lies above the cloud base over the observer
    ha_above_ground = min(HA_KM, max(cbh - profile.observer_km, 0.0))
    required = np.interp(x_check, profile.terrain_distances, profile.terrain_km) + ha_above_ground
    blocked = np.where(latest_light_y(x_check, cbh, t.r_earth) < required - 1e-9)[0]
    if blocked.size == 0:
        return delay, None, False
    blocked_range = (float(x_check[blocked[0]]), float(x_check[blocked[-1]]))
    step_back = timedelta(minutes=0.5)
    while blocked.size and delay > timedelta(0):
        delay = max(delay - step_back, timedelta(0))
        y = latest_light_y(x_check[blocked], cbh, t.r_earth, grazing_height(
            t.city.latitude, t.sun.position.declination, t.r_earth, cbh, delay))
        blocked = blocked[y < required[blocked] - 1e-9]
    return delay, blocked_range, bool(blocked.size)


def trace_rays(profile: Profile, aerosol_profile=None, aerosol_height_km=None, aerosol_wavelength: int = 532,
               transmittance: bool = True, ray_min_transmittance: float = 0.001, log=print) -> RayTrace | None:
    """Trace the rays to the local cloud base and integrate their transmittance; None when there is no cloud base."""
    if profile.local_cbh_km is None:
        log("No cloud base detected and none given (--cloud-base); no light rays are traced.")
        return None
    t = profile.transect
    cbh = profile.local_cbh_km
    event = t.event
    sun = t.sun
    distances = t.distances

    # The two bounding rays: ground-grazing = the local event time; the one grazing the aerosol top differs from it
    # by last_light_delay (after sunset / before sunrise).  At sunset the ground ray is the earliest, at sunrise the latest.
    ground_ray_time = sun.time
    aerosol_ray_delay = last_light_delay(t.city.latitude, sun.position.declination, t.r_earth,
                                         cbh, min(HA_KM, cbh))
    aerosol_ray_limit = "aerosol top"
    if profile.terrain_km is not None:
        unblocked_delay = aerosol_ray_delay
        aerosol_ray_delay, blocked_range, still_blocked = terrain_limited_delay(profile, aerosol_ray_delay)
        if blocked_range is not None:
            aerosol_ray_limit = "terrain"
            log(f"Aerosol-top-grazing ray ({unblocked_delay.total_seconds() / 60:.1f} min) is blocked by terrain at "
                f"x = {blocked_range[0]:.0f}-{blocked_range[1]:.0f} km; stepping back 0.5 min at a time -> "
                f"{aerosol_ray_delay.total_seconds() / 60:.1f} min")
            if still_blocked:
                log("Warning: even the earliest ray (at the ground-grazing time) is blocked by terrain.")
    aerosol_ray_time = ground_ray_time + (aerosol_ray_delay if event == "sunset" else -aerosol_ray_delay)
    ground_ray_name, aerosol_ray_name = (("Earliest", "Latest") if event == "sunset" else ("Latest", "Earliest"))
    log(f"{ground_ray_name} light to cloud base (ground-grazing ray): {ground_ray_time:%H:%M:%S %Z}; "
        f"{aerosol_ray_name.lower()} ({aerosol_ray_limit}-grazing ray): {aerosol_ray_time:%H:%M:%S %Z} "
        f"({aerosol_ray_delay.total_seconds() / 60:.1f} min {'after sunset' if event == 'sunset' else 'before sunrise'})")

    # Aerosol, Rayleigh and ice-cloud extinction along each ray; water/mixed-phase cloud other than the target blocks the beam
    extinction_model = None
    if transmittance:
        extinction_model = raytrace.ProfileExtinction(
            distances, profile.height_km, profile.cloud, profile.temp, underground=profile.underground,
            aerosol_extinction=None if aerosol_profile is None else aerosol_profile.extinction_per_km,
            aerosol_heights_km=aerosol_height_km, wavelength_nm=aerosol_wavelength,
            **(dict(zip(("liquid_extinction", "ice_extinction"), raytrace.condensate_extinction(
                np.asarray(PRESSURE_LEVELS, dtype=float)[:, None], profile.temp, profile.cloud_liquid,
                profile.cloud_ice, profile.cloud))) if profile.has_cloud_phase else {}))
        if aerosol_profile is None:
            log("Transmittance: no aerosol data (use --aerosol); only cloud and Rayleigh extinction are integrated.")

    trace = RayTrace(profile=profile, local_cbh_km=cbh, event=event, ground_ray_time=ground_ray_time,
                     aerosol_ray_delay=aerosol_ray_delay, aerosol_ray_time=aerosol_ray_time,
                     aerosol_ray_limit=aerosol_ray_limit, ground_ray_name=ground_ray_name,
                     aerosol_ray_name=aerosol_ray_name, extinction_model=extinction_model,
                     ray_min_transmittance=ray_min_transmittance, scan_delays=[], scan_results=[],
                     direction=1 if event == "sunset" else -1)

    ray_x_fine = np.arange(0.0, t.x_max + 0.5, 1.0)
    # One ray every 0.5 min
    scan_delays = [RAY_SCAN_STEP * k for k in range(int(aerosol_ray_delay / RAY_SCAN_STEP) + 1)]
    if not scan_delays or scan_delays[-1] < aerosol_ray_delay:
        scan_delays.append(aerosol_ray_delay)
    scan_results = [None] * len(scan_delays)
    if extinction_model is not None:
        log(f"Direct transmittance of {len(scan_delays)} light rays "
            f"(water-cloud blocking + Beer-Lambert ice cloud/CAMS aerosol/Rayleigh); "
            f"{percent(ray_min_transmittance)} is a direct-beam reporting threshold, not a sunset-glow boundary:")
        scan_results = [extinction_model.integrate(
            ray_x_fine, trace.ray_y_at(ray_x_fine, d),
            tau_stop=-math.log(max(ray_min_transmittance, 1e-6))) for d in scan_delays]
    trace.scan_delays, trace.scan_results = scan_delays, scan_results

    shown = [r is None or r.transmittance >= ray_min_transmittance for r in scan_results]
    for k, (d, r) in enumerate(zip(scan_delays, scan_results)):
        if r is not None and (k % 10 == 0 or k == len(scan_delays) - 1 or shown[k] != shown[k - 1]):
            stop = (f", cloud blocked at x={r.stopped_at_km:.0f} km" if r.blocked_by_cloud else
                    f", first below threshold at x={r.stopped_at_km:.0f} km" if r.stopped_at_km is not None else "")
            transmission = ("blocked" if r.blocked_by_cloud else
                            trace.t_text(r).replace(" <", "< ") if r.stopped_at_km is not None else
                            "= " + percent(r.transmittance))
            log(f"  {ground_ray_time + trace.direction * d:%H:%M:%S}: T {transmission} R/G/B "
                f"{'/'.join(percent(x) for x in r.rgb_transmittance)} (tau cloud {r.tau_cloud:.2f}, "
                f"aerosol {r.tau_aerosol:.2f}, Rayleigh {r.tau_rayleigh:.2f}){stop}"
                f"{'' if shown[k] else '  [below direct-beam threshold]'}")

    # Intermediate rays at multiples of 5 min: always shown (direct transmittance is no hard limit of visibility)
    sun_side_x = distances[distances >= 0]
    step = timedelta(minutes=INTERMEDIATE_RAY_STEP_MIN)
    delay = step
    while delay <= aerosol_ray_delay - timedelta(minutes=INTERMEDIATE_RAY_GAP_MIN):
        k = min(range(len(scan_delays)), key=lambda i: abs(scan_delays[i] - delay))
        trace.intermediate_rays.append((ground_ray_time + trace.direction * delay,
                                        trace.ray_y_at(sun_side_x, delay), scan_results[k]))
        delay += step
    return trace
