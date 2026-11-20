"""Sunset / sunrise vividness map over a region, drawn on one of the images/base_*.png radar base maps.

For every observer this runs the same ray tracing as main.py (via sunset_core, no figure code) and condenses
it into one 0-100 score of how well the cloud base over that observer is lit, overlaid on the base map.
Observers sit on sun lines that share cross-sections (--layout sunlines, see sunlines.py) or on a lat/lon grid.

Point score (`point_score`), all three factors multiplied:
  * local cloud cover   - cloud cover of the model layer at the cloud base over the observer: a lit cloud needs
                          a canvas (saturates at COVER_SATURATION_PERCENT);
  * transmittance       - direct-beam brightness of each ray (green transmittance, logarithmic: invisible at
                          T_DIM, fully bright from T_BRIGHT);
  * colour              - redness of the transmitted light (red/blue transmittance ratio: white-yellow at
                          RB_MIN, full red-orange from RB_FULL).
Rays are sampled 0.5 min apart; ray quality = brightness x colour; the point's value is the best WINDOW_MIN-minute
sliding-window mean of the ray quality (a family shorter than the window is padded with zeros, so a short glow
scores lower).  A ray blocked by water cloud has quality 0; no cloud over the observer scores 0.
An observer with no level of CLOUD_PRESENT_COVER_PERCENT cloud cover or CLOUD_PRESENT_RH_PERCENT relative humidity
over it is skipped (score 0) before anything else is computed; the local cloud base of the others is found with
main.py's thresholds (sunset_core.CLOUD_BASE_COVER_MIN / CLOUD_BASE_RH_MIN), and an observer without such a
level also scores 0.

Data, one UTC hour at a time (the hour of the observers' events): first the model grid cells the hour's transects
need are downloaded (weather_field.GridField), then the field is sampled at the transects' points (--sampling
nearest | bilinear) and the observers are scored; the download of the next hour overlaps with the scoring of this
one (--overlap).  The map's resolution (--grid-size) does not change what is downloaded.
AWS GFS has no rate limit.  Open-Meteo models (GFS, ECMWF, ICON, JMA_MSM, ...) rate-limit, so they go through
openmeteo_paced: distinct model columns only, spread over as many minutes as the quota needs, saved in om_cache/
(a re-run or an interrupted run continues from it).  `--om-plan-only` prints the estimate without downloading.

    python vividness_map.py --region AECN --event sunset --date 2026-11-12 --data-source GFS_AWS --save
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

import gfs_aws
import sunlines
import sunset_core as core
import terrain
import weather_field
from weather_data import DataSource, OpenMeteoBackend

# ==================== score parameters ====================
CLOUD_PRESENT_COVER_PERCENT = 50  # a level with this cloud cover, used only for masking out regions w/o cloud to save computation
# when searching for local cloud base, the script will still follow the thresholds in main.py
CLOUD_PRESENT_RH_PERCENT = 99.0   # ... or this relative humidity counts as cloud over the observer (main.py: 50 / 99)

COVER_SATURATION_PERCENT = 60.0   # layer cloud cover at which the "canvas" factor reaches 1
T_DIM = 0.001                     # green transmittance (0.1 %) below which a ray adds nothing
T_BRIGHT = 0.02                   # green transmittance (2 %) from which brightness counts fully
RB_MIN = 2.0                      # red/blue transmittance ratio of white-yellow light (colour factor 0)
RB_FULL = 30.0                    # ratio from which the light is fully red-orange (colour factor 1)
WINDOW_MIN = 6.0                  # sliding-window length (minutes): the light must last this long to score fully
# (was 3; a longer window rewards long exposure, --window-min overrides)
RAY_STEP_MIN = core.RAY_SCAN_STEP.total_seconds() / 60.0

POOL_MIN_POINTS = 300             # fewer points than this run in-process: worker start-up and pickling cost more

MIN_SHOWN_SCORE = 1.0             # lower scores are left transparent on the map

# Named sub-regions for --region: name -> (radar map code of the base image that contains it, (lon_min, lon_max,
# lat_min, lat_max)).  The map is drawn on that base image, over the bounding box.  Names are case-insensitive.
REGION_ALIASES = {
    # Hefei (west) to Zhoushan (east), Zhuji (south) to Yancheng (north)
    "YangtzeRiverDelta": ("AECN", (117.2, 122.3, 29.7, 33.4)),
}


def resolve_region(name: str, maps) -> tuple[str, tuple | None]:
    """(radar map code, bounding box or None for the code's own box) of a --region value: a radar map code
    (radar_map_cn.MAPS) or a name of REGION_ALIASES."""
    for alias, (code, bbox) in REGION_ALIASES.items():
        if alias.lower() == name.lower():
            return code, bbox
    for code in maps:
        if code.lower() == name.lower():
            return code, None
    known = ", ".join(list(maps) + list(REGION_ALIASES))
    raise SystemExit(f"unknown region {name!r} (known: {known})")


def _quiet(*_args, **_kwargs):
    return None


# ==================== point score ====================
def ray_quality(result) -> float:
    """0-1 quality of the light of one ray: log-brightness (green) x redness (red/blue); blocked -> 0."""
    if result is None or result.blocked_by_cloud:
        return 0.0
    t_red, t_green, t_blue = result.rgb_transmittance
    if t_green <= 0.0 or t_red <= 0.0:
        return 0.0
    brightness = np.clip((math.log10(t_green) - math.log10(T_DIM)) / (math.log10(T_BRIGHT) - math.log10(T_DIM)),
                         0.0, 1.0)
    redness = math.log(t_red / max(t_blue, 1e-12))
    colour = np.clip((redness - math.log(RB_MIN)) / (math.log(RB_FULL) - math.log(RB_MIN)), 0.0, 1.0)
    return float(brightness * colour)


def best_window_mean(quality, step_min: float = RAY_STEP_MIN, window_min: float | None = None):
    """Highest mean of `quality` over a sliding window of `window_min` minutes.

    Returns (mean, index of the window's first ray).  Fewer samples than the window are padded with zeros."""
    window_min = WINDOW_MIN if window_min is None else window_min   # read at call time, so --window-min applies
    quality = np.asarray(quality, dtype=float)
    size = int(round(window_min / step_min)) + 1   # rays 0.5 min apart: 3 min spans 7 rays
    padded = np.concatenate([quality, np.zeros(max(0, size - len(quality)))])
    means = np.convolve(padded, np.ones(size) / size, mode="valid")
    best = int(np.argmax(means))
    return float(means[best]), best


def local_cloud_cover_percent(profile: core.Profile) -> float:
    """Cloud cover (%) of the first model level at or above the observer's cloud base."""
    obs = profile.transect.observer_index
    heights = profile.height_km[:, obs]
    order = np.argsort(heights)
    heights, cover = heights[order], np.nan_to_num(profile.cloud[:, obs])[order]
    usable = ~profile.underground[order, obs]
    above = np.where(usable & (heights >= profile.local_cbh_km - 1e-6))[0]
    return float(cover[above[0]]) if len(above) else 0.0


@dataclass
class PointScore:
    score: float             # 0-100
    cover_percent: float     # local cloud cover at the cloud base
    cover_factor: float      # 0-1
    window_mean: float       # best WINDOW_MIN-minute mean ray quality, 0-1
    window_start_min: float  # delay (minutes from the event) of the best window's first ray


NO_CLOUD = PointScore(0.0, 0.0, 0.0, 0.0, 0.0)


def point_score(profile: core.Profile, rays: core.RayTrace | None) -> PointScore:
    """Vividness of one observer from its profile and ray trace (see the module docstring)."""
    if rays is None or profile.local_cbh_km is None or rays.extinction_model is None:
        return NO_CLOUD
    cover = local_cloud_cover_percent(profile)
    cover_factor = float(np.clip(cover / COVER_SATURATION_PERCENT, 0.0, 1.0))
    quality = [ray_quality(r) for r in rays.scan_results]
    mean, first = best_window_mean(quality)
    start_min = rays.scan_delays[first].total_seconds() / 60.0
    return PointScore(100.0 * cover_factor * mean, cover, cover_factor, mean, start_min)


# ==================== grid computation ====================
PointCity = sunlines.PointCity   # the few city attributes sunset_core needs, for an arbitrary point


class HourlyWeather:
    """The map's weather, one UTC hour at a time: `load` downloads the model grid cells the transects of that hour
    need (a `weather_field.GridField`), `weathers` samples them at the transects' points (`sampling`: nearest cell or
    bilinear).  What is downloaded depends on the model grid and the area only, not on the map's resolution.
    AWS GFS that cannot be downloaded falls back to Open-Meteo's GFS (`used` tells which source was used)."""

    def __init__(self, source, day: str, tz: str, sampling: str = "bilinear", om_backend=None, log=print):
        self.source, self.used, self.day, self.tz = source, source, day, tz
        self.sampling, self.om_backend, self.log = sampling, om_backend, log
        self.provider = self._make_provider(source)
        self.grid = weather_field.GRIDS.get(source)
        if self.provider is None:
            log(f"{source.value} is not on a lat/lon grid: its points are queried directly (--sampling is ignored)")

    @property
    def plan_only(self) -> bool:
        return bool(getattr(self.om_backend, "plan_only", False))

    def _make_provider(self, source):
        if source is DataSource.GFS_AWS:
            return gfs_aws.GfsAwsProvider(core.PROFILE_VARIABLES, core.PRESSURE_LEVELS)
        if source in weather_field.GRIDS:
            return weather_field.OpenMeteoFieldProvider(
                source, self.om_backend or OpenMeteoBackend(), self.day, self.tz, core.PROFILE_VARIABLES,
                core.PRESSURE_LEVELS, self.log)
        return None

    def cells(self, transects) -> np.ndarray | None:
        """Keys of the grid cells the transects' points are sampled from (None: the source has no grid)."""
        if self.grid is None:
            return None
        lats = np.array([lat for t in transects for lat in t.lats], dtype=float)
        lons = np.array([lon for t in transects for lon in t.lons], dtype=float)
        return weather_field.needed_cells(self.grid, lats, lons, self.sampling)

    def load(self, hour: datetime, cells):
        """Download the cells for one UTC hour; None for a source without a grid."""
        if self.provider is None:
            return None
        try:
            return self.provider.field(hour, cells)
        except Exception as exc:   # network / bucket problems must not kill the map: use Open-Meteo's GFS instead
            if self.used is not DataSource.GFS_AWS:
                raise
            self.log(f"GFS download from AWS failed ({exc}); using Open-Meteo GFS instead (no cloud phase)")
            self.used = DataSource.GFS
            self.provider = self._make_provider(DataSource.GFS)
            return self.provider.field(hour, cells)

    def weathers(self, hour: datetime, transects, loaded) -> list:
        """One `WeatherResult` per transect, sampled from the loaded cells."""
        lats = [lat for t in transects for lat in t.lats]
        lons = [lon for t in transects for lon in t.lons]
        if loaded is None:   # no grid: ask the backend for the points themselves
            weather = core.request_profile(self.used, lats, lons, self.day, self.tz, hour, backend=self.om_backend)
        else:
            weather = core.request_profile(self.used, lats, lons, self.day, self.tz, hour,
                                           backend=weather_field.FieldBackend(loaded, self.sampling),
                                           cloud_water=self.provider.cloud_water)
        return core.split_weather(weather, transects, self.used, self.log)


def score_by_hour(units, transect_of, make_job, worker, hourly: HourlyWeather, jobs: int = 1, parallel: bool = False,
                  overlap: bool = True, log=print):
    """Score the units (transects or groups of observers) hour by hour: the weather cells of one UTC hour are
    downloaded, sampled at the transects' points, and the units of that hour are scored.  With `overlap`, the
    download of the next hour runs in a thread while this hour is being scored.
    Returns (one result per unit, the first unit's weather)."""
    by_hour: dict[datetime, list[int]] = {}
    for k, unit in enumerate(units):
        by_hour.setdefault(sunlines.snapshot_hour(transect_of(unit).sun.time), []).append(k)
    hours = sorted(by_hour)
    cells = {h: hourly.cells([transect_of(units[k]) for k in by_hour[h]]) for h in hours}
    if hourly.plan_only and hourly.grid is not None:   # a dry run plans the whole download, not just the first hour
        hourly.load(hours[0], np.unique(np.concatenate(list(cells.values()))))
    sizes = ", ".join(f"{h:%H}:00 UTC: {len(by_hour[h])}" for h in hours)
    log(f"{len(units)} units in {len(hours)} UTC hours ({sizes}); weather {'overlapped with' if overlap else 'then'} ray tracing")

    results: list = [None] * len(units)
    first_weather = None
    waiting = tracing = 0.0
    pool = _make_pool(jobs) if parallel else None
    loader = ThreadPoolExecutor(1) if overlap else None
    try:
        pending = loader.submit(hourly.load, hours[0], cells[hours[0]]) if loader else None
        for n, hour in enumerate(hours):
            started = time.time()
            loaded = pending.result() if loader else hourly.load(hour, cells[hour])
            waiting += time.time() - started
            if loader and n + 1 < len(hours):
                pending = loader.submit(hourly.load, hours[n + 1], cells[hours[n + 1]])
            members = by_hour[hour]
            started = time.time()
            weathers = hourly.weathers(hour, [transect_of(units[k]) for k in members], loaded)
            first_weather = weathers[0] if first_weather is None else first_weather
            work = [make_job(units[k], w) for k, w in zip(members, weathers)]
            done = list(pool.map(worker, work, chunksize=4)) if pool else [worker(w) for w in work]
            for k, r in zip(members, done):
                results[k] = r
            tracing += time.time() - started
            log(f"{hour:%H}:00 UTC done ({len(members)} units, {time.time() - started:.0f} s)")
    finally:
        if loader:
            loader.shutdown(wait=True)
        if pool:
            pool.shutdown()
    log(f"Weather: waited {waiting:.0f} s for downloads; sampling and ray tracing {tracing:.0f} s")
    return results, first_weather


def prefetch_terrain(transects, step_km: float, workers: int = 1, log=print):
    """Download every DEM tile the transects need (those not cached yet) before any worker process reads terrain.
    All tiles are fetched, also those of cloud-free observers: runs on one region repeat, and a complete tile cache
    makes them download nothing.  `workers` threads (--jobs) in this process do the work, so no two processes fetch
    the same tile."""
    return terrain.prefetch_tiles((core.terrain_points(t, step_km) for t in transects), workers=max(1, workers), log=log)


def _score_observer(transect, weather, terrain_km=None):
    """Score of one observer from its own transect and weather; `terrain_km` as in `core.build_profile`.

    Terrain is always used: it limits the ray that grazes the aerosol top (`core.trace_rays`).  An observer with a
    cloud base whose terrain could not be read fails (counted in the failure summary) rather than being scored as if
    the land were flat."""
    # cheap mask: no cloud at all over the observer (strict thresholds) -> nothing to light, skip the observer
    if not core.cloud_present_over_observer(transect, weather, CLOUD_PRESENT_RH_PERCENT, CLOUD_PRESENT_COVER_PERCENT):
        return NO_CLOUD
    # the local cloud base is searched with main.py's thresholds (core.CLOUD_BASE_*)
    profile = core.build_profile(transect, weather, include_terrain=True, terrain_km=terrain_km,
                                 lazy_terrain=True, log=_quiet)
    if profile.local_cbh_km is not None and profile.terrain_km is None:
        raise RuntimeError("terrain (DEM) could not be read")
    rays = core.trace_rays(profile, log=_quiet)
    return point_score(profile, rays)


def summarize_failures(failures, noun: str, log=print, limit: int = 5):
    """One summary of the failed items (error strings): the count, then each distinct message with how often it
    happened (the `limit` most frequent)."""
    if not failures:
        return
    counts = Counter(str(f) for f in failures)
    log(f"{len(failures)} {noun} failed:")
    for message, count in counts.most_common(limit):
        log(f"  {count} x {message}")
    if len(counts) > limit:
        log(f"  ... and {len(counts) - limit} other messages")


def _init_worker(window_min, unavailable_tiles):
    """Pool initializer: settings the main process fixed at run time (workers start fresh, they do not inherit
    module globals) - the window length and the DEM tiles the prefetch failed to download."""
    global WINDOW_MIN
    WINDOW_MIN = window_min
    terrain.UNAVAILABLE_TILES.update(unavailable_tiles)


def _make_pool(jobs: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(WINDOW_MIN, set(terrain.UNAVAILABLE_TILES)))


def _score_job(job):
    transect, weather = job
    try:
        return _score_observer(transect, weather)
    except Exception as exc:   # one bad point must not kill the map
        return f"{type(exc).__name__}: {exc}"


class _SharedTerrain:
    """DEM heights along a group's whole transect, read once when the first observer needs them.  They are read on
    a grid of `fine_km` (the common divisor of the column and terrain steps, 1 km), so that every observer, who
    stands on a column, finds its own terrain-step samples among them and takes exactly those."""

    @staticmethod
    def fine_step_km() -> int:
        return math.gcd(int(core.STEP_KM), int(core.TERRAIN_STEP_KM))

    def __init__(self, transect: core.Transect):
        self.transect, self.loaded, self.heights, self.error = transect, False, None, ""
        self.fine_km = self.fine_step_km()

    def slice_for(self, offset_km: float, x_max: float):
        if not self.loaded:
            messages = []
            self.heights = core.load_terrain(self.transect, messages.append, step_km=self.fine_km)
            self.error = messages[-1] if messages else ""
            self.loaded = True
        if self.heights is None:   # fail with the cause rather than score the group as if the land were flat
            raise RuntimeError(f"terrain (DEM) could not be read: {self.error}")
        first = int(round(offset_km / self.fine_km))
        every = int(round(core.TERRAIN_STEP_KM / self.fine_km))
        count = len(np.arange(0, x_max + core.TERRAIN_STEP_KM, core.TERRAIN_STEP_KM))
        return self.heights[first::every][:count]


def observer_columns(group_transect: core.Transect, observer: sunlines.Observer, x_max: float) -> np.ndarray:
    """Indices, in the group's columns, of the observer's own columns (every `core.STEP_KM` from its position)."""
    column_km = group_transect.step_km
    first = int(round(observer.offset_km / column_km))
    stride = int(round(core.STEP_KM / column_km))
    return first + stride * np.arange(len(np.arange(0, x_max + core.STEP_KM, core.STEP_KM)))


def observer_transect(group_transect: core.Transect, observer: sunlines.Observer, x_max: float) -> core.Transect:
    """The observer's own transect as a slice of the group's: the columns every `core.STEP_KM` from its position on
    (every k-th of the group's, which may be finer), its own sun."""
    step = core.STEP_KM
    indices = observer_columns(group_transect, observer, x_max)
    distances = np.arange(0, x_max + step, step)
    city = PointCity(f"{observer.lat:.2f},{observer.lon:.2f}", observer.lat, observer.lon, group_transect.timezone)
    return core.Transect(
        city=city, date=group_transect.date, event=group_transect.event, sun=observer.sun,
        r_earth=core.earth_radius_km(observer.lat), distances=distances,
        lats=[group_transect.lats[i] for i in indices], lons=[group_transect.lons[i] for i in indices],
        x_min=0, x_max=x_max, step_km=step)


def _score_group_job(job):
    """Scores of all observers of one group, from the group's transect and weather (one entry per observer)."""
    group, weather, x_max = job
    shared = _SharedTerrain(group.transect)
    results = []
    for observer in group.observers:
        try:
            transect = observer_transect(group.transect, observer, x_max)
            sub_weather = weather.subset(observer_columns(group.transect, observer, x_max))
            terrain_km = lambda o=observer: shared.slice_for(o.offset_km, x_max)
            results.append(_score_observer(transect, sub_weather, terrain_km))
        except Exception as exc:
            results.append(f"{type(exc).__name__}: {exc}")
    return results


def compute_grid(lat_axis, lon_axis, day: str, event: str, tz: str, source: DataSource, x_max: float = core.X_MAX,
                 jobs: int = 1, log=print, om_backend=None, sampling: str = "bilinear", overlap: bool = True):
    """Score every grid point; returns dict of 2-D arrays (score, cover_percent, window_mean, window_start_min,
    azimuth) shaped (lat, lon), plus the data source used and the model-run label."""
    shape = (len(lat_axis), len(lon_axis))
    transects, index = [], []
    for i, lat in enumerate(lat_axis):
        for j, lon in enumerate(lon_axis):
            city = PointCity(f"{lat:.2f},{lon:.2f}", float(lat), float(lon), tz)
            try:
                transects.append(core.make_transect(city, day, event, x_min=0, x_max=x_max))
                index.append((i, j))
            except Exception as exc:   # e.g. no sunset at this latitude
                log(f"skipped ({lat:.2f}, {lon:.2f}): {exc}")
    log(f"{len(transects)} points, {len(transects[0].lats)} profile samples each")
    prefetch_terrain(transects, core.TERRAIN_STEP_KM, jobs, log)

    hourly = HourlyWeather(source, day, tz, sampling, om_backend, log)
    results, first = score_by_hour(transects, lambda t: t, lambda t, w: (t, w), _score_job, hourly, jobs,
                                   parallel=jobs > 1 and len(transects) >= POOL_MIN_POINTS, overlap=overlap, log=log)
    used, weathers = hourly.used, [first]

    out = {name: np.full(shape, np.nan) for name in
           ("score", "cover_percent", "window_mean", "window_start_min", "azimuth")}
    failures = []
    for (i, j), t, r in zip(index, transects, results):
        out["azimuth"][i, j] = t.azimuth
        if isinstance(r, PointScore):
            out["score"][i, j], out["cover_percent"][i, j] = r.score, r.cover_percent
            out["window_mean"][i, j], out["window_start_min"][i, j] = r.window_mean, r.window_start_min
        else:
            failures.append(r)
    summarize_failures(failures, "points", log)
    return out, used, core.model_run_label(used, weathers[0])


def compute_sunlines(bbox, day: str, event: str, tz: str, source: DataSource, cross_km: float = 20.0,
                     along_km: float = 20.0, share_km: float = 100.0, x_max: float = core.X_MAX,
                     jobs: int = 1, log=print, om_backend=None, sampling: str = "bilinear", overlap: bool = True):
    """Score observers laid out along sun lines (see `sunlines`): one weather/terrain transect per group of
    observers.  Returns a dict of 1-D arrays (lat, lon, line, azimuth, score, cover_percent, window_mean,
    window_start_min, one entry per observer), the data source used and the model-run label."""
    started = time.time()
    # one forecast hour per group: weather of different UTC hours is never shared (single_time)
    groups = sunlines.build_layout(bbox, day, event, tz, cross_km, along_km, share_km, x_max, True, log=log)
    if not groups:
        raise SystemExit("no observers: the region has no sun-facing edge or the sun does not rise/set there")
    n_observers = sum(len(g.observers) for g in groups)
    columns = sum(len(g.transect.lats) for g in groups)
    alone = n_observers * len(np.arange(0, x_max + core.STEP_KM, core.STEP_KM))   # columns, one transect each
    log(f"{n_observers} observers on {len({o.line for g in groups for o in g.observers})} sun lines in {len(groups)} "
        f"groups; {columns} model columns instead of {alone} ({alone / columns:.1f}x fewer) "
        f"[layout {time.time() - started:.0f} s]")
    prefetch_terrain([g.transect for g in groups], _SharedTerrain.fine_step_km(), jobs, log)

    hourly = HourlyWeather(source, day, tz, sampling, om_backend, log)
    results, first = score_by_hour(groups, lambda g: g.transect, lambda g, w: (g, w, x_max), _score_group_job, hourly,
                                   jobs, parallel=jobs > 1 and n_observers >= POOL_MIN_POINTS, overlap=overlap, log=log)
    used, weathers = hourly.used, [first]

    names = ("lat", "lon", "line", "azimuth", "group_azimuth", "score", "cover_percent", "window_mean",
             "window_start_min")
    out = {name: np.full(n_observers, np.nan) for name in names}
    failures, k = [], 0
    for g, group_results in zip(groups, results):
        for observer, r in zip(g.observers, group_results):
            out["lat"][k], out["lon"][k], out["line"][k] = observer.lat, observer.lon, observer.line
            out["azimuth"][k], out["group_azimuth"][k] = observer.sun.position.azimuth, g.transect.azimuth
            if isinstance(r, PointScore):
                out["score"][k], out["cover_percent"][k] = r.score, r.cover_percent
                out["window_mean"][k], out["window_start_min"][k] = r.window_mean, r.window_start_min
            else:
                failures.append(r)
            k += 1
    summarize_failures(failures, "observers", log)
    return out, used, core.model_run_label(used, weathers[0])


# ==================== drawing ====================
# Sequential ramp, light (low score) -> dark (high score): yellow -> orange -> red (ColorBrewer YlOrRd, 9 steps)
RAMP = ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"]


CROP_MARGIN = 0.05       # margin around the region of interest when the base map is cropped (fraction of its size)
CROP_MIN_MARGIN_PX = 12
MAX_ZOOM = 3.0           # a small crop is shown up to this much larger than its pixel size


def crop_window(radar_map, bbox, width: int, height: int):
    """Pixel window (col0, col1, row0, row1) of the base image that contains `bbox` (lon_min, lon_max, lat_min,
    lat_max) plus a margin, clipped to the image; the whole image when `bbox` is None."""
    if bbox is None:
        return 0, width, 0, height
    lon_min, lon_max, lat_min, lat_max = bbox
    t = np.linspace(0.0, 1.0, 25)       # the box edges are curved in the map projection: sample them
    lons = np.concatenate([lon_min + t * (lon_max - lon_min)] * 2 + [np.full_like(t, lon_min), np.full_like(t, lon_max)])
    lats = np.concatenate([np.full_like(t, lat_min), np.full_like(t, lat_max)] + [lat_min + t * (lat_max - lat_min)] * 2)
    cols, rows = radar_map.pixel_xy(lats, lons)
    pad_c = max(CROP_MIN_MARGIN_PX, CROP_MARGIN * (cols.max() - cols.min()))
    pad_r = max(CROP_MIN_MARGIN_PX, CROP_MARGIN * (rows.max() - rows.min()))
    c0, c1 = int(max(0, np.floor(cols.min() - pad_c))), int(min(width, np.ceil(cols.max() + pad_c)))
    r0, r1 = int(max(0, np.floor(rows.min() - pad_r))), int(min(height, np.ceil(rows.max() + pad_r)))
    if c1 - c0 < 2 or r1 - r0 < 2:      # the box lies outside the image: show all of it
        return 0, width, 0, height
    return c0, c1, r0, r1


def _open_figure(radar_map, base_png: str, bbox=None):
    """Figure with the base map (cropped to `bbox` plus a margin, if given); returns (fig, ax, window)."""
    import matplotlib.pyplot as plt
    image = plt.imread(base_png)[:radar_map.map_rows]   # the rows below hold the radar legend
    height, width = image.shape[:2]
    c0, c1, r0, r1 = window = crop_window(radar_map, bbox, width, height)
    zoom = float(np.clip(900.0 / max(c1 - c0, r1 - r0), 1.0, MAX_ZOOM)) if window != (0, width, 0, height) else 1.0
    fig, ax = plt.subplots(figsize=(zoom * (c1 - c0) / 100 + 1.2, zoom * (r1 - r0) / 100 + 0.6), dpi=100)
    # pixel coordinates of the full image are kept, so everything is placed with radar_map.pixel_xy as before
    ax.imshow(image[r0:r1, c0:c1], extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5),
              interpolation="bicubic" if zoom > 1 else "antialiased")
    return fig, ax, window


def _close_figure(fig, ax, mappable, window, title: str, path: str | None):
    c0, c1, r0, r1 = window
    ax.set_xlim(c0 - 0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, loc="left")
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.01)
    colorbar.set_label("Vividness (0-100; transparent below %g)" % MIN_SHOWN_SCORE, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=100, bbox_inches="tight")
        print(f"Map saved to {path}")
    return fig


def draw_map(score, lat_axis, lon_axis, radar_map, base_png: str, title: str, path: str | None = None,
             show_points: bool = True, bbox=None):
    from matplotlib.colors import ListedColormap

    fig, ax, window = _open_figure(radar_map, base_png, bbox)
    lat2d, lon2d = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    cols, rows = radar_map.pixel_xy(lat2d, lon2d)
    shown = np.ma.masked_where(~np.isfinite(score) | (score < MIN_SHOWN_SCORE), score)
    cmap = ListedColormap(RAMP)
    mesh = ax.pcolormesh(cols, rows, shown, cmap=cmap, vmin=0, vmax=100, alpha=0.72, shading="nearest")
    if show_points:
        ax.plot(cols.ravel(), rows.ravel(), ".", color="#444444", markersize=1.5, alpha=0.5)
    return _close_figure(fig, ax, mesh, window, title, path)


def cell_corners(lat, lon, azimuth, along_km: float, cross_km: float):
    """Corners (lat, lon), shape (n, 4), of the cells centred on the observers: `along_km` long in the sun direction
    and `cross_km` wide across it (local flat approximation, fine for cells of tens of km)."""
    lat, lon, az = (np.asarray(v, dtype=float) for v in (lat, lon, np.radians(azimuth)))
    east_along, north_along = np.sin(az), np.cos(az)         # unit vector towards the sun (east, north)
    east_cross, north_cross = np.cos(az), -np.sin(az)        # unit vector across it
    km_per_deg_lon = sunlines.KM_PER_DEG * np.cos(np.radians(lat))
    corners_lat, corners_lon = [], []
    for sign_along, sign_cross in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        a, c = sign_along * along_km / 2, sign_cross * cross_km / 2
        corners_lat.append(lat + (a * north_along + c * north_cross) / sunlines.KM_PER_DEG)
        corners_lon.append(lon + (a * east_along + c * east_cross) / km_per_deg_lon)
    return np.stack(corners_lat, axis=1), np.stack(corners_lon, axis=1)


def draw_cells(grids, along_km: float, cross_km: float, radar_map, base_png: str, title: str,
               path: str | None = None, show_points: bool = True, bbox=None):
    """Map of sun-line observers: each is a cell rotated to its sun azimuth.  Cells of neighbouring lines may
    overlap where the lines converge (and leave gaps where they diverge); the higher score is drawn on top."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap, Normalize

    fig, ax, window = _open_figure(radar_map, base_png, bbox)
    score = np.asarray(grids["score"], dtype=float)
    keep = np.isfinite(score) & (score >= MIN_SHOWN_SCORE)
    order = np.argsort(score[keep], kind="stable")            # low scores first, so high ones stay visible
    lat, lon, az = (np.asarray(grids[k], dtype=float)[keep][order] for k in ("lat", "lon", "azimuth"))
    corner_lat, corner_lon = cell_corners(lat, lon, az, along_km, cross_km)
    cols, rows = radar_map.pixel_xy(corner_lat, corner_lon)
    cells = PolyCollection(np.stack([cols, rows], axis=-1), array=score[keep][order], cmap=ListedColormap(RAMP),
                           norm=Normalize(0, 100), alpha=0.72, edgecolors="face", linewidths=0.4)
    ax.add_collection(cells)
    if show_points:
        cols, rows = radar_map.pixel_xy(np.asarray(grids["lat"], dtype=float), np.asarray(grids["lon"], dtype=float))
        ax.plot(cols, rows, ".", color="#444444", markersize=1.5, alpha=0.5)
    return _close_figure(fig, ax, cells, window, title, path)


# ==================== command line ====================
def parse_args():
    p = argparse.ArgumentParser(description="Draw a sunrise/sunset vividness map on a radar base map.")
    p.add_argument("--region", default="AECN",
                   help="radar map code of images/base_<code>.png (see radar_map_cn.MAPS) or a name from "
                        "REGION_ALIASES (e.g. YangtzeRiverDelta: that base map, restricted to the named box)")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                   help="grid extent in degrees (default: the region's bounding box; overrides a region alias's box)")
    p.add_argument("--layout", default="sunlines", choices=["sunlines", "grid"],
                   help="sunlines: observers along lines that follow the sun azimuth, sharing cross-sections "
                        "(see sunlines.py); grid: a lat/lon grid, one cross-section per point")
    p.add_argument("--grid-size", type=float, default=20.0,
                   help="size of the map's cells (km, a whole number): the spacing of the observers, along the sun "
                        "lines and between them (sunlines layout) or along latitude and longitude (grid layout); "
                        "independent of the model grid and of what is downloaded")
    p.add_argument("--sampling", default="bilinear", choices=list(weather_field.SAMPLING_METHODS),
                   help="how the model grid is read at the transects' points: the nearest grid cell, or "
                        "bilinear interpolation of the four around the point")
    p.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True,
                   help="download the weather of the next UTC hour while the current hour is being computed")
    p.add_argument("--share-km", type=float, default=100.0,
                   help="sunlines layout: observers within this distance along a line of a group's first observer "
                        "share its cross-section (the sun azimuth drifts with distance)")
    p.add_argument("--date", default=date.today().strftime("%Y-%m-%d"), help="local date YYYY-MM-DD")
    p.add_argument("--event", default="sunset", choices=["sunrise", "sunset"])
    p.add_argument("--timezone", default="Asia/Shanghai", help="IANA time zone shared by the whole map")
    p.add_argument("--data-source", default=DataSource.GFS_AWS.name, type=str.upper,
                   choices=[s.name for s in DataSource],
                   help="GFS_AWS has no rate limit; Open-Meteo models are downloaded paced over several minutes (om_cache/)")
    p.add_argument("--om-cache", default=None, help="Open-Meteo download cache directory (default: om_cache/)")
    p.add_argument("--om-plan-only", action="store_true", help="Open-Meteo: print the download estimate and exit")
    p.add_argument("--om-max-age-h", type=float, default=6.0,
                   help="Open-Meteo: ignore cached data older than this many hours (model runs change)")
    p.add_argument("--om-max-wait-min", type=float, default=120.0,
                   help="Open-Meteo: stop (progress is kept) rather than wait longer than this for the quota")
    p.add_argument("--x-max", type=float, default=core.X_MAX, help="transect length towards the sun (km)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="worker processes")
    p.add_argument("--window-min", type=float, default=WINDOW_MIN,
                   help="length (min) of the sliding window of ray quality; longer = rewards longer-lasting light")
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                   help="when the region is given as a named alias or --bbox, crop the base map to it (plus a margin)")
    p.add_argument("--show-points", action=argparse.BooleanOptionalAction, default=False,
                   help="draw a gray dot at each sample point (observer) on the map")
    p.add_argument("--from-npz", help="redraw from a saved .npz instead of computing")
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="save PNG (+ .npz of the scores)")
    p.add_argument("--silent", action=argparse.BooleanOptionalAction, default=False, help="do not show the figure")
    return p.parse_args()


def main():
    args = parse_args()
    global WINDOW_MIN
    WINDOW_MIN = args.window_min
    import radar_map_cn
    map_code, region_bbox = resolve_region(args.region, radar_map_cn.MAPS)
    rmap = radar_map_cn.MAPS[map_code]
    if not rmap.calibrated:
        sys.exit(f"radar map {map_code} has no calibration")
    base_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", f"base_{map_code}.png")
    if not os.path.exists(base_png):
        sys.exit(f"missing {base_png}")

    sunlines_layout = args.layout == "sunlines"
    stem_of = lambda src: f"{args.date.replace('-', '')}_{args.region}_{args.event}_vividness_{src}"
    cell_km = (args.grid_size, args.grid_size)
    crop_bbox = None
    if args.from_npz:
        data = np.load(args.from_npz, allow_pickle=True)
        grids = {k: data[k] for k in data.files}
        source_name, run = str(data["source"]), str(data["run"])
        sunlines_layout = "lat_axis" not in data.files
        if sunlines_layout:
            cell_km = (float(data["along_km"]), float(data["cross_km"]))
        if args.crop and "crop_bbox" in data.files and len(data["crop_bbox"]):
            crop_bbox = tuple(float(v) for v in data["crop_bbox"])
    else:
        bbox = tuple(args.bbox or region_bbox or rmap.bbox)
        if args.crop and (args.bbox or region_bbox):    # a sub-region of the base map: show just that part
            crop_bbox = bbox
        source = DataSource.__members__[args.data_source]
        om_backend = None
        if source is not DataSource.GFS_AWS:
            from openmeteo_paced import DEFAULT_CACHE_DIR, PacedOpenMeteoBackend
            om_backend = PacedOpenMeteoBackend(
                args.om_cache or DEFAULT_CACHE_DIR, max_age_h=args.om_max_age_h,
                max_wait_s=args.om_max_wait_min * 60.0, plan_only=args.om_plan_only)
        if sunlines_layout:
            grids, used, run = compute_sunlines(bbox, args.date, args.event, args.timezone, source, args.grid_size,
                                                args.grid_size, args.share_km, x_max=args.x_max,
                                                jobs=args.jobs, om_backend=om_backend, sampling=args.sampling,
                                                overlap=args.overlap)
            extra = dict(along_km=args.grid_size, cross_km=args.grid_size, share_km=args.share_km)
        else:
            lon_min, lon_max, lat_min, lat_max = bbox
            step_lat = args.grid_size / sunlines.KM_PER_DEG                 # degrees of the cell size
            step_lon = step_lat / math.cos(math.radians(0.5 * (lat_min + lat_max)))
            lat_axis = np.arange(lat_min, lat_max + step_lat * 1e-6, step_lat)
            lon_axis = np.arange(lon_min, lon_max + step_lon * 1e-6, step_lon)
            grids, used, run = compute_grid(lat_axis, lon_axis, args.date, args.event, args.timezone, source,
                                            x_max=args.x_max, jobs=args.jobs, om_backend=om_backend,
                                            sampling=args.sampling, overlap=args.overlap)
            extra = dict(lat_axis=lat_axis, lon_axis=lon_axis)
        source_name = used.name
        if args.save:
            np.savez(f"{stem_of(source_name)}.npz", source=source_name, run=run, crop_bbox=np.array(crop_bbox or ()), **extra, **grids)
        grids = {**extra, **grids}
    title = f"{args.date} {args.event} vividness of the local cloud base  ({source_name} {run})"
    out_png = f"{stem_of(source_name)}.png" if args.save else None
    if sunlines_layout:
        fig = draw_cells(grids, cell_km[0], cell_km[1], rmap, base_png, title, out_png,
                         show_points=args.show_points, bbox=crop_bbox)
    else:
        fig = draw_map(grids["score"], grids["lat_axis"], grids["lon_axis"], rmap, base_png, title, out_png,
                       show_points=args.show_points, bbox=crop_bbox)
    if not args.silent:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
