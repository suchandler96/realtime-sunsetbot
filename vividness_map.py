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

Data: one batched fetch per forecast hour for AWS GFS (single-time backend).  Open-Meteo models (GFS, ECMWF,
JMA_MSM, ...) rate-limit big batches, so they go through openmeteo_paced: distinct model columns only, spread over
as many minutes as the quota needs, saved in om_cache/ (a re-run or an interrupted run continues from it), and the
map is drawn once everything has arrived.  `--om-plan-only` prints the estimate without downloading.

    python vividness_map.py --region AECN --event sunset --date 2026-11-12 --data-source GFS_AWS --save
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np

import sunlines
import sunset_core as core
import terrain
from weather_data import DataSource

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


def fetch_grouped(transects, source, log=print, om_backend=None):
    """Batched weather for many transects: AWS GFS (one forecast hour per request) gets one batch per hour of the
    event time, Open-Meteo a single batch (`om_backend`: e.g. the quota-paced `PacedOpenMeteoBackend`, which spreads
    it over as many minutes as the free tier needs).  Returns (weather per transect, data source actually used)."""
    if source is not DataSource.GFS_AWS:
        return core.fetch_weather_batch(transects, source, log=log, openmeteo_backend=om_backend)
    groups: dict[datetime, list[int]] = {}
    for i, t in enumerate(transects):
        hour = sunlines.snapshot_hour(t.sun.time)
        groups.setdefault(hour, []).append(i)
    weathers = [None] * len(transects)
    used = source
    for hour, members in sorted(groups.items()):
        log(f"Fetching {len(members)} profiles for events around {hour:%H}:00 UTC ...")
        batch, used_here = core.fetch_weather_batch([transects[i] for i in members], source, valid_at=hour, log=log,
                                                    openmeteo_backend=om_backend)
        used = used_here if used_here is not source else used
        for i, w in zip(members, batch):
            weathers[i] = w
    return weathers, used


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


def observer_transect(group_transect: core.Transect, observer: sunlines.Observer, x_max: float) -> core.Transect:
    """The observer's own transect as a slice of the group's: same columns from its position on, its own sun."""
    step = core.STEP_KM
    first = int(round(observer.offset_km / step))
    distances = np.arange(0, x_max + step, step)
    city = PointCity(f"{observer.lat:.2f},{observer.lon:.2f}", observer.lat, observer.lon, group_transect.timezone)
    return core.Transect(
        city=city, date=group_transect.date, event=group_transect.event, sun=observer.sun,
        r_earth=core.earth_radius_km(observer.lat), distances=distances,
        lats=group_transect.lats[first:first + len(distances)], lons=group_transect.lons[first:first + len(distances)],
        x_min=0, x_max=x_max, step_km=step)


def _score_group_job(job):
    """Scores of all observers of one group, from the group's transect and weather (one entry per observer)."""
    group, weather, x_max = job
    shared = _SharedTerrain(group.transect)
    results = []
    for observer in group.observers:
        try:
            transect = observer_transect(group.transect, observer, x_max)
            first = int(round(observer.offset_km / core.STEP_KM))
            sub_weather = weather.subset(slice(first, first + len(transect.distances)))
            terrain_km = lambda o=observer: shared.slice_for(o.offset_km, x_max)
            results.append(_score_observer(transect, sub_weather, terrain_km))
        except Exception as exc:
            results.append(f"{type(exc).__name__}: {exc}")
    return results


def compute_grid(lat_axis, lon_axis, day: str, event: str, tz: str, source: DataSource, x_max: float = core.X_MAX,
                 jobs: int = 1, log=print, om_backend=None):
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
    started = time.time()
    weathers, used = fetch_grouped(transects, source, log=log, om_backend=om_backend)
    log(f"Weather fetched in {time.time() - started:.0f} s; tracing rays ...")

    started = time.time()
    work = [(t, w) for t, w in zip(transects, weathers)]
    if jobs > 1 and len(work) >= POOL_MIN_POINTS:
        with _make_pool(jobs) as pool:
            results = list(pool.map(_score_job, work, chunksize=4))
    else:
        results = [_score_job(w) for w in work]
    log(f"Rays traced in {time.time() - started:.0f} s")

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
                     jobs: int = 1, log=print, om_backend=None):
    """Score observers laid out along sun lines (see `sunlines`): one weather/terrain transect per group of
    observers.  Returns a dict of 1-D arrays (lat, lon, line, azimuth, score, cover_percent, window_mean,
    window_start_min, one entry per observer), the data source used and the model-run label."""
    single_time = source is DataSource.GFS_AWS    # one forecast hour per request: snapshots must not be mixed
    started = time.time()
    groups = sunlines.build_layout(bbox, day, event, tz, cross_km, along_km, share_km, x_max, single_time, log=log)
    if not groups:
        raise SystemExit("no observers: the region has no sun-facing edge or the sun does not rise/set there")
    n_observers = sum(len(g.observers) for g in groups)
    columns = sum(len(g.transect.lats) for g in groups)
    alone = n_observers * len(np.arange(0, x_max + core.STEP_KM, core.STEP_KM))
    log(f"{n_observers} observers on {len({o.line for g in groups for o in g.observers})} sun lines in {len(groups)} "
        f"groups; {columns} model columns instead of {alone} ({alone / columns:.1f}x fewer) "
        f"[layout {time.time() - started:.0f} s]")
    prefetch_terrain([g.transect for g in groups], _SharedTerrain.fine_step_km(), jobs, log)
    started = time.time()
    weathers, used = fetch_grouped([g.transect for g in groups], source, log=log, om_backend=om_backend)
    log(f"Weather fetched in {time.time() - started:.0f} s; tracing rays ...")

    started = time.time()
    work = [(g, w, x_max) for g, w in zip(groups, weathers)]
    if jobs > 1 and n_observers >= POOL_MIN_POINTS:
        with _make_pool(jobs) as pool:
            results = list(pool.map(_score_group_job, work, chunksize=4))
    else:
        results = [_score_group_job(w) for w in work]
    log(f"Rays traced in {time.time() - started:.0f} s")

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


def _open_figure(radar_map, base_png: str):
    import matplotlib.pyplot as plt
    image = plt.imread(base_png)[:radar_map.map_rows]   # the rows below hold the radar legend
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width / 100 + 1.2, height / 100 + 0.6), dpi=100)
    ax.imshow(image)
    return fig, ax, width, height


def _close_figure(fig, ax, mappable, width, height, title: str, path: str | None):
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
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
             show_points: bool = True):
    from matplotlib.colors import ListedColormap

    fig, ax, width, height = _open_figure(radar_map, base_png)
    lat2d, lon2d = np.meshgrid(lat_axis, lon_axis, indexing="ij")
    cols, rows = radar_map.pixel_xy(lat2d, lon2d)
    shown = np.ma.masked_where(~np.isfinite(score) | (score < MIN_SHOWN_SCORE), score)
    cmap = ListedColormap(RAMP)
    mesh = ax.pcolormesh(cols, rows, shown, cmap=cmap, vmin=0, vmax=100, alpha=0.72, shading="nearest")
    if show_points:
        ax.plot(cols.ravel(), rows.ravel(), ".", color="#444444", markersize=1.5, alpha=0.5)
    return _close_figure(fig, ax, mesh, width, height, title, path)


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
               path: str | None = None, show_points: bool = True):
    """Map of sun-line observers: each is a cell rotated to its sun azimuth.  Cells of neighbouring lines may
    overlap where the lines converge (and leave gaps where they diverge); the higher score is drawn on top."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap, Normalize

    fig, ax, width, height = _open_figure(radar_map, base_png)
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
    return _close_figure(fig, ax, cells, width, height, title, path)


# ==================== command line ====================
def parse_args():
    p = argparse.ArgumentParser(description="Draw a sunrise/sunset vividness map on a radar base map.")
    p.add_argument("--region", default="AECN", help="radar map code of images/base_<code>.png (see radar_map_cn.MAPS)")
    p.add_argument("--bbox", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                   help="grid extent in degrees (default: the region's bounding box)")
    p.add_argument("--layout", default="sunlines", choices=["sunlines", "grid"],
                   help="sunlines: observers along lines that follow the sun azimuth, sharing cross-sections "
                        "(see sunlines.py); grid: a lat/lon grid, one cross-section per point")
    p.add_argument("--step-deg", type=float, default=1.0, help="grid layout: spacing in degrees")
    p.add_argument("--cross-km", type=float, default=20.0, help="sunlines layout: spacing between the lines (km)")
    p.add_argument("--along-km", type=float, default=20.0,
                   help="sunlines layout: spacing of the observers along a line (a multiple of 20 km)")
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
    p.add_argument("--from-npz", help="redraw from a saved .npz instead of computing")
    p.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="save PNG (+ .npz of the scores)")
    p.add_argument("--silent", action=argparse.BooleanOptionalAction, default=False, help="do not show the figure")
    return p.parse_args()


def main():
    args = parse_args()
    global WINDOW_MIN
    WINDOW_MIN = args.window_min
    import radar_map_cn
    rmap = radar_map_cn.MAPS[args.region]
    if not rmap.calibrated:
        sys.exit(f"radar map {args.region} has no calibration")
    base_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", f"base_{args.region}.png")
    if not os.path.exists(base_png):
        sys.exit(f"missing {base_png}")

    sunlines_layout = args.layout == "sunlines"
    stem = f"{args.date.replace('-', '')}_{args.region}_{args.event}_vividness" + ("_sunlines" if sunlines_layout else "")
    cell_km = (args.along_km, args.cross_km)
    if args.from_npz:
        data = np.load(args.from_npz, allow_pickle=True)
        grids = {k: data[k] for k in data.files}
        source_name, run = str(data["source"]), str(data["run"])
        sunlines_layout = "lat_axis" not in data.files
        if sunlines_layout:
            cell_km = (float(data["along_km"]), float(data["cross_km"]))
    else:
        bbox = tuple(args.bbox or rmap.bbox)
        source = DataSource.__members__[args.data_source]
        om_backend = None
        if source is not DataSource.GFS_AWS:
            from openmeteo_paced import DEFAULT_CACHE_DIR, PacedOpenMeteoBackend
            om_backend = PacedOpenMeteoBackend(
                args.om_cache or DEFAULT_CACHE_DIR, max_age_h=args.om_max_age_h,
                max_wait_s=args.om_max_wait_min * 60.0, plan_only=args.om_plan_only)
        if sunlines_layout:
            grids, used, run = compute_sunlines(bbox, args.date, args.event, args.timezone, source, args.cross_km,
                                                args.along_km, args.share_km, x_max=args.x_max,
                                                jobs=args.jobs, om_backend=om_backend)
            extra = dict(along_km=args.along_km, cross_km=args.cross_km, share_km=args.share_km)
        else:
            lon_min, lon_max, lat_min, lat_max = bbox
            eps = args.step_deg * 1e-6
            lat_axis = np.arange(lat_min, lat_max + eps, args.step_deg)
            lon_axis = np.arange(lon_min, lon_max + eps, args.step_deg)
            grids, used, run = compute_grid(lat_axis, lon_axis, args.date, args.event, args.timezone, source,
                                            x_max=args.x_max, jobs=args.jobs, om_backend=om_backend)
            extra = dict(lat_axis=lat_axis, lon_axis=lon_axis)
        source_name = used.name
        if args.save:
            np.savez(f"{stem}.npz", source=source_name, run=run, **extra, **grids)
        grids = {**extra, **grids}
    title = f"{args.date} {args.event} vividness of the local cloud base  ({source_name} {run})"
    out_png = f"{stem}.png" if args.save else None
    if sunlines_layout:
        fig = draw_cells(grids, cell_km[0], cell_km[1], rmap, base_png, title, out_png)
    else:
        fig = draw_map(grids["score"], grids["lat_axis"], grids["lon_axis"], rmap, base_png, title, out_png)
    if not args.silent:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
