"""NMC (China Meteorological Administration) radar-mosaic access.

NMC publishes regional composite-reflectivity mosaics as PNG pictures, not as
numeric data, so this module recovers dBZ from the pictures:

  1. a *base image* (the same map with no radar echoes, assembled by hand from
     several pictures) is subtracted from the new picture, leaving only the
     pixels that radar painted;
  2. each remaining pixel's colour is matched to the legend colour bar
     (5 .. 70 dBZ, one 5 dBZ bin per colour);
  3. with a base image, small and weak echo regions (speckle) are removed
     from the subtracted result (`remove_speckle`);
     regions that are not part of the mosaic (legend, inset maps) are masked
     out as "not covered" (NaN);
  4. pixel (row, col) <-> (lat, lon) comes from `radar_map_cn.RadarMap`, which
     also decides which regional pictures a transect needs.

`read_reflectivity()` is the one call `main.py` needs; it mirrors
`satellite.SatelliteSource.read_cloud_top()`.

Picture file naming (UTC timestamp, a new picture every 6 minutes):

    SEVP_AOC_RDCP_SLDAS3_ECREF_<region>_L88_PI_<YYYYMMDDhhmmss>000.png
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image

import radar_map_cn
from radar_map_cn import RadarMap

URL_TEMPLATE = "http://image.nmc.cn/product/{t:%Y/%m/%d}/RDCP/{name}"
SLOT_MINUTES = 6
IMAGE_DIR = "images"

_TIME_RE = re.compile(r"_PI_(\d{14})\d{3}\.png$", re.IGNORECASE)

# ---- Legend ------------------------------------------------------------
# RGB of each echo class as painted on the map (same in every region; the two
# palest colour-bar swatches differ slightly from the painted echoes); class i covers
# [5 + 5 i, 10 + 5 i) dBZ (last: >= 70).
LEGEND_RGB = np.array([
    (1, 160, 246), (0, 236, 236), (109, 250, 61), (0, 216, 0), (1, 144, 0),
    (255, 255, 0), (231, 192, 0), (255, 144, 0), (255, 0, 0), (214, 0, 0),
    (192, 0, 0), (255, 0, 240), (150, 0, 180), (173, 144, 240),
], dtype=float)
LEGEND_DBZ = 5.0 + 5.0 * np.arange(len(LEGEND_RGB)) + 2.5  # bin centres

BASE_DIFF_TOL = 24.0         # min max-channel difference from the base to count as "painted"
COLOR_TOL_WITH_BASE = 45.0   # max RGB distance to a legend colour (base available)
COLOR_TOL_NO_BASE = 20.0     # stricter without a base: blue sea / rivers look alike

# Speckle filter: an echo region (8-connected pixels with dbz > 0, any class) is
# removed when it is both small and weak.
SPECKLE_MAX_AREA_PX = 3      # region with at most this many pixels is "small"
SPECKLE_WEAK_DBZ = 15.0      # region whose strongest pixel is below this is "weak"


@dataclass
class RadarReflectivity:
    """Composite reflectivity sampled along the observer transect.

    `dbz` is 0 where the radar sees no echo and NaN where no mosaic covers the
    point.  `obs_times_utc` maps region code -> picture time for the pictures used.
    """

    dbz: np.ndarray | None = None
    obs_times_utc: dict[str, datetime] | None = None


# ==================== file selection / download ====================
def _to_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def parse_file_time(path: str) -> datetime | None:
    """Return the UTC time encoded in a picture filename, or None."""
    m = _TIME_RE.search(os.path.basename(path))
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _floor_slot(dt: datetime) -> datetime:
    return dt.replace(minute=dt.minute - dt.minute % SLOT_MINUTES, second=0, microsecond=0)


def _filename(radar_map: RadarMap, t: datetime) -> str:
    return f"{radar_map.filename_pattern()}{t:%Y%m%d%H%M%S}000.png"


def _local_pictures(radar_map: RadarMap, local_dir: str) -> list[tuple[datetime, str]]:
    out = []
    for path in glob.glob(os.path.join(local_dir, radar_map.filename_pattern() + "*")):
        t = parse_file_time(path)
        if t is not None:
            out.append((t, path))
    return out


def _download(radar_map: RadarMap, t: datetime, local_dir: str) -> str | None:
    """Fetch the picture for slot time `t`; None if the server has no such picture."""
    name = _filename(radar_map, t)
    local_path = os.path.join(local_dir, name)
    if os.path.exists(local_path):
        return local_path
    try:
        resp = requests.get(URL_TEMPLATE.format(t=t, name=name.replace(".png", ".PNG")), timeout=20)
    except requests.RequestException:
        return None
    if resp.status_code != 200 or not resp.content.startswith(b"\x89PNG"):
        return None
    os.makedirs(local_dir, exist_ok=True)
    with open(local_path, "wb") as fh:
        fh.write(resp.content)
    return local_path


def select_file(radar_map: RadarMap, event_utc: datetime, now_utc: datetime | None = None,
                local_dir: str = IMAGE_DIR, max_probe: int = 30) -> str | None:
    """Pick (downloading if needed) the picture of one region for an event.

    Same rule as `himawari.select_file`: for a future event use the latest
    picture available now, for a past event use the one closest to the event.
    Already-downloaded pictures in `local_dir` are preferred over new requests.
    """
    event_utc = _to_utc(event_utc)
    now_utc = _to_utc(now_utc or datetime.now(timezone.utc))
    target = now_utc if event_utc > now_utc else event_utc

    local = [(t, p) for t, p in _local_pictures(radar_map, local_dir) if t <= now_utc]
    if local:
        t, p = min(local, key=lambda e: abs((e[0] - target).total_seconds()))
        if abs((t - target).total_seconds()) <= SLOT_MINUTES * 60:
            return p

    slot = _floor_slot(target)
    # Probe outwards from the target slot (backwards only for "latest").
    offsets = range(0, max_probe) if event_utc > now_utc else \
        [k for i in range(max_probe) for k in ((i, -i) if i else (0,))]
    for k in offsets:
        t = slot - timedelta(minutes=SLOT_MINUTES * k)
        if t > now_utc:
            continue
        path = _download(radar_map, t, local_dir)
        if path:
            return path
    if local:  # fall back to whatever is on disk
        return min(local, key=lambda e: abs((e[0] - target).total_seconds()))[1]
    return None


# ==================== picture -> dBZ ====================
def _load_rgb(path: str, radar_map: RadarMap) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    if (arr.shape[1], arr.shape[0]) != radar_map.size:
        raise ValueError(f"{path}: size {arr.shape[1]}x{arr.shape[0]} != {radar_map.code} "
                         f"size {radar_map.size[0]}x{radar_map.size[1]}")
    return arr[:radar_map.map_rows]


def remove_speckle(dbz: np.ndarray, max_area_px: int = SPECKLE_MAX_AREA_PX,
                   weak_dbz: float = SPECKLE_WEAK_DBZ) -> np.ndarray:
    """Zero out echo regions that are small (<= `max_area_px` pixels) *and* weak
    (peak < `weak_dbz`).  Regions are 8-connected groups of pixels with dbz > 0,
    regardless of class; NaN (masked) pixels are left untouched.  A small but
    strong region, or a large weak one, is kept.
    """
    echo = np.nan_to_num(dbz, nan=0.0) > 0
    h, w = echo.shape
    seen = np.zeros_like(echo)
    out = dbz.copy()
    for y0, x0 in zip(*np.nonzero(echo)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        stack, region = [(y0, x0)], []
        while stack:
            y, x = stack.pop()
            region.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and echo[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if len(region) <= max_area_px:
            ys, xs = zip(*region)
            if dbz[ys, xs].max() < weak_dbz:
                out[ys, xs] = 0.0
    return out


def decode(image_path: str, radar_map: RadarMap, base_path: str | None = None) -> np.ndarray:
    """Convert a radar picture to a (map_rows, width) dBZ array.

    0 = no echo, NaN = not covered by the mosaic (masked regions such as
    insets).  Masking is applied after the base subtraction.
    """
    img = _load_rgb(image_path, radar_map)
    base = _load_rgb(base_path, radar_map) if base_path else None

    # Nearest legend colour for every pixel.
    dist = np.stack([np.linalg.norm(img - rgb, axis=2) for rgb in LEGEND_RGB])
    nearest = dist.argmin(axis=0)
    best = dist.min(axis=0)

    if base is not None:
        painted = np.abs(img - base).max(axis=2) > BASE_DIFF_TOL
        valid = painted & (best <= COLOR_TOL_WITH_BASE)
    else:
        valid = best <= COLOR_TOL_NO_BASE
    dbz = np.where(valid, LEGEND_DBZ[nearest], 0.0)
    dbz[radar_map.mask_array(dbz.shape)] = np.nan
    # Speckle filtering only on the base-subtracted result (colour-only decoding
    # is already the less reliable path and is left unfiltered).
    return remove_speckle(dbz) if base is not None else dbz


def sample_at(dbz: np.ndarray, radar_map: RadarMap, lats, lons, half_window: int = 3) -> np.ndarray:
    """Max reflectivity in a (2*half_window+1)^2 pixel window around each point.

    A window (default +-3 px ~ +-6 km) rather than one pixel keeps thin echoes
    from being missed by the transect spacing.  Points off the picture, or
    whose whole window is masked, give NaN.
    """
    cols, rows = radar_map.pixel_xy(lats, lons)
    h, w = dbz.shape
    out = np.full(len(cols), np.nan)
    for i, (c, r) in enumerate(zip(np.rint(cols).astype(int), np.rint(rows).astype(int))):
        if 0 <= c < w and 0 <= r < h:
            win = dbz[max(r - half_window, 0):r + half_window + 1,
            max(c - half_window, 0):c + half_window + 1]
            if not np.isnan(win).all():
                out[i] = np.nanmax(win)
    return out


def read_reflectivity(event_utc, lats, lons, now_utc=None,
                      local_file: str | None = None, half_window: int = 3) -> RadarReflectivity:
    """Return radar reflectivity (dBZ) at each (lat, lon) transect point.

    Regions are chosen from the transect: every calibrated `RadarMap` whose
    bounding box contains a transect point is fetched, and each point takes
    the first region (most transect points first) that covers it.

    `half_window` is passed to `sample_at` (0 = the single nearest pixel, for a
    finely spaced transect).

    `local_file` restricts the query to that one picture (its region comes
    from the filename).  Each region's base image is `RadarMap.base_file`;
    without one only strictly legend-coloured pixels are accepted (sea and
    river lines may leak in).
    """
    lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
    if local_file:
        radar_map = radar_map_cn.map_for_file(local_file)
        if radar_map is None:
            raise ValueError(f"cannot tell the radar region from file name {local_file}")
        regions = [(radar_map, local_file)]
    else:
        regions = []
        for radar_map in radar_map_cn.maps_for_points(lats, lons):
            if not radar_map.calibrated:
                print(f"Radar: region {radar_map.code} ({radar_map.name}) overlaps the transect "
                      "but is not calibrated; skipped.")
                continue
            path = select_file(radar_map, event_utc, now_utc)
            if path:
                regions.append((radar_map, path))
            else:
                print(f"Radar: no picture found for region {radar_map.code}.")

    dbz = np.full(len(lats), np.nan)
    times: dict[str, datetime] = {}
    for radar_map, path in regions:
        if not radar_map.calibrated:
            print(f"Radar: region {radar_map.code} is not calibrated; skipped.")
            continue
        base = radar_map.base_file
        if base is not None and not os.path.isabs(base):
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), base)
        if base is None or not os.path.exists(base):
            print(f"Radar: no base image for {radar_map.code}; "
                  "decoding by colour only (less reliable).")
            base = None
        sampled = sample_at(decode(path, radar_map, base), radar_map, lats, lons, half_window)
        dbz = np.where(np.isnan(dbz), sampled, dbz)
        times[radar_map.code] = parse_file_time(path)

    if not times:
        return RadarReflectivity()
    return RadarReflectivity(dbz=dbz, obs_times_utc=times)
