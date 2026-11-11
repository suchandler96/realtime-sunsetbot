"""Terrain elevation along a transect.

Elevation comes from the free AWS "terrarium" tiles (global DEM mosaic, no key needed): 256 x 256 px PNG
tiles whose RGB encodes the height in metres as  R * 256 + G + B / 256 - 32768.  Tiles are cached on disk,
so a repeated request for the same area needs no network, and the most recent ones also stay decoded in memory
(a map traces many neighbouring transects).

`read_elevation()` is the one call `main.py` needs.
"""

from __future__ import annotations

import functools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

from aerosol import _rate, _size, _StatusLine

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "terrain_tiles")
TILE_URLS = ["https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png",
             "http://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"]   # tried in this order
ZOOM = 9  # ~300 m per pixel at the equator, ~150 m at 60 degrees: finer than the 1 km sampling
TILE_SIZE = 256
TILE_CACHE_TILES = 1024     # decoded tiles kept in memory per process (256 kB each as float32,
# i.e., up to 256 MB per process): neighbouring transects reuse them
USER_AGENT = "realtime-sunsetbot terrain reader"


def _tile_path(zoom, x, y, cache_dir):
    return os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")


def _download_tile(zoom, x, y, attempts=4):
    """The tile's PNG response.  Each URL scheme is tried in turn (HTTPS, then HTTP): some networks / proxies refuse
    one of them (a 403 or a dropped connection), so a refusal moves on to the other scheme, and the scheme that
    worked is tried first from then on.  A dropped connection, timeout or 5xx is retried with a growing pause (a
    single hiccup must not cost a whole group of observers their terrain)."""
    problem = None
    for template in list(TILE_URLS):
        url = template.format(z=zoom, x=x, y=y)
        for attempt in range(attempts):
            try:
                response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
                if response.status_code == 200:
                    if TILE_URLS[0] != template:   # remember what works on this network
                        TILE_URLS.remove(template)
                        TILE_URLS.insert(0, template)
                    return response
                problem = f"HTTP {response.status_code} for {url}"
                if response.status_code < 500:
                    break   # this scheme refuses the request: try the other one
            except (requests.ConnectionError, requests.Timeout) as exc:
                problem = f"{type(exc).__name__} for {url}"
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DEM tile {zoom}/{x}/{y} could not be downloaded ({problem})")


# Tiles a prefetch could not download: reading them fails at once instead of trying the network again.  Worker
# processes do not inherit this at run time (forkserver / spawn start methods): hand it to them explicitly
# (see `vividness_map._init_worker`).
UNAVAILABLE_TILES: set[tuple[int, int, int]] = set()


def _ensure_tile(zoom, x, y, cache_dir):
    """Path of the tile's PNG in the disk cache, downloading it first if it is not there yet.  The temporary file
    has a name of its own (process + thread), so concurrent downloads of one tile cannot trip over each other."""
    path = _tile_path(zoom, x, y, cache_dir)
    if not os.path.exists(path):
        if (zoom, x, y) in UNAVAILABLE_TILES:
            raise RuntimeError(f"DEM tile {zoom}/{x}/{y} is unavailable (its download failed earlier in this run)")
        response = _download_tile(zoom, x, y)
        os.makedirs(cache_dir, exist_ok=True)
        temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.part"
        with open(temporary, "wb") as f:
            f.write(response.content)
        os.replace(temporary, path)
    return path


@functools.lru_cache(maxsize=TILE_CACHE_TILES)
def _load_tile(zoom, x, y, cache_dir):
    """Elevation (m) of one tile as a (256, 256) array; downloaded once, then read from the disk cache and, decoded,
    kept in memory (the returned array is shared: callers must not modify it)."""
    path = _ensure_tile(zoom, x, y, cache_dir)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    # float32 keeps ~0.001 m at 9 km height (24-bit mantissa), plenty for a DEM, and halves the memory per tile
    return (rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0).astype(np.float32)


def _pixel_coordinates(lats, lons, zoom):
    """Global pixel coordinates (float) of the points in the Web-Mercator tile grid."""
    lats = np.clip(np.asarray(lats, dtype=float), -85.05, 85.05)
    lons = (np.asarray(lons, dtype=float) + 180.0) % 360.0 - 180.0
    scale = (1 << zoom) * TILE_SIZE
    gx = (lons + 180.0) / 360.0 * scale
    gy = (1.0 - np.arcsinh(np.tan(np.radians(lats))) / np.pi) / 2.0 * scale
    return gx, gy, scale


def _corners(lats, lons, zoom):
    """The four DEM pixels around every point as (pixel x, pixel y, bilinear weight) triples, and the grid size."""
    gx, gy, scale = _pixel_coordinates(lats, lons, zoom)
    gx, gy = gx - 0.5, gy - 0.5  # pixel centres
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    corners = [(x0 + dx, np.clip(y0 + dy, 0, int(scale) - 1), (1 - fx if dx == 0 else fx) * (1 - fy if dy == 0 else fy))
               for dx in (0, 1) for dy in (0, 1)]
    return corners, int(scale)


def _tile_keys(corners, scale):
    keys = set()
    for cx, cy, _ in corners:
        keys.update(zip(((cx % scale) // TILE_SIZE).tolist(), (cy // TILE_SIZE).tolist()))
    return keys


def tiles_needed(lats, lons, zoom=ZOOM):
    """{(tile x, tile y)} that `read_elevation` needs for these points."""
    corners, scale = _corners(lats, lons, zoom)
    return _tile_keys(corners, scale)


def prefetch_tiles(point_sets, cache_dir=CACHE_DIR, zoom=ZOOM, workers=1, log=print, progress=True):
    """Download every DEM tile the given (lats, lons) point sets need and the disk cache lacks, in a thread pool.

    Meant to run once in the main process before many worker processes read terrain: tiles are then all on disk,
    nothing is downloaded concurrently by several processes, and a repeated run finds them all.  A tile that cannot
    be downloaded is reported and skipped (the observers needing it fail later with the reason); returns those tiles."""
    needed = set()
    for lats, lons in point_sets:
        needed |= tiles_needed(lats, lons, zoom)
    missing = sorted(t for t in needed if not os.path.exists(_tile_path(zoom, t[0], t[1], cache_dir)))
    log(f"Terrain: {len(needed)} DEM tiles needed, {len(needed) - len(missing)} cached, {len(missing)} to download")
    if not missing:
        return []
    started, failed = time.time(), []
    state = {"done": 0, "bytes": 0}
    lock = threading.Lock()
    status = _StatusLine(period=5.0) if progress else None

    def fetch(tile):
        size = 0
        try:
            size = os.path.getsize(_ensure_tile(zoom, tile[0], tile[1], cache_dir))
        except Exception as exc:
            failed.append((tile, exc))
        with lock:
            state["done"] += 1
            state["bytes"] += size
            if status is not None:
                elapsed = max(time.time() - started, 1e-3)
                eta = elapsed / state["done"] * (len(missing) - state["done"])
                status.show(f"Terrain: {state['done']}/{len(missing)} tiles  {_size(state['bytes'])} "
                            f"{'MB' if state['bytes'] >= 1e6 else 'KB'}  {_rate(state['bytes'] / elapsed)}  "
                            f"ETA {eta:.0f} s" + (f"  {len(failed)} failed" if failed else ""))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, missing))
    if status is not None:
        status.clear()
    log(f"Terrain: {len(missing) - len(failed)} tiles downloaded in {time.time() - started:.0f} s"
        + (f"; {len(failed)} FAILED, e.g. {failed[0][1]}" if failed else ""))
    UNAVAILABLE_TILES.update((zoom, x, y) for (x, y), _ in failed)
    return [tile for tile, _ in failed]


def read_elevation(lats, lons, cache_dir=CACHE_DIR, zoom=ZOOM):
    """Terrain height above sea level (km) at the given points; the sea (negative DEM values) counts as 0.

    Bilinear interpolation between DEM pixels. Raises on download problems; the caller decides what to do."""
    corners, scale = _corners(lats, lons, zoom)
    gx_len = len(corners[0][0])
    needed = _tile_keys(corners, scale)
    n_tiles = 1 << zoom
    missing = [t for t in needed if not os.path.exists(_tile_path(zoom, t[0] % n_tiles, t[1], cache_dir))
               and (zoom, t[0], t[1]) not in UNAVAILABLE_TILES]   # known-failed tiles were reported by the prefetch
    if missing:
        print(f"Terrain: downloading {len(missing)} DEM tile(s) ({len(needed)} needed) ...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        tiles = dict(zip(sorted(needed), pool.map(lambda t: _load_tile(zoom, t[0], t[1], cache_dir), sorted(needed))))

    height_m = np.zeros(gx_len)
    for cx, cy, weight in corners:
        px, py = cx % scale, cy
        tx, ty = px // TILE_SIZE, py // TILE_SIZE
        values = np.empty(gx_len)
        for key in set(zip(tx.tolist(), ty.tolist())):
            sel = (tx == key[0]) & (ty == key[1])
            values[sel] = tiles[key][py[sel] % TILE_SIZE, px[sel] % TILE_SIZE]
        height_m += weight * values
    return np.maximum(height_m, 0.0) / 1000.0
