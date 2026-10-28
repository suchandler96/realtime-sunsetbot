"""Terrain elevation along a transect.

Elevation comes from the free AWS "terrarium" tiles (global DEM mosaic, no key needed): 256 x 256 px PNG
tiles whose RGB encodes the height in metres as  R * 256 + G + B / 256 - 32768.  Tiles are cached on disk,
so a repeated request for the same area needs no network.

`read_elevation()` is the one call `main.py` needs.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

CACHE_DIR = "terrain_tiles"
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ZOOM = 9  # ~300 m per pixel at the equator, ~150 m at 60 degrees: finer than the 1 km sampling
TILE_SIZE = 256
USER_AGENT = "realtime-sunsetbot terrain reader"


def _tile_path(zoom, x, y, cache_dir):
    return os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")


def _load_tile(zoom, x, y, cache_dir):
    """Elevation (m) of one tile as a (256, 256) array; downloaded once, then read from the cache."""
    path = _tile_path(zoom, x, y, cache_dir)
    if not os.path.exists(path):
        response = requests.get(TILE_URL.format(z=zoom, x=x, y=y), headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        os.makedirs(cache_dir, exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(response.content)
        os.replace(path + ".part", path)
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


def _pixel_coordinates(lats, lons, zoom):
    """Global pixel coordinates (float) of the points in the Web-Mercator tile grid."""
    lats = np.clip(np.asarray(lats, dtype=float), -85.05, 85.05)
    lons = (np.asarray(lons, dtype=float) + 180.0) % 360.0 - 180.0
    scale = (1 << zoom) * TILE_SIZE
    gx = (lons + 180.0) / 360.0 * scale
    gy = (1.0 - np.arcsinh(np.tan(np.radians(lats))) / np.pi) / 2.0 * scale
    return gx, gy, scale


def read_elevation(lats, lons, cache_dir=CACHE_DIR, zoom=ZOOM):
    """Terrain height above sea level (km) at the given points; the sea (negative DEM values) counts as 0.

    Bilinear interpolation between DEM pixels. Raises on download problems; the caller decides what to do."""
    gx, gy, scale = _pixel_coordinates(lats, lons, zoom)
    gx, gy = gx - 0.5, gy - 0.5  # pixel centres
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - x0, gy - y0
    n_tiles = 1 << zoom
    corners = [(x0 + dx, np.clip(y0 + dy, 0, int(scale) - 1), (1 - fx if dx == 0 else fx) * (1 - fy if dy == 0 else fy))
               for dx in (0, 1) for dy in (0, 1)]

    needed = set()
    for cx, cy, _ in corners:
        needed.update(zip(((cx % int(scale)) // TILE_SIZE).tolist(), (cy // TILE_SIZE).tolist()))
    missing = [t for t in needed if not os.path.exists(_tile_path(zoom, t[0] % n_tiles, t[1], cache_dir))]
    if missing:
        print(f"Terrain: downloading {len(missing)} DEM tile(s) ({len(needed)} needed) ...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        tiles = dict(zip(sorted(needed), pool.map(lambda t: _load_tile(zoom, t[0], t[1], cache_dir), sorted(needed))))

    height_m = np.zeros(len(gx))
    for cx, cy, weight in corners:
        px, py = cx % int(scale), cy
        tx, ty = px // TILE_SIZE, py // TILE_SIZE
        values = np.empty(len(gx))
        for key in set(zip(tx.tolist(), ty.tolist())):
            sel = (tx == key[0]) & (ty == key[1])
            values[sel] = tiles[key][py[sel] % TILE_SIZE, px[sel] % TILE_SIZE]
        height_m += weight * values
    return np.maximum(height_m, 0.0) / 1000.0
