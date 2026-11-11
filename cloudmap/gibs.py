"""Fetch cropped Himawari-9 imagery (NASA GIBS WMS) for a lat/lon box.

GIBS serves the same Himawari AHI data as JAXA/JMA, 10-minute cadence, in EPSG:4326.
Visible = Band 3 (red, 1 km); infrared = Band 13 (clean longwave window, 2 km).
A time with no scan yields a fully transparent image, which we treat as "missing".
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from PIL import Image

WMS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
LAYERS = {
    "visible": "Himawari_AHI_Band3_Red_Visible_1km",
    "infrared": "Himawari_AHI_Band13_Clean_Infrared",
}
COASTLINES = "Coastlines_15m"
LATENCY = timedelta(minutes=20)   # newest scans are not on GIBS immediately


def _get(layer: str, bbox, size, when: datetime | None, timeout=60) -> np.ndarray | None:
    """One GetMap call -> RGBA uint8 array (H, W, 4), or None if the server has nothing."""
    params = {
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap", "LAYERS": layer,
        "SRS": "EPSG:4326", "BBOX": ",".join(f"{v:.5f}" for v in bbox),
        "WIDTH": size[0], "HEIGHT": size[1], "FORMAT": "image/png", "TRANSPARENT": "TRUE",
    }
    if when is not None:
        params["TIME"] = when.strftime("%Y-%m-%dT%H:%M:00Z")
    r = requests.get(WMS_URL, params=params, timeout=timeout)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image/"):
        return None
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))


def _has_data(img: np.ndarray | None) -> bool:
    return img is not None and (img[..., 3] > 0).mean() > 0.5


def image_size(bbox, km_per_px=1.5, max_px=1800):
    """Pixel size so that pixels are roughly square on the ground."""
    west, south, east, north = bbox
    mid_lat = np.radians((south + north) / 2)
    w_km = (east - west) * 111.32 * np.cos(mid_lat)
    h_km = (north - south) * 110.57
    scale = min(1.0, max_px / max(w_km, h_km) * km_per_px)
    return max(64, int(w_km / km_per_px * scale)), max(64, int(h_km / km_per_px * scale))


def fetch_pair(bbox, when_utc: datetime, max_back_min=60, km_per_px=1.5):
    """Newest Himawari visible + infrared images at or before `when_utc` (and before now - latency).

    Returns {"time": datetime, "size": (w, h), "visible": RGBA|None, "infrared": RGBA|None,
             "coast": RGBA|None}; "time" is None if nothing was found within `max_back_min`.
    """
    when = min(when_utc.astimezone(timezone.utc), datetime.now(timezone.utc) - LATENCY)
    when = when.replace(minute=when.minute // 10 * 10, second=0, microsecond=0)
    size = image_size(bbox, km_per_px)
    out = {"time": None, "size": size, "visible": None, "infrared": None, "coast": None}
    for back in range(0, max_back_min + 1, 10):
        t = when - timedelta(minutes=back)
        ir = _get(LAYERS["infrared"], bbox, size, t)        # IR exists around the clock: drives the time
        if not _has_data(ir):
            continue
        vis = _get(LAYERS["visible"], bbox, size, t)         # absent at night
        out.update(time=t, infrared=ir, visible=vis if _has_data(vis) else None)
        break
    try:
        out["coast"] = _get(COASTLINES, bbox, size, None)
    except requests.RequestException:
        pass
    return out
