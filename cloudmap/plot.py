"""Draw Himawari visible + infrared crops with the sun light path on top."""

from __future__ import annotations

import math
from datetime import datetime

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from .gibs import fetch_pair
from .irpalette import colorbar_parts, windy_style

R_EARTH = 6371.0
# Cloud tops further than this along the beam rarely matter, and the box needs a margin around the line.
PAD_KM = 120


def _point(lat, lon, dist_km, azimuth):
    b, d = math.radians(azimuth), dist_km / R_EARTH
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def light_path_bbox(lat, lon, azimuth, x_min, x_max, pad_km=PAD_KM):
    """(west, south, east, north) around the observer-to-far-end segment, padded by `pad_km`."""
    pts = [_point(lat, lon, d, azimuth) for d in np.linspace(x_min, x_max, 20)]
    lats, lons = zip(*pts)
    dlat = pad_km / 110.57
    dlon = pad_km / (111.32 * math.cos(math.radians(lat)))
    return (min(lons) - dlon, max(-85.0, min(lats) - dlat), max(lons) + dlon, min(85.0, max(lats) + dlat))


def _stretch_visible(rgba):
    """Low-sun visible reflectance is dark: percentile stretch + gamma so clouds stay readable."""
    g = rgba[..., 0].astype(float) / 255
    valid = rgba[..., 3] > 0
    if not valid.any():
        return rgba[..., :3]
    hi = max(np.percentile(g[valid], 99.5), 0.05)
    g = np.clip(g / hi, 0, 1) ** 0.6
    return np.dstack([g, g, g])


def draw_cloud_map(lat, lon, azimuth, when_utc: datetime, title="", x_min=0, x_max=900,
                   save_path=None, show=False, tick_km=100):
    """Fetch Himawari visible/IR around the light path and draw them side by side.

    `x_min/x_max` are km from the observer along `azimuth` (towards the sun). Returns the figure.
    """
    bbox = light_path_bbox(lat, lon, azimuth, x_min, x_max)
    data = fetch_pair(bbox, when_utc)
    west, south, east, north = bbox
    extent = (west, east, south, north)

    fig = plt.figure(figsize=(16, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[1, 1])
    panels = [("visible", "Himawari-9 Band 3 visible (red)"), ("infrared", "Himawari-9 Band 13 infrared (10.4 µm), Windy colours")]
    line_d = np.linspace(x_min, x_max, 200)
    line = np.array([_point(lat, lon, d, azimuth) for d in line_d])
    ticks = [d for d in range(int(math.ceil(x_min / tick_km) * tick_km), int(x_max) + 1, tick_km)]
    for ax, (key, label) in zip(axes, panels):
        img = data[key]
        if img is not None:
            ax.imshow(_stretch_visible(img) if key == "visible" else windy_style(img[..., :3]), extent=extent,
                      origin="upper", interpolation="nearest" if key == "infrared" else "bilinear")
            if data["coast"] is not None:
                ax.imshow(data["coast"], extent=extent, origin="upper")
        else:
            ax.set_facecolor("#222")
            ax.text(0.5, 0.5, "no data (night?)" if key == "visible" else "no data",
                    color="w", ha="center", va="center", transform=ax.transAxes)
        ax.plot(line[:, 1], line[:, 0], color="#ff2a2a", lw=1.8, zorder=5)
        for d in ticks:
            tlat, tlon = _point(lat, lon, d, azimuth)
            ax.plot(tlon, tlat, "o", ms=4, color="#ffd400", mec="k", mew=0.6, zorder=6)
            if d:
                ax.annotate(f"{d} km", (tlon, tlat), xytext=(6, 5), textcoords="offset points",
                            fontsize=9, color="#ffd400", fontweight="bold", zorder=6,
                            path_effects=[pe.withStroke(linewidth=2, foreground="k")])
        ax.plot(lon, lat, "*", ms=14, color="w", mec="k", zorder=7)
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect(1 / math.cos(math.radians(lat)))
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(label, fontsize=11)
    if data["infrared"] is not None:
        cmap, norm = colorbar_parts()
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        cb.set_label("Brightness temperature (°C)")
    else:
        cax.axis("off")
    scan = f"{data['time']:%Y-%m-%d %H:%M} UTC" if data["time"] else "no scan found"
    fig.suptitle(f"{title}  |  scan {scan}  |  light path az {azimuth % 360:.1f}°  (NASA GIBS)", fontsize=13,
                 fontweight="bold")
    if save_path:
        fig.savefig(save_path, dpi=110)
        print(f"Cloud map saved to {save_path}")
    if show:
        plt.show()
    return fig
