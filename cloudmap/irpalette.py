"""Re-colour NASA GIBS Himawari Band 13 imagery with a Windy-style infrared palette.

GIBS ships the brightness temperature already colour-coded (legend: Clean_Longwave_Infrared_Window_Band_H):
grey from +57 to -13 degC, then cyan/blue/green/yellow/red/magenta for colder tops.  We invert that legend back
to a temperature and paint it with a palette that stays grey for warm surfaces and low cloud, and only turns
colourful for the cold high cloud (cirrus / deep convection), with the colours of windy.com's colour band.

Limitation: the legend's very-cold tail (black/grey below about -72 degC) looks like warm grey, so those
(very rare, tropical) pixels are not recovered.
"""

from __future__ import annotations

import numpy as np

# (brightness temperature degC, GIBS RGB) sampled from the GIBS legend, colour section only.
_GIBS_STOPS = [
    (-90.3, (140, 13, 135)), (-87.8, (178, 51, 157)), (-85.3, (217, 89, 180)), (-82.8, (242, 114, 195)),
    (-80.3, (204, 204, 204)), (-77.8, (129, 129, 129)), (-75.3, (54, 54, 54)), (-72.8, (5, 5, 5)),
    (-70.3, (77, 0, 0)), (-67.8, (153, 0, 0)), (-65.3, (204, 0, 0)), (-62.8, (255, 26, 0)),
    (-60.3, (255, 102, 0)), (-57.8, (255, 179, 0)), (-55.3, (255, 230, 0)), (-52.8, (204, 255, 0)),
    (-50.3, (128, 255, 0)), (-47.8, (51, 255, 0)), (-45.3, (0, 255, 0)), (-42.8, (0, 191, 29)),
    (-40.2, (0, 128, 58)), (-37.7, (0, 85, 77)), (-35.2, (0, 21, 105)), (-32.7, (0, 13, 122)),
    (-30.2, (0, 51, 143)), (-27.7, (0, 76, 157)), (-25.2, (0, 115, 178)), (-22.7, (0, 153, 199)),
    (-20.2, (0, 178, 213)), (-17.7, (0, 217, 234)), (-15.2, (0, 255, 255)),
]
# Grey section: (grey level, brightness temperature degC).
_GIBS_GREY = [(192, -12.7), (158, -0.2), (122, 12.3), (88, 24.9), (52, 37.4), (16, 49.9), (3, 54.9)]

# Windy-style palette: (brightness temperature degC, RGB 0-255), warm -> cold.
# Colour section sampled from a screenshot of windy.com's infrared colour band (-33 ... -73 degC);
# the grey section (warm surface / low cloud, Windy leaves it grey-white) is my own.
_PALETTE = [
    (40, (30, 30, 30)), (-10, (205, 205, 205)), (-22, (235, 235, 240)),
    (-27, (65, 67, 168)), (-33, (65, 67, 168)), (-35.2, (67, 89, 198)), (-38, (67, 116, 230)),
    (-41, (51, 161, 231)), (-44, (39, 186, 202)), (-47, (50, 231, 169)), (-49, (54, 245, 144)),
    (-52, (119, 249, 96)), (-54.3, (165, 228, 55)), (-57, (218, 218, 57)), (-60, (242, 182, 50)),
    (-62.6, (240, 132, 35)), (-65.4, (228, 91, 21)), (-68.2, (206, 49, 5)), (-71, (161, 24, 3)),
    (-73.8, (117, 4, 3)), (-90, (70, 0, 0)),
]

_BT = np.array([b for b, _ in _GIBS_STOPS])
_RGB = np.array([c for _, c in _GIBS_STOPS], float)
# Dense legend (0.25 degC steps) for nearest-colour lookup.
# the legend's grey tail (-81..-71 degC) is skipped below: it is indistinguishable from warm grey
_DENSE_BT = np.arange(_BT.min(), _BT.max() + 0.01, 0.25)
_DENSE_BT = _DENSE_BT[(_DENSE_BT < -81) | (_DENSE_BT > -71)]
_DENSE_RGB = np.stack([np.interp(_DENSE_BT, _BT, _RGB[:, i]) for i in range(3)], axis=1)
_PAL_BT = np.array([b for b, _ in _PALETTE][::-1], float)   # np.interp wants increasing x
_PAL_RGB = np.array([c for _, c in _PALETTE][::-1], float)


def brightness_temperature(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 GIBS image -> (H, W) float degC."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3).astype(int)
    grey = flat.max(1) - flat.min(1) < 12                      # unsaturated => grey section
    bt = np.empty(len(flat))
    gl = np.array([g for g, _ in _GIBS_GREY], float)[::-1]
    gt = np.array([t for _, t in _GIBS_GREY], float)[::-1]
    bt[grey] = np.interp(flat[grey].mean(1), gl, gt)
    colored = np.flatnonzero(~grey)
    if colored.size:
        packed = flat[colored, 0] * 65536 + flat[colored, 1] * 256 + flat[colored, 2]
        uniq, inv = np.unique(packed, return_inverse=True)
        u_rgb = np.stack([uniq >> 16, (uniq >> 8) & 255, uniq & 255], axis=1).astype(float)
        best = np.empty(len(uniq), int)
        for i in range(0, len(uniq), 2000):                    # chunked nearest-colour search
            d = ((u_rgb[i:i + 2000, None, :] - _DENSE_RGB[None]) ** 2).sum(2)
            best[i:i + 2000] = d.argmin(1)
        bt[colored] = _DENSE_BT[best][inv]
    return bt.reshape(h, w)


def windy_style(rgb: np.ndarray) -> np.ndarray:
    """Recolour a GIBS Band 13 image; returns (H, W, 3) float in 0-1."""
    bt = brightness_temperature(rgb)
    out = np.stack([np.interp(bt, _PAL_BT, _PAL_RGB[:, i]) for i in range(3)], axis=-1)
    return out / 255.0


def colorbar_parts(lo=-90.0, hi=40.0):
    """(ListedColormap, Normalize) of the palette over [lo, hi] degC for a colour bar."""
    from matplotlib.colors import ListedColormap, Normalize
    bt = np.linspace(lo, hi, 256)
    rgb = np.stack([np.interp(bt, _PAL_BT, _PAL_RGB[:, i]) for i in range(3)], axis=-1) / 255.0
    return ListedColormap(rgb), Normalize(lo, hi)
