"""Geography and metadata of the NMC regional radar-mosaic pictures.

NMC publishes one composite-reflectivity picture per region.  Each picture has
its own size, map/legend split, projection and (sometimes) an inset that must
be ignored.  Everything that is a property of *a map* lives here as a
`RadarMap`; `radar.py` only does download / decoding / sampling.

To add or fix a region, edit `MAPS` at the bottom:
  * `bbox`  - rough (lon_min, lon_max, lat_min, lat_max) in degrees, used to
              decide which pictures a transect needs;
  * `gcps`  - ground control points `(col, row, lon, lat)`: pixel centres of
              city markers.  The pixel <-> lat/lon mapping is a Lambert
              conformal projection (`lcc_n`, `lcc_lon0`) whose scale and offset
              are least-squares fitted to the GCPs.  A map without GCPs is
              "uncalibrated" and is skipped when sampling.
  * `base_file` - path of the radar-free base picture (same size as the
              mosaic) subtracted before decoding; None until one is made.
  * `masks` - pixel rectangles `(col0, row0, col1, row1)` (end exclusive) that
              are not part of the mosaic, e.g. the South-China inset map.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

_CODE_RE = re.compile(r"_ECREF_([A-Z]{4})_")


@dataclass(frozen=True)
class RadarMap:
    code: str                                  # region code in the NMC filename, e.g. "AECN"
    name: str
    size: tuple[int, int]                      # picture (width, height) in pixels
    map_rows: int                              # rows [0, map_rows) are map; below is legend
    bbox: tuple[float, float, float, float]    # (lon_min, lon_max, lat_min, lat_max)
    lcc_n: float = 0.45
    lcc_lon0: float = 120.0
    gcps: tuple[tuple[float, float, float, float], ...] = ()
    masks: tuple[tuple[int, int, int, int], ...] = ()
    base_file: str | None = None               # radar-free base picture; None = colour-only decoding
    _fit: tuple[float, float, float] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.gcps:
            object.__setattr__(self, "_fit", self._fit_projection())

    # ---- calibration ----------------------------------------------------
    def _rho_theta(self, lats, lons):
        rho = 1.0 / np.tan(np.pi / 4 + np.radians(lats) / 2) ** self.lcc_n
        theta = self.lcc_n * np.radians(np.asarray(lons) - self.lcc_lon0)
        return rho, theta

    def _fit_projection(self) -> tuple[float, float, float]:
        """Least-squares (scale, col0, row0) so that
        col = col0 + scale*rho*sin(theta),  row = row0 + scale*rho*cos(theta)."""
        cols, rows, lons, lats = np.array(self.gcps, dtype=float).T
        n = len(cols)
        rho, theta = self._rho_theta(lats, lons)
        a = np.zeros((2 * n, 3))
        a[:n, 0], a[:n, 1] = rho * np.sin(theta), 1
        a[n:, 0], a[n:, 2] = rho * np.cos(theta), 1
        (scale, col0, row0), *_ = np.linalg.lstsq(a, np.r_[cols, rows], rcond=None)
        return float(scale), float(col0), float(row0)

    @property
    def calibrated(self) -> bool:
        return self._fit is not None

    def fit_rms_px(self) -> float:
        """RMS pixel error of the projection at the GCPs."""
        cols, rows, lons, lats = np.array(self.gcps, dtype=float).T
        c, r = self.pixel_xy(lats, lons)
        return float(np.sqrt(np.mean((c - cols) ** 2 + (r - rows) ** 2)))

    # ---- geometry --------------------------------------------------------
    def pixel_xy(self, lats, lons) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) in degrees -> (column, row) in picture pixels."""
        if self._fit is None:
            raise ValueError(f"radar map {self.code} has no calibration (gcps)")
        scale, col0, row0 = self._fit
        rho, theta = self._rho_theta(np.asarray(lats, dtype=float), np.asarray(lons, dtype=float))
        return col0 + scale * rho * np.sin(theta), row0 + scale * rho * np.cos(theta)

    def contains(self, lats, lons) -> np.ndarray:
        """Boolean array: which points fall inside the (rough) bounding box."""
        lon_min, lon_max, lat_min, lat_max = self.bbox
        lats, lons = np.asarray(lats, dtype=float), np.asarray(lons, dtype=float)
        return (lons >= lon_min) & (lons <= lon_max) & (lats >= lat_min) & (lats <= lat_max)

    def mask_array(self, shape: tuple[int, int]) -> np.ndarray:
        """Boolean (rows, cols) array, True where the picture is *not* mosaic."""
        out = np.zeros(shape, dtype=bool)
        for c0, r0, c1, r1 in self.masks:
            out[r0:r1, c0:c1] = True
        return out

    def filename_pattern(self) -> str:
        return f"SEVP_AOC_RDCP_SLDAS3_ECREF_{self.code}_L88_PI_"


# ==================== registry ====================
MAPS: dict[str, RadarMap] = {m.code: m for m in (
    RadarMap(
        code="AECN", name="East China", size=(774, 1326), map_rows=1205,
        bbox=(111.9, 126.1, 21.0, 38.7),
        lcc_n=0.42, lcc_lon0=120.25,
        gcps=(
            (143.5, 41.5, 114.51, 38.04), (278.5, 142.5, 117.00, 36.67),
            (107.5, 558.5, 114.30, 30.59), (278.5, 478.5, 117.23, 31.82),
            (140.5, 139.5, 114.54, 36.61), (122.5, 266.5, 114.31, 34.80),
            (204.5, 64.5, 115.67, 37.74), (140.5, 109.5, 114.50, 37.07),
            (14.5, 887.5, 113.01, 25.77), (29.5, 747.5, 113.15, 27.83),
            (100.5, 389.5, 114.02, 33.01), (103.5, 230.5, 113.93, 35.30),
            (130.5, 175.5, 114.39, 36.10), (164.5, 200.5, 115.03, 35.76),
            (195.5, 295.5, 115.66, 34.41), (94.5, 318.5, 113.85, 34.04),
            (101.5, 451.5, 114.07, 32.13), (187.5, 237.5, 115.48, 35.23),
            (248.5, 227.5, 116.59, 35.41), (278.5, 173.5, 117.09, 36.20),
            (217.5, 154.5, 115.99, 36.46), (284.5, 313.5, 117.18, 34.26),
            (199.5, 402.5, 115.81, 32.89), (200.5, 334.5, 115.78, 33.84),
            (267.5, 351.5, 116.98, 33.63), (385.5, 359.5, 119.02, 33.61),
            (450.5, 587.5, 120.15, 30.27), (194.5, 695.5, 115.86, 28.68),
            (270.5, 567.5, 117.06, 30.54), (312.5, 540.5, 117.81, 30.93),
            (238.5, 483.5, 116.52, 31.73), (267.5, 425.5, 117.00, 32.63),
            (273.5, 655.5, 117.18, 29.27), (317.5, 713.5, 117.94, 28.45),
            (263.5, 724.5, 117.07, 28.24), (222.5, 746.5, 116.36, 27.95),
            (132.5, 889.5, 114.93, 25.83), (137.5, 801.5, 114.99, 27.11),
            (106.5, 750.5, 114.42, 27.82), (136.5, 752.5, 114.92, 27.82),
            (296.5, 864.5, 117.64, 26.27), (358.5, 959.5, 118.68, 24.87),
            (323.5, 986.5, 118.09, 24.48), (298.5, 983.5, 117.65, 24.51),
            (257.5, 945.5, 117.02, 25.08), (483.5, 1009.5, 120.67, 24.15),
            (461.5, 1114.5, 120.30, 22.63), (74.5, 1130.5, 114.17, 22.28),
            (34.5, 1134.6, 113.54, 22.20), (50.5, 1078.5, 113.75, 23.02),
            (10.5, 1076.5, 113.12, 23.02), (26.5, 1110.5, 113.39, 22.52),
            (7.5, 1105.5, 113.08, 22.58), (92.5, 1074.5, 114.42, 23.11),
            (47.5, 955.5, 113.60, 24.81), (201.5, 997.5, 116.12, 24.29),
            (112.5, 1031.5, 114.70, 23.74), (8.5, 1030.5, 113.05, 23.68),
            (231.5, 1042.5, 116.62, 23.66), (215.5, 1048.5, 116.37, 23.55),
            (151.5, 1098.5, 115.38, 22.78), (20.5, 1068.5, 113.26, 23.13),
            (520.5, 519.5, 121.47, 31.23), (365.5, 461.5, 118.78, 32.06),
            (397.5, 875.5, 119.30, 26.08), (535.5, 946.5, 121.56, 25.04),
            (85.5, 268.5, 113.65, 34.76),
        ),
        base_file="images/base_AECN.png",
    ),
    RadarMap(
        code="ASCN", name="South China", size=(990, 959), map_rows=838,
        bbox=(103.7, 118.0, 15.5, 26.5),
        lcc_n=0.45, lcc_lon0=111.5,
        gcps=(
            (316.5, 295.5, 108.37, 22.82), (732.5, 337.5, 114.17, 22.28),
            (687.5, 344.6, 113.54, 22.20), (702.5, 279.5, 113.75, 23.02),
            (656.5, 279.5, 113.12, 23.02), (689.5, 337.5, 113.58, 22.27),
            (677.5, 317.5, 113.39, 22.52), (655.5, 313.5, 113.08, 22.58),
            (750.5, 272.5, 114.42, 23.11), (691.5, 138.5, 113.60, 24.81),
            (771.5, 222.5, 114.70, 23.74), (652.5, 227.5, 113.05, 23.68),
            (610.5, 277.5, 112.47, 23.05), (576.5, 368.5, 111.98, 21.86),
            (499.5, 384.5, 110.92, 21.66), (459.5, 415.5, 110.36, 21.27),
            (819.5, 297.5, 115.38, 22.78), (580.5, 287.5, 112.04, 22.92),
            (526.5, 243.5, 111.28, 23.48), (372.5, 399.5, 109.12, 21.48),
            (336.5, 360.5, 108.65, 21.98), (316.5, 382.5, 108.35, 21.69),
            (445.5, 307.5, 110.15, 22.65), (404.5, 272.5, 109.60, 23.11),
            (544.5, 170.5, 111.57, 24.40), (377.5, 222.5, 109.22, 23.75),
            (402.5, 549.5, 109.58, 19.52), (667.5, 270.5, 113.26, 23.13),
            (447.5, 510.5, 110.33, 20.04), (447.5, 103.5, 110.29, 25.27),
            (398.5, 645.5, 109.50, 18.25),
        ),
        # South-China-Sea inset map in the lower-left corner (frame included).
        masks=((0, 505, 261, 838),),
        base_file="images/base_ASCN.png",
    ),
    # Calibrated from city-marker dots (see module docstring); fitted n, lon0 differ per picture.
    RadarMap(
        code="ANCN", name="North China", size=(954, 949), map_rows=828,
        bbox=(105.0, 125.0, 33.0, 46.0),
        lcc_n=0.56, lcc_lon0=113.13,
        gcps=(
            (197.5, 681.5, 108.94, 34.34), (440.5, 662.5, 113.62, 34.75),
            (486.5, 446.5, 114.51, 38.04), (618.5, 377.5, 117.20, 39.13),
            (834.5, 379.5, 121.61, 38.91), (665.5, 339.5, 118.18, 39.63),
            (487.5, 539.5, 114.54, 36.61), (378.5, 669.5, 112.45, 34.62),
            (232.5, 535.5, 109.49, 36.59), (105.5, 672.5, 107.24, 34.36),
            (90.5, 756.5, 107.02, 33.07), (226.5, 672.5, 109.51, 34.50),
            (334.5, 573.5, 111.52, 36.09), (415.5, 566.5, 113.12, 36.20),
            (401.5, 612.5, 112.85, 35.49), (304.5, 640.5, 111.01, 35.03),
            (440.5, 458.5, 113.58, 37.86), (732.5, 317.5, 119.60, 39.94),
            (602.5, 428.5, 116.84, 38.30), (544.5, 466.5, 115.67, 37.74),
            (486.5, 511.5, 114.50, 37.07), (891.5, 231.5, 122.99, 41.11),
            (933.5, 176.5, 123.96, 41.88), (802.5, 239.5, 121.13, 41.10),
            (919.5, 155.5, 123.73, 42.22), (823.5, 176.5, 121.67, 42.02),
            (246.5, 714.5, 109.94, 33.87), (184.5, 680.5, 108.71, 34.33),
            (456.5, 626.5, 113.93, 35.30), (479.5, 574.5, 114.39, 36.10),
            (512.5, 596.5, 115.03, 35.76), (416.5, 726.5, 113.19, 33.74),
            (314.5, 658.5, 111.20, 34.77), (420.5, 630.5, 113.24, 35.24),
            (535.5, 630.5, 115.48, 35.23), (665.5, 525.5, 118.05, 36.81),
            (560.5, 550.5, 115.99, 36.46), (659.5, 487.5, 117.97, 37.38),
        ),
        base_file="images/base_ANCN.png",
    ),
    RadarMap(
        code="ANEC", name="Northeast China", size=(936, 983), map_rows=862,
        bbox=(118.0, 136.0, 38.0, 54.0),
        lcc_n=0.54, lcc_lon0=126.42,
        gcps=(
            (451.5, 567.5, 125.32, 43.82), (499.5, 459.5, 126.53, 45.80),
            (153.5, 784.5, 118.18, 39.63), (37.5, 815.5, 115.46, 38.87),
            (150.5, 710.5, 117.96, 40.95), (27.5, 707.5, 114.89, 40.77),
            (212.5, 772.5, 119.60, 39.94), (91.5, 853.5, 116.84, 38.30),
            (92.5, 784.0, 116.70, 39.52), (194.5, 640.5, 118.89, 42.26),
            (331.5, 572.5, 122.24, 43.65), (256.5, 265.5, 119.77, 49.21),
            (402.5, 371.5, 123.92, 47.35), (442.5, 414.5, 125.10, 46.59),
            (499.5, 567.5, 126.55, 43.84), (355.5, 715.5, 122.99, 41.11),
            (395.5, 673.5, 123.96, 41.88), (279.5, 713.5, 121.13, 41.10),
            (386.5, 653.5, 123.73, 42.22), (255.5, 682.5, 120.45, 41.57),
            (303.5, 661.5, 121.67, 42.02), (619.5, 527.5, 129.63, 44.55),
            (643.5, 402.5, 130.32, 46.80), (668.5, 484.5, 130.97, 45.30),
            (516.5, 413.5, 126.97, 46.64), (433.5, 494.5, 124.83, 45.14),
            (358.5, 466.5, 122.84, 45.62), (474.5, 683.5, 125.94, 41.73),
            (443.5, 619.5, 125.14, 42.89), (616.5, 617.5, 129.51, 42.89),
            (640.5, 370.5, 130.30, 47.35),
        ),
        base_file="images/base_ANEC.png",
    ),
    RadarMap(
        code="ASWC", name="Southwest China", size=(971, 818), map_rows=697,
        bbox=(96.0, 112.0, 20.0, 36.0),
        lcc_n=0.4, lcc_lon0=102.71,
        gcps=(
            (752.5, 10.5, 108.94, 34.34), (661.5, 255.5, 106.55, 29.56),
            (938.5, 308.5, 112.94, 28.23), (676.5, 77.5, 107.02, 33.07),
            (570.5, 178.5, 104.40, 31.13), (542.5, 257.5, 103.77, 29.55),
            (581.5, 297.5, 104.63, 28.77), (615.5, 290.5, 105.44, 28.87),
            (598.5, 256.5, 105.06, 29.58), (664.5, 210.5, 106.63, 30.46),
            (511.5, 235.5, 103.01, 29.98), (546.5, 233.5, 103.85, 30.05),
            (453.5, 406.5, 101.72, 26.58), (478.5, 340.5, 102.27, 27.89),
            (389.5, 389.5, 100.23, 26.87), (386.5, 453.5, 100.27, 25.61),
            (337.5, 475.5, 99.17, 25.11), (377.5, 538.5, 100.09, 23.88),
            (415.5, 594.5, 100.97, 22.83), (488.5, 517.5, 102.54, 24.35),
            (544.5, 461.5, 103.80, 25.49), (526.5, 567.5, 103.40, 23.36),
            (565.5, 567.5, 104.24, 23.37), (638.5, 422.5, 105.95, 26.25),
            (778.5, 344.5, 109.19, 27.72), (727.5, 403.5, 107.98, 26.58),
            (708.5, 420.5, 107.52, 26.26), (900.5, 203.5, 112.24, 30.33),
            (882.5, 272.5, 111.70, 29.03), (829.5, 271.5, 110.48, 29.12),
            (813.5, 350.5, 110.00, 27.57), (878.5, 362.5, 111.47, 27.24),
            (927.5, 376.5, 112.57, 26.89), (951.5, 429.5, 113.01, 25.77),
            (887.5, 403.5, 111.61, 26.42), (759.5, 94.5, 109.03, 32.69),
        ),
        base_file="images/base_ASWC.png",
    ),
    RadarMap(
        code="ANWC", name="Northwest China", size=(971, 818), map_rows=697,
        bbox=(80.0, 110.0, 31.0, 49.0),
        lcc_n=0.62, lcc_lon0=102.99,
        gcps=(
            (118.5, 79.5, 87.62, 43.83), (651.5, 471.5, 103.83, 36.06),
            (576.5, 445.5, 101.78, 36.62), (732.5, 356.5, 106.23, 38.49),
            (840.5, 543.5, 108.94, 34.34), (955.5, 368.5, 112.55, 37.87),
            (968.5, 519.5, 112.45, 34.62), (852.5, 438.5, 109.49, 36.59),
            (775.5, 545.5, 107.24, 34.36), (721.5, 537.5, 105.72, 34.58),
            (607.5, 385.5, 102.64, 37.93), (531.5, 337.5, 100.45, 38.93),
            (788.5, 481.5, 107.64, 35.71), (755.5, 493.5, 106.67, 35.54),
            (737.5, 470.5, 106.28, 36.00), (856.5, 358.5, 109.73, 38.29),
            (927.5, 455.5, 111.52, 36.09), (912.5, 505.5, 111.01, 35.03),
            (959.5, 343.5, 112.73, 38.42), (908.5, 389.5, 111.14, 37.52),
            (305.5, 146.5, 93.51, 42.82), (163.5, 126.5, 89.18, 42.95),
            (679.5, 492.5, 104.63, 35.58), (737.5, 332.5, 106.38, 39.02),
            (698.5, 402.5, 105.20, 37.50), (878.5, 562.5, 109.94, 33.87),
            (921.5, 517.5, 111.20, 34.77),
        ),
        base_file="images/base_ANWC.png",
    ),
)}


def code_from_filename(path: str) -> str | None:
    m = _CODE_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def map_for_file(path: str) -> RadarMap | None:
    """The `RadarMap` a picture file belongs to (from its filename), or None."""
    code = code_from_filename(path)
    return MAPS.get(code) if code else None


def maps_for_points(lats, lons) -> list[RadarMap]:
    """Maps whose bounding box contains any of the points, most points first.

    Includes uncalibrated maps so callers can warn about them.
    """
    scored = [(int(m.contains(lats, lons).sum()), m) for m in MAPS.values()]
    return [m for k, m in sorted(scored, key=lambda e: -e[0]) if k > 0]
