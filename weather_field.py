"""Weather on a model grid, sampled afterwards at any point.

A vividness map needs the atmosphere along thousands of transects.  Instead of asking a data source for those
points one by one, the map first downloads the *model grid cells* it needs for one UTC hour (a `GridField`, the
cells' values only), and then samples the field wherever it integrates: `nearest` takes the value of the closest
grid cell, `bilinear` blends the four cells around the point.  What is downloaded depends only on the model grid
(and on the area), not on how finely the map is drawn or how the transects are laid out.

  * `Grid`, `GRIDS`        the regular lat/lon grid of each model (cell centres at multiples of the spacing)
  * `GridField`            the values of the needed cells; `sample()` reads them at arbitrary points
  * `FieldBackend`         a `WeatherBackend` that serves a field, so everything downstream (`WeatherRequest`,
                           `WeatherResult`, `sunset_core`) is unchanged
  * `OpenMeteoFieldProvider` downloads the cells from Open-Meteo (through the quota-paced backend); AWS GFS has
                           its own provider in `gfs_aws.py`
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from weather_data import DataSource, WeatherBackend, WeatherData, WeatherResult

SAMPLING_METHODS = ("nearest", "bilinear")


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid: cell (i, j) is centred on (lat0 + i * dlat, lon0 + j * dlon) degrees."""
    dlat: float
    dlon: float
    lat0: float = 0.0
    lon0: float = 0.0

    def centre(self, i, j):
        return np.round(self.lat0 + np.asarray(i) * self.dlat, 6), np.round(self.lon0 + np.asarray(j) * self.dlon, 6)


# Spacing of the model grids (cell centres are multiples of it).  GFS, ECMWF IFS 0.25 and CMA GRAPES: their names /
# the Open-Meteo documentation (https://open-meteo.com/en/docs); ICON global: measured on the Open-Meteo API (values
# change every 0.125 deg, at odd multiples of 0.0625 deg), although the documentation speaks of ~11 km; JMA MSM:
# the model's own grid, 0.05 deg x 0.0625 deg (not verified here).  HRRR is on a projected grid: not a lat/lon grid.
GRIDS: dict[DataSource, Grid] = {
    DataSource.GFS: Grid(0.25, 0.25),
    DataSource.GFS_AWS: Grid(0.25, 0.25),
    DataSource.ECMWF: Grid(0.25, 0.25),
    DataSource.ICON: Grid(0.125, 0.125),
    DataSource.CMA_GFS: Grid(0.125, 0.125, lat0=0.0625),
    DataSource.JMA_MSM: Grid(0.05, 0.0625),
}

_OFFSET, _STRIDE = 20000, 1 << 16     # cell (i, j) -> one sortable integer key; |i|, |j| < 20000 cells


def encode(i, j) -> np.ndarray:
    return (np.asarray(i, dtype=np.int64) + _OFFSET) * _STRIDE + (np.asarray(j, dtype=np.int64) + _OFFSET)


def decode(keys) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(keys, dtype=np.int64)
    return keys // _STRIDE - _OFFSET, keys % _STRIDE - _OFFSET


def _corners(grid: Grid, lats, lons, method: str):
    """[(i, j, weight)] of the cells a sample at each point is made of (arrays over the points)."""
    if method not in SAMPLING_METHODS:
        raise ValueError(f"sampling must be one of {', '.join(SAMPLING_METHODS)}, got {method!r}")
    fi = (np.asarray(lats, dtype=float) - grid.lat0) / grid.dlat
    fj = (np.asarray(lons, dtype=float) - grid.lon0) / grid.dlon
    if method == "nearest":
        return [(np.rint(fi).astype(np.int64), np.rint(fj).astype(np.int64), np.ones(len(fi)))]
    i0, j0 = np.floor(fi).astype(np.int64), np.floor(fj).astype(np.int64)
    wi, wj = fi - i0, fj - j0
    return [(i0, j0, (1 - wi) * (1 - wj)), (i0, j0 + 1, (1 - wi) * wj),
            (i0 + 1, j0, wi * (1 - wj)), (i0 + 1, j0 + 1, wi * wj)]


def needed_cells(grid: Grid, lats, lons, method: str) -> np.ndarray:
    """Sorted unique keys of the cells needed to sample the points with `method`."""
    corners = _corners(grid, lats, lons, method)
    keys = np.concatenate([encode(i, j)[w > 0] for i, j, w in corners])
    return np.unique(keys)


def cell_centres(grid: Grid, keys) -> tuple[np.ndarray, np.ndarray]:
    return grid.centre(*decode(keys))


class GridField:
    """Values of some cells of a model grid: `data[variable]` is (levels, cells, times), `elevation_m` (cells,) the
    model's own terrain height.  `keys` (sorted) identify the cells; cells that were not loaded are NaN when sampled."""

    def __init__(self, grid: Grid, keys, data: dict[str, np.ndarray], elevation_m, pressure_levels, times_utc,
                 model_run: datetime | None = None):
        order = np.argsort(keys)
        self.grid = grid
        self.keys = np.asarray(keys, dtype=np.int64)[order]
        self.data = {name: np.asarray(a, dtype=np.float32)[:, order, :] for name, a in data.items()}
        self.elevation_m = np.asarray(elevation_m, dtype=np.float32)[order]
        self.pressure_levels = [int(p) for p in pressure_levels]
        self.times_utc = list(times_utc)
        self.model_run = model_run

    def __len__(self) -> int:
        return len(self.keys)

    def _positions(self, i, j) -> np.ndarray:
        """Index of each cell in the field, -1 where it was not loaded."""
        wanted = encode(i, j)
        if not len(self.keys):
            return np.full(len(wanted), -1)
        pos = np.minimum(np.searchsorted(self.keys, wanted), len(self.keys) - 1)
        return np.where(self.keys[pos] == wanted, pos, -1)

    def _blend(self, array: np.ndarray, corners) -> np.ndarray:
        """`array` is (..., cells, ...) along axis 1; returns the weighted mean at the points (NaN corners and missing
        cells are left out and the remaining weights renormalised; all missing: NaN)."""
        total = weight_sum = None
        for i, j, w in corners:
            pos = self._positions(i, j)
            values = array[:, np.maximum(pos, 0), :].astype(np.float64)
            weight = np.where((pos >= 0) & (w > 0), w, 0.0)[None, :, None] * np.isfinite(values)
            values = np.where(weight > 0, values, 0.0)
            total = values * weight if total is None else total + values * weight
            weight_sum = weight if weight_sum is None else weight_sum + weight
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight_sum > 0, total / weight_sum, np.nan)

    def sample(self, lats, lons, method: str = "bilinear"):
        """Values at the points: ({variable: (levels, points, times)}, terrain height in m (points,))."""
        corners = _corners(self.grid, lats, lons, method)
        out = {name: self._blend(a, corners) for name, a in self.data.items()}
        elevation = self._blend(self.elevation_m[None, :, None], corners)[0, :, 0]
        return out, elevation


class FieldBackend(WeatherBackend):
    """Serves a `GridField` as if it were a weather service: the requested points are sampled from the field."""

    def __init__(self, field: GridField, method: str = "bilinear"):
        self.field, self.method = field, method
        self.model_run = field.model_run

    def fetch(self, request) -> list[dict | None]:
        raise NotImplementedError("FieldBackend delivers its result directly (fetch_result)")

    def fetch_result(self, request) -> WeatherResult:
        field = self.field
        missing = [v for v in request._variables if v not in field.data]
        if missing:
            raise ValueError(f"the weather field has no variable {', '.join(missing)}")
        levels = [field.pressure_levels.index(p) for p in request._pressure_levels]
        values, elevation = field.sample(request._latitudes, request._longitudes, self.method)
        zone = ZoneInfo(request._timezone)
        times = [t.astimezone(zone).replace(tzinfo=None).isoformat(timespec="minutes") for t in field.times_utc]
        arrays = {v: values[v][levels] for v in request._variables}
        return WeatherResult.from_arrays(request, times, arrays, elevation / 1000.0, self.model_run)


# ==================== Open-Meteo ====================
class OpenMeteoFieldProvider:
    """Cells of an Open-Meteo model, downloaded as the map asks for them (`backend`: usually the quota-paced one,
    which also caches on disk).  Open-Meteo answers a point with the whole day, so a cell is downloaded once however
    many hours need it; `field(hour, keys)` only asks for the cells it does not hold yet."""

    cloud_water = False

    def __init__(self, source: DataSource, backend, day: str, tz: str, variables, pressure_levels, log=print):
        self.source, self.backend, self.day, self.tz = source, backend, day, tz
        self.variables, self.levels, self.log = list(variables), list(pressure_levels), log
        self.grid = GRIDS[source]
        self.plan_only = getattr(backend, "plan_only", False)
        self._keys: list[np.ndarray] = []
        self._data: dict[str, list[np.ndarray]] = {v: [] for v in self.variables}
        self._elevation: list[np.ndarray] = []
        self._times_utc: list[datetime] | None = None
        self._have: set[int] = set()
        self._lock = threading.Lock()

    def _download(self, keys: np.ndarray) -> None:
        lats, lons = cell_centres(self.grid, keys)
        request = (WeatherData.source(self.source, backend=self.backend)
                   .coordinates(lats, lons).pressure_levels(self.levels).variables(*self.variables)
                   .on(self.day).timezone(self.tz))
        result = request.fetch()
        zone = ZoneInfo(self.tz)
        times = [t.replace(tzinfo=zone).astimezone(timezone.utc) for t in result._time_objects]
        if self._times_utc is None:
            self._times_utc = times
        elif times != self._times_utc:
            raise RuntimeError("Open-Meteo returned different times for different cells")
        self._keys.append(keys)
        for v in self.variables:
            self._data[v].append(result.values(v).astype(np.float32))
        self._elevation.append((result.model_elevation_km * 1000.0).astype(np.float32))
        self._have.update(int(k) for k in keys)

    def field(self, hour: datetime | None, keys: np.ndarray) -> GridField:
        with self._lock:
            missing = np.array([k for k in keys if int(k) not in self._have], dtype=np.int64)
            if len(missing):
                self._download(missing)
            elif self.plan_only:
                self.backend._report(0, 0, 0, [], 1.0)   # nothing to download: still end the dry run
            return GridField(self.grid, np.concatenate(self._keys),
                             {v: np.concatenate(self._data[v], axis=1) for v in self.variables},
                             np.concatenate(self._elevation), self.levels, self._times_utc)
