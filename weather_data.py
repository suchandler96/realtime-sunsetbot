"""Fluent data-access layer for GFS and ECMWF pressure-level profiles.

The layer currently talks to the Open-Meteo forecast API, which exposes both
GFS and ECMWF IFS data behind the same endpoint.  A small `DataSource` enum is
the single place a caller has to touch to switch models:

    from weather_data import WeatherData, DataSource

    data = (
        WeatherData.source(DataSource.ECMWF)
        .coordinates(lats, lons)
        .pressure_levels(PRESSURE_LEVELS)
        .relative_humidity()
        .cloud_cover()
        .fetch()
    )

The adapter boundary is `WeatherBackend.fetch()`; if raw GFS/ECMWF downloads are
ever required, a new backend can be added without changing the fluent call-sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Iterable

import numpy as np
import requests


class DataSource(str, Enum):
    """Model identifiers understood by the current backend."""

    GFS = "gfs_seamless"
    ECMWF = "ecmwf_ifs025"
    JMA_MSM = "jma_msm"
    CMA_GFS = "cma_grapes_global"


class WeatherBackend(ABC):
    """Transport boundary for fetching raw point payloads.

    The fluent request only knows about coordinates, levels, variables, and a
    `DataSource`.  New backends (for example native NOMADS GFS or ECMWF S3
    access) can be added by implementing this one method.
    """

    @abstractmethod
    def fetch(self, request: "WeatherRequest") -> list[dict]:
        """Return one raw response object per requested coordinate."""


class OpenMeteoBackend(WeatherBackend):
    """Open-Meteo forecast API backend (supports GFS and ECMWF IFS)."""

    _ENDPOINT = "https://api.open-meteo.com/v1/forecast"

    def fetch(self, request: "WeatherRequest") -> list[dict]:
        lat_str, lon_str = request._format_coordinates()
        params = {
            "latitude": lat_str,
            "longitude": lon_str,
            "hourly": ",".join(request._expanded_hourly_variables()),
            "models": request._source.value,
            "timezone": request._timezone,
        }
        if request._start_date is not None:
            params["start_date"] = request._start_date
            params["end_date"] = request._end_date or request._start_date
        else:
            params["forecast_days"] = request._forecast_days

        response = requests.get(self._ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")

        # A single point is returned as an object; multiple points are a list.
        return payload if isinstance(payload, list) else [payload]


class WeatherData:
    """Convenience entry point for starting a fluent request."""

    @staticmethod
    def gfs(backend: WeatherBackend | None = None) -> "WeatherRequest":
        return WeatherRequest(DataSource.GFS, backend=backend)

    @staticmethod
    def ecmwf(backend: WeatherBackend | None = None) -> "WeatherRequest":
        return WeatherRequest(DataSource.ECMWF, backend=backend)

    @staticmethod
    def source(source: DataSource, backend: WeatherBackend | None = None) -> "WeatherRequest":
        return WeatherRequest(source, backend=backend)


class WeatherRequest:
    """Fluent builder describing a multi-point pressure-level request."""

    def __init__(self, source: DataSource = DataSource.GFS, backend: WeatherBackend | None = None):
        self._source = source
        self._backend = backend or OpenMeteoBackend()
        self._latitudes: list[float] = []
        self._longitudes: list[float] = []
        self._pressure_levels: list[int] = []
        self._variables: list[str] = []
        self._forecast_days = 1
        self._start_date: str | None = None
        self._end_date: str | None = None
        self._timezone = "UTC"

    # ----- data source -------------------------------------------------
    def source(self, source: DataSource) -> "WeatherRequest":
        self._source = source
        return self

    def gfs(self) -> "WeatherRequest":
        self._source = DataSource.GFS
        return self

    def ecmwf(self) -> "WeatherRequest":
        self._source = DataSource.ECMWF
        return self

    # ----- target locations -------------------------------------------
    def coordinates(self, latitude, longitude) -> "WeatherRequest":
        """Set one or more sampling points.

        `latitude`/`longitude` may be scalars or equal-length iterables.
        """
        lats = self._as_float_list(latitude)
        lons = self._as_float_list(longitude)
        if len(lats) != len(lons):
            raise ValueError("latitude and longitude must have the same length")
        if not lats:
            raise ValueError("at least one coordinate is required")
        self._latitudes = lats
        self._longitudes = lons
        return self

    # ----- vertical levels and variables ------------------------------
    def pressure_levels(self, levels: Iterable[int]) -> "WeatherRequest":
        """Set the pressure levels (hPa) for all level-dependent variables."""
        self._pressure_levels = [int(level) for level in levels]
        if not self._pressure_levels:
            raise ValueError("at least one pressure level is required")
        return self

    def variables(self, *names: str) -> "WeatherRequest":
        """Append base variable names (without pressure-level suffixes)."""
        for name in names:
            if name not in self._variables:
                self._variables.append(name)
        return self

    def relative_humidity(self) -> "WeatherRequest":
        return self.variables("relative_humidity")

    def cloud_cover(self) -> "WeatherRequest":
        return self.variables("cloud_cover")

    def temperature(self) -> "WeatherRequest":
        return self.variables("temperature")

    # ----- temporal window --------------------------------------------
    def forecast_days(self, days: int) -> "WeatherRequest":
        self._forecast_days = days
        return self

    def start_date(self, value: str) -> "WeatherRequest":
        self._start_date = value
        return self

    def end_date(self, value: str) -> "WeatherRequest":
        self._end_date = value
        return self

    def on(self, value: str) -> "WeatherRequest":
        """Request a single ISO date (overrides `forecast_days`)."""
        self._start_date = value
        self._end_date = value
        return self

    def timezone(self, tz: str) -> "WeatherRequest":
        self._timezone = tz
        return self

    # ----- execution ---------------------------------------------------
    def _expanded_hourly_variables(self) -> list[str]:
        if not self._variables:
            raise ValueError("no variables requested; call relative_humidity() or cloud_cover() first")
        if not self._pressure_levels:
            raise ValueError("no pressure levels requested; call pressure_levels() first")
        return [f"{name}_{level}hPa" for name in self._variables for level in self._pressure_levels]

    def _format_coordinates(self) -> tuple[str, str]:
        lat_str = ",".join(f"{lat:.4f}" for lat in self._latitudes)
        lon_str = ",".join(f"{lon:.4f}" for lon in self._longitudes)
        return lat_str, lon_str

    def fetch(self) -> "WeatherResult":
        if not self._latitudes:
            raise ValueError("coordinates() must be called before fetch()")
        # Validate variable/level configuration before handing off to a backend.
        self._expanded_hourly_variables()

        point_payloads = self._backend.fetch(self)
        return WeatherResult(point_payloads, self)

    @staticmethod
    def _as_float_list(value) -> list[float]:
        if isinstance(value, (int, float)):
            return [float(value)]
        return [float(item) for item in value]


class WeatherResult:
    """Parsed multi-point, multi-level result from a `WeatherRequest`."""

    def __init__(self, point_payloads: list[dict], request: WeatherRequest):
        self.source = request._source
        self.latitudes = list(request._latitudes)
        self.longitudes = list(request._longitudes)
        self.pressure_levels = list(request._pressure_levels)
        self.variables = list(request._variables)
        self.timezone = request._timezone
        self.times: list[str] = list(point_payloads[0]["hourly"]["time"])
        self._time_objects = [datetime.fromisoformat(value) for value in self.times]
        self._arrays: dict[str, np.ndarray] = {}
        self._parse(point_payloads)

    def _parse(self, point_payloads: list[dict]) -> None:
        time_count = len(self.times)
        for variable in self.variables:
            level_arrays: list[np.ndarray] = []
            for level in self.pressure_levels:
                key = f"{variable}_{level}hPa"
                point_series: list[list[float]] = []
                for point in point_payloads:
                    raw_series = point.get("hourly", {}).get(key)
                    if raw_series is None:
                        series = [np.nan] * time_count
                    else:
                        series = [float(value) if value is not None else np.nan for value in raw_series]
                    point_series.append(series)
                # Shape: (n_points, n_times)
                level_arrays.append(np.asarray(point_series, dtype=float))
            # Shape: (n_levels, n_points, n_times)
            self._arrays[variable] = np.stack(level_arrays)

    def values(self, variable: str) -> np.ndarray:
        """Return a (n_levels, n_points, n_times) array for one base variable."""
        if variable not in self._arrays:
            available = ", ".join(self._arrays) or "none"
            raise KeyError(f"unknown variable '{variable}' (available: {available})")
        return self._arrays[variable]

    def level(self, variable: str, pressure: int) -> np.ndarray:
        """Return a (n_points, n_times) array for one variable at one pressure."""
        try:
            level_index = self.pressure_levels.index(int(pressure))
        except ValueError as exc:
            raise KeyError(f"pressure level {pressure} hPa was not requested") from exc
        return self.values(variable)[level_index]

    def nearest_index(self, when: datetime) -> int:
        """Return the time index closest to `when` (naive local wall time)."""
        if when.tzinfo is not None:
            when = when.replace(tzinfo=None)
        return min(range(len(self._time_objects)), key=lambda i: abs(self._time_objects[i] - when))
