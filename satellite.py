"""Satellite data-source abstraction.

`main.py` chooses a satellite provider via `SATELLITE_SOURCE` and calls one
common method, `SatelliteSource.read_cloud_top()`.  Adding a new satellite is
just a matter of implementing `SatelliteSource` and registering it in
`SOURCES`.
"""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import fy4b
import himawari


@dataclass
class SatelliteCloudTop:
    """Cloud-top information sampled along the observer transect.

    Providers fill whichever fields they support:
      * `temperature_c` - cloud-top temperature proxy (°C)
      * `altitude_km`   - direct cloud-top altitude (km), when the product has it
    """

    temperature_c: np.ndarray | None = None
    altitude_km: np.ndarray | None = None


class SatelliteSource(ABC):
    """Common interface implemented by every satellite provider."""

    name: str = ""

    @abstractmethod
    def read_cloud_top(
        self,
        event_utc,
        lats,
        lons,
        now_utc=None,
        local_file: str | None = None,
    ) -> SatelliteCloudTop:
        """Return cloud-top data sampled at each (lat, lon) transect point."""


class HimawariSource(SatelliteSource):
    """Himawari-8/9 full-disk L1 data via JAXA P-Tree.

    Band-14 brightness temperature (`tbb_14`) is used as a cloud-top
    temperature proxy; no direct cloud-top altitude is available from this L1
    product.
    """

    name = "himawari"

    def read_cloud_top(self, event_utc, lats, lons, now_utc=None, local_file=None):
        path = local_file if local_file and os.path.exists(local_file) else None
        if path is None:
            remote_path, _ = himawari.select_file(event_utc, now_utc)
            path = himawari.download(remote_path) if remote_path else None
        if not path:
            return SatelliteCloudTop()

        sat_lat, sat_lon, tbb_k = himawari.read_grid(path, "tbb_14")
        temperature_c = himawari.sample_at(sat_lat, sat_lon, tbb_k, lats, lons) - 273.15
        return SatelliteCloudTop(temperature_c=temperature_c)


class Fy4bSource(SatelliteSource):
    """FY-4B AGRI L2 Cloud Top Height product."""

    name = "fy4b"

    def __init__(self, cth_method: str = "nearest"):
        self.cth_method = cth_method

    def read_cloud_top(self, event_utc, lats, lons, now_utc=None, local_file=None):
        path = local_file if local_file and os.path.exists(local_file) else None
        if path is None:
            candidates = glob.glob("FY4B*CTH*.NC") + glob.glob("FY4B*CTH*.nc")
            path = candidates[0] if candidates else None
        if not path:
            return SatelliteCloudTop()

        altitude_km = fy4b.cloud_top_km(path, lats, lons, self.cth_method)
        return SatelliteCloudTop(altitude_km=altitude_km)


SOURCES = {
    HimawariSource.name: HimawariSource,
    Fy4bSource.name: Fy4bSource,
}


def get_source(name: str, cth_method: str = "nearest") -> SatelliteSource:
    """Return a satellite provider instance by name (``cth_method`` only applies to fy4b)."""
    try:
        source_cls = SOURCES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCES))
        raise KeyError(f"unknown satellite source '{name}' (available: {available})") from exc
    return source_cls(cth_method) if name == Fy4bSource.name else source_cls()
