"""FY-4B AGRI L2 access.

Reads the Cloud Top Height (CTH) product and samples it at arbitrary
latitude/longitude points.  The L2 data is stored on the FY-4B fixed-grid
projection (``x``/``y`` pixel indices), so this module also implements the
geostationary projection used to map pixels to latitude/longitude.

The cross-section x-axis (distance from the observer) is equivalent to a series
of lat/lon points; ``sample_at`` queries the CTH field at those points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

# FY-4B nominal geostationary projection parameters (4 km full disk, 2748x2748).
_COFF = 1373.5
_LOFF = 1373.5
_CFAC = 10233137.0
_LFAC = 10233137.0
_SUB_LON = 105.0          # FY-4B sub-satellite longitude (FY-4A is 104.7)
_EA = 6378.137            # Earth semi-major axis (km)
_EB = 6356.7523           # Earth semi-minor axis (km)
_H = 42164.0              # distance from Earth centre to satellite (km)

_SCALE = (2 ** 16) * math.pi / 180.0
_EA_OVER_EB_SQ = (_EA / _EB) ** 2

_CTH_VALID_RANGE = (1.0, 20000.0)  # metres

# COT special codes from the CPD product Description attribute.
_COT_SPECIAL = (-999.0, 65531.0, 65532.0, 65535.0)  # fill, clear, nighttime, space

_OBS_RADIUS_KM = 6371.0  # mean Earth radius used for the local transect geometry


@dataclass
class FY4BCTH:
    """Cloud Top Height product sampled onto the fixed grid."""

    latitude: np.ndarray    # (nx, ny) degrees_north
    longitude: np.ndarray   # (nx, ny) degrees_east
    cth_m: np.ndarray       # (nx, ny) metres, invalid pixels are NaN
    dqf: np.ndarray         # (nx, ny) data quality flags


@dataclass
class FY4BCOT:
    """Cloud Optical Thickness product sampled onto the fixed grid."""

    latitude: np.ndarray    # (nx, ny) degrees_north
    longitude: np.ndarray   # (nx, ny) degrees_east
    cot: np.ndarray         # (nx, ny) dimensionless, invalid pixels are NaN
    dqf: np.ndarray         # (nx, ny) data quality flags


def _scan_angle(count: np.ndarray, offset: float, factor: float) -> np.ndarray:
    """Convert pixel index to a scan angle in radians."""
    return (count - offset) / factor * _SCALE


def _grid_geolocation(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(latitude, longitude)`` grids for the FY-4B fixed grid."""
    cols = np.arange(nx, dtype=float)
    lines = np.arange(ny, dtype=float)

    x = _scan_angle(cols, _COFF, _CFAC)[:, None]   # (nx, 1) E-W scan angle
    y = _scan_angle(lines, _LOFF, _LFAC)[None, :]  # (1, ny) N-S scan angle

    cosx = np.cos(x)
    sinx = np.sin(x)
    cosy = np.cos(y)
    siny = np.sin(y)

    cc = cosx * cosy
    cs = sinx * cosy
    cosy2 = cosy ** 2
    siny2 = siny ** 2

    disc = (_H * cc) ** 2 - (cosy2 + _EA_OVER_EB_SQ * siny2) * (_H ** 2 - _EA ** 2)
    sd = np.sqrt(np.maximum(disc, 0.0))
    on_disk = disc >= 0.0

    denom = cosy2 + _EA_OVER_EB_SQ * siny2
    sn = (_H * cc - sd) / denom

    s1 = _H - sn * cc
    s2 = sn * cs
    s3 = -sn * siny
    sxy = np.sqrt(s1 ** 2 + s2 ** 2)

    lon = np.degrees(np.arctan2(s2, s1)) + _SUB_LON
    lat = np.degrees(np.arctan(_EA_OVER_EB_SQ * (s3 / sxy)))

    # Normalise longitude to [-180, 180] and mask space pixels.
    lon = ((lon + 180.0) % 360.0) - 180.0
    lat = np.where(on_disk, lat, np.nan)
    lon = np.where(on_disk, lon, np.nan)

    return lat, lon


def read_cth(path: str) -> FY4BCTH:
    """Read the CTH (and DQF) fields and compute the lat/lon grids."""
    ds = netCDF4.Dataset(path)
    try:
        ds.set_auto_maskandscale(False)
        var = ds.variables["CTH"]
        raw = np.asarray(var[:], dtype=float)
        dqf = np.asarray(ds.variables["DQF"][:], dtype=float)

        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        cth = raw * scale + offset

        lo, hi = _CTH_VALID_RANGE
        cth = np.where((cth >= lo) & (cth <= hi), cth, np.nan)

        nx, ny = cth.shape
        lat, lon = _grid_geolocation(nx, ny)
        return FY4BCTH(latitude=lat, longitude=lon, cth_m=cth, dqf=dqf)
    finally:
        ds.close()


def read_cot(path: str) -> FY4BCOT:
    """Read the COT (and DQF) fields and compute the lat/lon grids."""
    ds = netCDF4.Dataset(path)
    try:
        ds.set_auto_maskandscale(False)
        var = ds.variables["COT"]
        raw = np.asarray(var[:], dtype=float)
        dqf = np.asarray(ds.variables["DQF"][:], dtype=float)

        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        cot = raw * scale + offset
        cot = np.where(np.isin(cot, _COT_SPECIAL), np.nan, cot)

        nx, ny = cot.shape
        lat, lon = _grid_geolocation(nx, ny)
        return FY4BCOT(latitude=lat, longitude=lon, cot=cot, dqf=dqf)
    finally:
        ds.close()


def sample_at(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    lats,
    lons,
) -> np.ndarray:
    """Nearest-neighbour sample of a curvilinear grid at (lat, lon) points."""
    flat_lat = np.asarray(latitude, dtype=float).ravel()
    flat_lon = np.asarray(longitude, dtype=float).ravel()
    flat_vals = np.asarray(values, dtype=float).ravel()

    out = np.empty(len(lats), dtype=float)
    for idx, (la, lo) in enumerate(zip(lats, lons)):
        lo = ((lo + 180.0) % 360.0) - 180.0
        d2 = (flat_lat - la) ** 2 + (flat_lon - lo) ** 2
        out[idx] = flat_vals[int(np.nanargmin(d2))]
    return out


def cloud_top_km(path: str, lats, lons) -> np.ndarray:
    """Convenience: read CTH and return sampled cloud-top height in km."""
    data = read_cth(path)
    return sample_at(data.latitude, data.longitude, data.cth_m, lats, lons) / 1000.0


def _nearest_pixel_indices(latitude, longitude, lats, lons):
    """Return ``(x_indices, y_indices)`` nearest to each (lat, lon) point."""
    flat_lat = np.asarray(latitude, dtype=float).ravel()
    flat_lon = np.asarray(longitude, dtype=float).ravel()

    xs = np.empty(len(lats), dtype=int)
    ys = np.empty(len(lats), dtype=int)
    for idx, (la, lo) in enumerate(zip(lats, lons)):
        lo = ((lo + 180.0) % 360.0) - 180.0
        d2 = (flat_lat - la) ** 2 + (flat_lon - lo) ** 2
        i = int(np.nanargmin(d2))
        xs[idx], ys[idx] = np.unravel_index(i, np.asarray(latitude).shape)
    return xs, ys


def _transect_offsets(latitude, longitude, obs_lat, obs_lon, azimuth_deg):
    """Along/cross-track distance (km) of every grid pixel from the observer.

    ``along`` is the distance in the viewing direction (great-circle bearing
    `azimuth_deg`, degrees clockwise from north); ``cross`` is the signed
    perpendicular offset.
    """
    lat = np.radians(np.asarray(latitude, dtype=float))
    lon = np.radians(np.asarray(longitude, dtype=float))
    lat0 = math.radians(obs_lat)
    lon0 = math.radians(obs_lon)

    dlat = lat - lat0
    dlon = lon - lon0

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2.0) ** 2
    dist = 2.0 * _OBS_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    y = np.sin(dlon) * np.cos(lat)
    x = np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon)
    rel = np.arctan2(y, x) - math.radians(azimuth_deg)

    return dist * np.cos(rel), dist * np.sin(rel)


def transect_profile(
    latitude,
    longitude,
    values,
    obs_lat,
    obs_lon,
    azimuth_deg,
    x_min,
    x_max,
    half_width_km,
):
    """Sample a curvilinear grid along a transect at its native resolution.

    Keeps every pixel whose cross-track offset is within `half_width_km` and
    whose along-track distance lies in ``[x_min, x_max]``, sorts by distance and
    drops non-finite values.  Returns ``(distance_km, value)`` arrays.
    """
    along, cross = _transect_offsets(latitude, longitude, obs_lat, obs_lon, azimuth_deg)
    flat_along = along.ravel()
    flat_vals = np.asarray(values, dtype=float).ravel()

    keep = (
        (np.abs(cross.ravel()) <= half_width_km)
        & (flat_along >= x_min)
        & (flat_along <= x_max)
    )
    distance = flat_along[keep]
    value = flat_vals[keep]

    order = np.argsort(distance)
    distance, value = distance[order], value[order]
    finite = np.isfinite(value)
    return distance[finite], value[finite]


def _plot_full_disk(
    latitude: np.ndarray,
    longitude: np.ndarray,
    values: np.ndarray,
    lats,
    lons,
    title: str,
    cbar_label: str,
    save_path: str | None,
    stride: int,
    show: bool,
    latlon: bool,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Shared full-disk plotting routine for X/Y and lat/lon modes."""
    nx, ny = values.shape
    fig, ax = plt.subplots(figsize=(9, 9), dpi=100)

    if latlon:
        lon_sub = longitude[::stride, ::stride]
        lat_sub = latitude[::stride, ::stride]
        val_sub = values[::stride, ::stride]
        valid = np.isfinite(lon_sub) & np.isfinite(lat_sub) & np.isfinite(val_sub)
        sc = ax.scatter(lon_sub[valid], lat_sub[valid], c=val_sub[valid],
                        s=1, cmap="turbo", vmin=vmin, vmax=vmax, linewidths=0)
        fig.colorbar(sc, ax=ax, label=cbar_label)

        if lats is not None and lons is not None:
            ax.plot(lons, lats, color="red", marker="o", markersize=3, linewidth=1,
                    label="Cross-section transect")
            ax.legend()

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(f"{title} (lat/lon)")
    else:
        x = np.arange(0, nx, stride)
        y = np.arange(0, ny, stride)
        C = values[::stride, ::stride].T
        X, Y = np.meshgrid(x, y)

        pc = ax.pcolormesh(X, Y, C, shading="nearest", cmap="turbo", vmin=vmin, vmax=vmax)
        fig.colorbar(pc, ax=ax, label=cbar_label)

        if lats is not None and lons is not None:
            xs, ys = _nearest_pixel_indices(latitude, longitude, lats, lons)
            ax.plot(xs, ys, color="red", marker="o", markersize=3, linewidth=1,
                    label="Cross-section transect")
            ax.legend()

        ax.set_xlabel("X pixel (E-W)")
        ax.set_ylabel("Y pixel (N-S)")
        ax.set_title(f"{title} (full disk)")
        ax.set_aspect("equal")
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.invert_yaxis()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_cth(
    path: str,
    lats=None,
    lons=None,
    save_path: str | None = None,
    stride: int = 4,
    show: bool = True,
    latlon: bool = False,
):
    """Plot the full-disk FY-4B cloud-top height field (X/Y or lat/lon)."""
    data = read_cth(path)
    return _plot_full_disk(
        data.latitude, data.longitude, data.cth_m / 1000.0,
        lats, lons,
        title="FY-4B AGRI L2 Cloud Top Height",
        cbar_label="Cloud top height (km)",
        save_path=save_path, stride=stride, show=show, latlon=latlon,
        vmin=0.0, vmax=20.0,
    )


def plot_cot(
    path: str,
    lats=None,
    lons=None,
    save_path: str | None = None,
    stride: int = 4,
    show: bool = True,
    latlon: bool = False,
):
    """Plot the full-disk FY-4B cloud optical thickness field (X/Y or lat/lon)."""
    data = read_cot(path)
    return _plot_full_disk(
        data.latitude, data.longitude, data.cot,
        lats, lons,
        title="FY-4B AGRI L2 Cloud Optical Thickness",
        cbar_label="Cloud optical thickness",
        save_path=save_path, stride=stride, show=show, latlon=latlon,
        vmin=None, vmax=None,
    )


def plot_transect(
    cth_path: str,
    cot_path: str,
    obs_lat: float,
    obs_lon: float,
    azimuth_deg: float,
    x_min: float = -100.0,
    x_max: float = 900.0,
    half_width_km: float = 2.0,
    save_path: str | None = None,
    show: bool = True,
):
    """Stacked CTH / COT transect at the satellite's native pixel resolution.

    Both panels share the distance-from-observer x-axis so the sampled pixels
    line up with the atmosphere cross-section drawn by ``main.py``.  CTH (km) is
    on top; COT uses a logarithmic axis because it spans ~0.02-100.
    """
    cth = read_cth(cth_path)
    cot = read_cot(cot_path)

    d_cth, h_cth = transect_profile(
        cth.latitude, cth.longitude, cth.cth_m,
        obs_lat, obs_lon, azimuth_deg, x_min, x_max, half_width_km,
    )
    d_cot, t_cot = transect_profile(
        cot.latitude, cot.longitude, cot.cot,
        obs_lat, obs_lon, azimuth_deg, x_min, x_max, half_width_km,
    )
    positive = t_cot > 0.0
    d_cot, t_cot = d_cot[positive], t_cot[positive]

    fig, (ax_cth, ax_cot) = plt.subplots(2, 1, figsize=(14, 7), dpi=120, sharex=True)

    ax_cth.scatter(d_cth, h_cth / 1000.0, s=8, color="darkgreen")
    ax_cth.set_ylabel("Cloud top height (km)", fontsize=11)
    ax_cth.set_ylim(0, 16)
    ax_cth.set_title("FY-4B AGRI L2 cloud products along the observer transect", fontsize=13)

    ax_cot.scatter(d_cot, t_cot, s=8, color="purple")
    # ax_cot.set_yscale("log")
    ax_cot.set_ylabel("Cloud optical thickness", fontsize=11)
    ax_cot.set_xlabel("Distance from Observer (km) [Towards Sun direction]", fontsize=11)

    for ax in (ax_cth, ax_cot):
        ax.grid(True, linestyle=":", alpha=0.5)
    ax_cth.axvline(0.0, color="red", linewidth=0.8, alpha=0.6, label="Observer")
    ax_cot.axvline(0.0, color="red", linewidth=0.8, alpha=0.6)
    ax_cth.legend(loc="upper right", fontsize=9)
    ax_cot.set_xlim(x_min, x_max)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig, (ax_cth, ax_cot)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cth-file", help="FY-4B CTH (Cloud Top Height) NetCDF file path")
    parser.add_argument("--cpd-file", help="FY-4B CPD (Cloud Parameters, including cloud optical thickness) NetCDF file path")
    parser.add_argument("--latlon", action=argparse.BooleanOptionalAction, default=False,
                        help="draw with lat/lon axis instead of raw X-Y pixels")
    parser.add_argument("--transect", action=argparse.BooleanOptionalAction, default=False,
                        help="draw the stacked CTH/COT transect figure (requires --cth-file and --cpd-file)")
    parser.add_argument("--obs-lat", type=float, help="observer latitude (deg) for the transect")
    parser.add_argument("--obs-lon", type=float, help="observer longitude (deg) for the transect")
    parser.add_argument("--azimuth", type=float, default=90.0,
                        help="viewing azimuth toward the sun (deg clockwise from north)")
    parser.add_argument("--x-min", type=float, default=-100.0, help="transect start distance (km)")
    parser.add_argument("--x-max", type=float, default=900.0, help="transect end distance (km)")
    parser.add_argument("--half-width", type=float, default=2.0,
                        help="cross-track half-width of the sampled corridor (km)")
    parser.add_argument("--save", default="", help="output PNG path for the transect figure")
    args = parser.parse_args()
    if args.transect:
        if not (args.cth_file and args.cpd_file):
            parser.error("--transect requires both --cth-file and --cpd-file")
        if args.obs_lat is None or args.obs_lon is None:
            parser.error("--transect requires --obs-lat and --obs-lon")
        plot_transect(
            args.cth_file, args.cpd_file, args.obs_lat, args.obs_lon, args.azimuth,
            x_min=args.x_min, x_max=args.x_max, half_width_km=args.half_width,
            save_path=args.save or None, show=True,
        )
    else:
        if args.cth_file:
            plot_cth(args.cth_file, latlon=args.latlon, show=True)
        if args.cpd_file:
            plot_cot(args.cpd_file, latlon=args.latlon, show=True)
