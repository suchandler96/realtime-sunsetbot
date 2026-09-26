"""Access JAXA P-Tree Himawari L1 gridded NetCDF files over FTP.

The P-Tree server exposes Himawari-8/9 full-disk and Japan-area *L1* radiance
data (not L2 cloud products).  File layout (from README_HimawariNetCDF_en.txt):

    /jma/netcdf/<YYYYMM>/<DD>/<file>.nc

File naming for full-disk data:

    NC_Hnn_YYYYMMDD_hhmm_R21_FLDK.xxxxx_yyyyy.nc

where `hhmm` is the UTC observation start time.

The selection rule implemented here:
  * future event (sunset/sunrise still to happen)  -> latest available file
  * past event   (sunset/sunrise already happened) -> file closest to the event
"""

from __future__ import annotations

import ftplib
import os
import re
from datetime import datetime, timedelta, timezone

import netCDF4
import numpy as np

# ---- FTP credentials / server -----------------------------------------
FTP_HOST = "ftp.ptree.jaxa.jp"
FTP_USER = "zj20140160_163.com"
FTP_PASSWORD = "SP+wari8"
FTP_BASE = "jma/netcdf"

# Which product to use.  Hong Kong (22N, 114E) is outside the Japan-area box,
# so the full-disk product is required.
#   "R21_FLDK.02801_02401" : full disk, 5 km
#   "R21_FLDK.07001_06001" : full disk, 2 km
#   "r14_FLDK.02701_02601" : Japan area, 1 km
AREA = "R21_FLDK.02801_02401"

# NC_H09_20260912_1100_R21_FLDK.02801_02401.nc
_TIMESTAMP_RE = re.compile(r"^NC_H\d{2}_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})_")


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def connect() -> ftplib.FTP:
    """Open and authenticate an FTP connection."""
    ftp = ftplib.FTP(FTP_HOST, timeout=60)
    ftp.login(FTP_USER, FTP_PASSWORD)
    return ftp


def parse_file_time(filename: str) -> datetime | None:
    """Return the UTC observation start time encoded in a filename, or None."""
    m = _TIMESTAMP_RE.match(filename)
    if not m:
        return None
    year, month, day, hour, minute = (int(g) for g in m.groups())
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _day_dir(dt: datetime) -> str:
    return f"{dt:%Y%m}/{dt:%d}"


def _remote_path(dt: datetime, filename: str) -> str:
    return f"{FTP_BASE}/{_day_dir(dt)}/{filename}"


def _list_day(ftp: ftplib.FTP, dt: datetime) -> list[str]:
    """List AREA-matching filenames for one UTC day.

    Returns an empty list if the day directory does not exist (yet) on the
    server, so a search over adjacent days never crashes.
    """
    path = f"{FTP_BASE}/{_day_dir(dt)}"
    try:
        ftp.cwd("/")
        ftp.cwd(path)
        return [name for name in ftp.nlst() if AREA in name]
    except ftplib.error_perm:
        return []


def _files_with_times(ftp: ftplib.FTP, dt: datetime) -> list[tuple[datetime, str]]:
    entries: list[tuple[datetime, str]] = []
    for name in _list_day(ftp, dt):
        t = parse_file_time(name)
        if t is not None:
            entries.append((t, name))
    return entries


def select_file(
    event_utc: datetime,
    now_utc: datetime | None = None,
    ftp: ftplib.FTP | None = None,
) -> tuple[str | None, datetime | None]:
    """Pick the NetCDF file for a sunset/sunrise event.

    Returns ``(remote_path, observation_time_utc)``, or ``(None, None)`` if no
    matching file is found.  `event_utc` may be aware or naive (naive = UTC).
    """
    event_utc = _to_utc(event_utc)
    now_utc = _to_utc(now_utc or datetime.now(timezone.utc))

    owns_ftp = ftp is None
    if owns_ftp:
        ftp = connect()

    try:
        if event_utc > now_utc:
            # Future event: use the latest available observation.
            for back in range(0, 8):
                day = now_utc - timedelta(days=back)
                entries = [
                    (t, name)
                    for t, name in _files_with_times(ftp, day)
                    if t <= now_utc
                ]
                if entries:
                    latest = max(entries, key=lambda e: e[0])
                    t, name = latest
                    return _remote_path(day, name), t
            return None, None

        # Past event: use the observation closest to the event time.
        best: tuple[float, datetime, str, datetime] | None = None  # (dt, t, name, day)
        for offset in (-1, 0, 1):
            day = event_utc + timedelta(days=offset)
            for t, name in _files_with_times(ftp, day):
                dt = abs((t - event_utc).total_seconds())
                if best is None or dt < best[0]:
                    best = (dt, t, name, day)
        if best is None:
            return None, None
        _, t, name, day = best
        return _remote_path(day, name), t
    finally:
        if owns_ftp:
            ftp.quit()


def download(remote_path: str, local_dir: str = ".", ftp: ftplib.FTP | None = None) -> str:
    """Download a remote file and return the local path.

    If the file already exists in `local_dir`, it is reused without connecting
    to the server or re-downloading.
    """
    os.makedirs(local_dir, exist_ok=True)
    filename = remote_path.rsplit("/", 1)[-1]
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        return local_path

    owns_ftp = ftp is None
    if owns_ftp:
        ftp = connect()
    try:
        with open(local_path, "wb") as fh:
            ftp.retrbinary(f"RETR {remote_path}", fh.write)
    finally:
        if owns_ftp:
            ftp.quit()
    return local_path


def read_grid(local_path: str, variable: str = "tbb_14") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one 2D variable from a downloaded full-disk NetCDF file.

    Returns ``(latitudes, longitudes, values)`` in physical units, with fill
    values replaced by NaN.  `latitudes` is (nlat,), `longitudes` is (nlon,),
    and `values` is (nlat, nlon).
    """
    ds = netCDF4.Dataset(local_path)
    try:
        ds.set_auto_maskandscale(False)  # read raw storage values, scale manually
        latitudes = np.asarray(ds.variables["latitude"][:], dtype=float)
        longitudes = np.asarray(ds.variables["longitude"][:], dtype=float)
        var = ds.variables[variable]
        raw = np.asarray(var[:], dtype=float)
        scale = float(getattr(var, "scale_factor", 1.0))
        offset = float(getattr(var, "add_offset", 0.0))
        fill = getattr(var, "missing_value", None)
        if fill is None:
            fill = getattr(var, "_FillValue", None)

        values = raw * scale + offset
        if fill is not None:
            values[raw == fill] = np.nan
        return latitudes, longitudes, values
    finally:
        ds.close()


def sample_at(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    values: np.ndarray,
    lats,
    lons,
) -> np.ndarray:
    """Nearest-neighbour sample of a 2D grid at arbitrary (lat, lon) points."""
    out = np.empty(len(lats), dtype=float)
    for idx, (la, lo) in enumerate(zip(lats, lons)):
        lo = lo % 360  # file longitudes are 0..360
        j = int(np.argmin(np.abs(latitudes - la)))
        i = int(np.argmin(np.abs(longitudes - lo)))
        out[idx] = values[j, i]
    return out
