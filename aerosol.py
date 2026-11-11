"""CAMS global forecast: 3D aerosol extinction coefficient along the transect.

Data come from the Copernicus Atmosphere Data Store (ADS, dataset
cams-global-atmospheric-composition-forecasts, 0.4 deg, 25 pressure levels, 3-hourly lead times).
Credentials are read by cdsapi from ~/.cdsapirc (url https://ads.atmosphere.copernicus.eu/api).
"""
import glob
import os
import sys
import time
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

DATASET = "cams-global-atmospheric-composition-forecasts"
WAVELENGTHS_NM = (355, 532, 1064)
# ADS pressure levels from the surface up to 200 hPa (no 975 hPa in CAMS)
LEVELS_HPA = [1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300, 250, 200]
RUN_HOURS = (0, 12)          # forecast base times offered by the dataset
MAX_LEAD_HOURS = 120
LEAD_STEP_HOURS = 3
RUN_DELAY_HOURS = 8          # approximate time before a run shows up on ADS
SKIP_NEWEST_RUNS = 1         # start one cycle earlier than the estimate: a run that is not published yet is rejected (400)
CACHE_DIR = "cams_ads"


@dataclass
class AerosolProfile:
    levels_hpa: np.ndarray       # (levels,)
    extinction_per_km: np.ndarray  # (levels, points), NaN where missing
    wavelength_nm: int
    run: datetime                # forecast base time (UTC)
    valid: datetime              # valid time (UTC) of the interpolated field


def candidate_runs(now=None, count=4):
    """Newest-first base times that should already be available (the newest estimate is skipped, see SKIP_NEWEST_RUNS)."""
    available = (now or datetime.now(timezone.utc)) - timedelta(hours=RUN_DELAY_HOURS)
    run = available.replace(hour=max(h for h in RUN_HOURS if h <= available.hour),
                            minute=0, second=0, microsecond=0)
    run -= timedelta(hours=24 // len(RUN_HOURS)) * SKIP_NEWEST_RUNS
    runs = [run]
    while len(runs) < count:
        run -= timedelta(hours=24 // len(RUN_HOURS))
        runs.append(run)
    return runs


def _variable(wavelength_nm):
    if wavelength_nm not in WAVELENGTHS_NM:
        raise ValueError(f"wavelength must be one of {WAVELENGTHS_NM} nm")
    return f"aerosol_extinction_coefficient_{wavelength_nm}nm"


def _area(lats, lons, margin=1.0):
    """[north, west, south, east] covering the transect (longitudes in -180..180)."""
    lons = (np.asarray(lons, dtype=float) + 180.0) % 360.0 - 180.0
    return [min(90.0, float(np.max(lats)) + margin), max(-180.0, float(lons.min()) - margin),
            max(-90.0, float(np.min(lats)) - margin), min(180.0, float(lons.max()) + margin)]


def _rate(bytes_per_s):
    return f"{bytes_per_s / 1e6:6.2f} MB/s" if bytes_per_s >= 1e6 else f"{bytes_per_s / 1e3:6.0f} KB/s"


def _size(n, unit=None):
    unit = unit or ("MB" if n >= 1e6 else "KB")
    return f"{n / (1e6 if unit == 'MB' else 1e3):.1f}"


class _StatusLine:
    """One live line (rewritten in place on a terminal; a new line every `period` seconds otherwise)."""

    def __init__(self, period=10.0):
        self.interactive = sys.stdout.isatty()
        self.period = period
        self.last = 0.0

    def show(self, text, force=False):
        now = time.time()
        if self.interactive:
            if now - self.last >= 0.25 or force:
                print("\r" + text.ljust(79), end="", flush=True)
                self.last = now
        elif now - self.last >= self.period or force:
            print(text, flush=True)
            self.last = now

    def clear(self):
        if self.interactive:
            print("\r" + " " * 79 + "\r", end="", flush=True)


def _fetch(cdsapi, label, request, zip_path, attempts=3):
    """Submit the ADS request, wait in the queue showing its state, then download with live progress."""
    import requests
    client = cdsapi.Client(quiet=True, wait_until_complete=False)
    remote = client.retrieve(DATASET, request)
    if not hasattr(remote, "get_results"):  # an old-style cdsapi server: no queue/progress hooks
        remote.download(zip_path)
        return
    started, line, last_status = time.time(), _StatusLine(30.0), None
    sleep = 1.0
    try:
        while not remote.results_ready:  # raises with the server's full message if the job failed
            status = remote.last_status
            waited = int(time.time() - started)
            line.show(f"{label}  {'queued' if status == 'accepted' else status} on ADS, {waited // 60}:{waited % 60:02d} elapsed",
                      force=status != last_status)
            last_status = status
            time.sleep(sleep)
            sleep = min(sleep * 1.5, 5.0)
    finally:
        line.clear()
    results = remote.get_results()
    url, expected = results.location, results.content_length
    verify = results.request_options.get("verify", True)
    part, got = zip_path + ".part", 0
    started, line = time.time(), _StatusLine(10.0)
    try:
        for attempt in range(1, attempts + 1):
            headers = {"Range": f"bytes={got}-"} if got else {}
            try:
                with requests.get(url, stream=True, headers=headers, timeout=(15, 60), verify=verify) as resp:
                    resp.raise_for_status()
                    if got and resp.status_code != 206:  # the server ignored the range: start over
                        got = 0
                    recent = deque([(time.time(), got)])
                    with open(part, "ab" if got else "wb") as f:
                        for chunk in resp.iter_content(1 << 16):
                            f.write(chunk)
                            got += len(chunk)
                            now = time.time()
                            recent.append((now, got))
                            while len(recent) > 2 and now - recent[0][0] > 2:
                                recent.popleft()
                            speed = (got - recent[0][1]) / max(now - recent[0][0], 1e-3)
                            unit = "MB" if expected >= 1e6 else "KB"
                            line.show(f"{label}  {_size(got, unit):>6}/{_size(expected, unit)} {unit} {got / expected:4.0%}  {_rate(speed)}")
                if got >= expected:
                    break
                raise requests.ConnectionError(f"incomplete: {got} of {expected} bytes")
            except requests.RequestException as exc:
                line.clear()
                if attempt == attempts:
                    raise
                print(f"{label}  download interrupted at {_size(got)}/{_size(expected)} ({exc}); resuming ({attempt}/{attempts})")
                time.sleep(5)
        os.replace(part, zip_path)
    finally:
        line.clear()
        if os.path.exists(part):
            os.remove(part)
    elapsed = max(time.time() - started, 1e-3)
    print(f"{label}  downloaded {_size(expected)} {'MB' if expected >= 1e6 else 'KB'}  {_rate(expected / elapsed).strip()}")


def _download(run, leads, wavelength_nm, area, cache_dir):
    """Fetch (or reuse) the netCDF for one run; returns the path of the extracted .nc file."""
    import cdsapi  # imported lazily: only needed when the aerosol layer is requested
    os.makedirs(cache_dir, exist_ok=True)
    tag = (f"{run:%Y%m%d%H}_{wavelength_nm}nm_L{'-'.join(map(str, leads))}_"
           + "_".join(f"{v:.0f}" for v in area))
    nc_path = os.path.join(cache_dir, tag + ".nc")
    if os.path.exists(nc_path):
        return nc_path
    request = {
        "variable": [_variable(wavelength_nm)],
        "pressure_level": [str(p) for p in LEVELS_HPA],
        "date": [f"{run:%Y-%m-%d}"],
        "time": [f"{run:%H}:00"],
        "leadtime_hour": [str(h) for h in leads],
        "type": ["forecast"],
        "data_format": "netcdf_zip",
        "area": area,
    }
    zip_path = os.path.join(cache_dir, tag + ".zip")
    _fetch(cdsapi, f"CAMS {run:%Y-%m-%d %Hz}", request, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.namelist() if m.endswith(".nc")]
        if not members:
            raise RuntimeError("ADS returned no netCDF file")
        with archive.open(members[0]) as src, open(nc_path, "wb") as dst:
            dst.write(src.read())
    os.remove(zip_path)
    return nc_path


def _bilinear(field, grid_lat, grid_lon, lats, lons):
    """field (..., lat, lon) sampled at points; grid_lat may be descending."""
    if grid_lat[0] > grid_lat[-1]:
        grid_lat, field = grid_lat[::-1], field[..., ::-1, :]
    glon = np.mod(grid_lon, 360.0)
    order = np.argsort(glon)
    glon, field = glon[order], field[..., order]
    lons = np.mod(np.asarray(lons, dtype=float), 360.0)
    lats = np.asarray(lats, dtype=float)
    iy = np.clip(np.searchsorted(grid_lat, lats) - 1, 0, len(grid_lat) - 2)
    ix = np.clip(np.searchsorted(glon, lons) - 1, 0, len(glon) - 2)
    ty = np.clip((lats - grid_lat[iy]) / (grid_lat[iy + 1] - grid_lat[iy]), 0, 1)
    tx = np.clip((lons - glon[ix]) / (glon[ix + 1] - glon[ix]), 0, 1)
    return ((1 - ty) * (1 - tx) * field[..., iy, ix] + (1 - ty) * tx * field[..., iy, ix + 1]
            + ty * (1 - tx) * field[..., iy + 1, ix] + ty * tx * field[..., iy + 1, ix + 1])


def _read(nc_path):
    """Field as (time, level, lat, lon) plus its axes; every other dimension is folded into time
    (ADS files hold one forecast run, so that axis is just the requested lead times in order)."""
    import netCDF4
    with netCDF4.Dataset(nc_path) as ds:
        names = ds.variables
        # the data variable is the one defined on latitude, longitude and a pressure-level axis
        var = next(v for v in names if names[v].ndim >= 3
                   and any(d.startswith("lat") for d in names[v].dimensions)
                   and any("level" in d or d.startswith("plev") for d in names[v].dimensions))
        dims = list(names[var].dimensions)
        lat_name = next(d for d in dims if d.startswith("lat"))
        lon_name = next(d for d in dims if d.startswith("lon"))
        level_name = next(d for d in dims if "level" in d or d.startswith("plev"))
        data = np.ma.filled(names[var][:].astype(float), np.nan)
        glat = np.asarray(names[lat_name][:], dtype=float)
        glon = np.asarray(names[lon_name][:], dtype=float)
        glev = np.asarray(names[level_name][:], dtype=float)
    spatial = [dims.index(level_name), dims.index(lat_name), dims.index(lon_name)]
    others = [i for i in range(len(dims)) if i not in spatial]
    data = np.transpose(data, others + spatial)
    data = data.reshape((-1,) + data.shape[len(others):])
    return data, glat, glon, glev


def read_extinction(valid, lats, lons, wavelength_nm=532, cache_dir=CACHE_DIR, now=None):
    """Aerosol extinction (km^-1) on LEVELS_HPA along the transect at UTC time `valid`.

    Tries the newest forecast runs first; the valid time is interpolated linearly between the two
    3-hourly lead times around it. Returns None when no run could be fetched.
    """
    valid = valid.astimezone(timezone.utc)
    area = _area(lats, lons)
    for run in candidate_runs(now):
        hours = (valid - run).total_seconds() / 3600.0
        if hours < 0 or hours > MAX_LEAD_HOURS:
            continue
        lead0 = int(hours // LEAD_STEP_HOURS) * LEAD_STEP_HOURS
        leads = [lead0] if hours == lead0 else [lead0, lead0 + LEAD_STEP_HOURS]
        if leads[-1] > MAX_LEAD_HOURS:
            continue
        try:
            path = _download(run, leads, wavelength_nm, area, cache_dir)
        except Exception as exc:  # run not published yet / network / credentials
            detail = str(exc).strip().replace("\n", "\n    ")  # the whole message: ADS puts the reason after the URL
            print(f"CAMS run {run:%Y-%m-%d %Hz}: {detail}")
            continue
        data, glat, glon, glev = _read(path)
        if data.shape[0] != len(leads):
            print(f"CAMS run {run:%Y-%m-%d %Hz}: unexpected time axis ({data.shape[0]} steps for leads {leads})")
            continue
        order = np.argsort(glev)[::-1]  # surface first
        sampled = _bilinear(data[:, order], glat, glon, lats, lons)  # (time, level, point)
        if len(leads) == 1:
            profile = sampled[0]
        else:
            profile = (1 - (hours - leads[0]) / LEAD_STEP_HOURS) * sampled[0] + (hours - leads[0]) / LEAD_STEP_HOURS * sampled[1]
        levels = glev[order]
        return AerosolProfile(levels, profile * 1000.0, wavelength_nm, run, valid)  # m^-1 -> km^-1
    return None


def level_heights_km(levels_hpa, model_levels_hpa, height_km):
    """Heights (km) of `levels_hpa` at every point, interpolating the model's height in ln(p).

    height_km: (model levels, points) geopotential height of `model_levels_hpa`.
    """
    model_levels_hpa = np.asarray(model_levels_hpa, dtype=float)
    order = np.argsort(np.log(model_levels_hpa))
    log_model = np.log(model_levels_hpa)[order]
    out = np.empty((len(levels_hpa), height_km.shape[1]))
    for j in range(height_km.shape[1]):
        out[:, j] = np.interp(np.log(levels_hpa), log_model, height_km[order, j])
    return out
