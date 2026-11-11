"""GFS 0.25° pressure-level profiles straight from NOAA's public AWS bucket (noaa-gfs-bdp-pds, no account needed).

Compared with the Open-Meteo `gfs_seamless` feed this is the plain global GFS run and carries the cloud liquid
(`CLMR`) and cloud ice (`ICMR`) mixing ratios, so cloud phase comes from the model instead of a temperature guess.

Only the needed GRIB2 messages are downloaded (HTTP Range requests guided by the `.idx` file, about 3 MB per
pressure level instead of a 540 MB file) and cached under `gfs_aws/`.  `GfsAwsBackend` plugs into
`weather_data.WeatherRequest` like the Open-Meteo backend does.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import requests

import grib2
from aerosol import _StatusLine, _rate, _size
from weather_data import WeatherBackend

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gfs_aws")
CYCLE_HOURS = 6
MAX_RUNS_BACK = 4        # how many cycles to step back when the newest run lacks the wanted forecast hour
MAX_FORECAST_HOUR = 384
WORKERS = 6

# request variable -> (GRIB short name, conversion to the unit the Open-Meteo backend uses)
VARIABLES = {
    "temperature": ("TMP", lambda k: k - 273.15),            # K -> °C
    "relative_humidity": ("RH", lambda v: v),                # %
    "cloud_cover": ("TCDC", lambda v: v),                    # %
    "geopotential_height": ("HGT", lambda v: v),             # gpm ~ m
    "cloud_liquid_water": ("CLMR", lambda v: v),             # kg/kg, grid-box mean cloud liquid
    "cloud_ice_water": ("ICMR", lambda v: v),                # kg/kg, grid-box mean cloud ice
}


def _unit(n):
    return "MB" if n >= 1e6 else "KB"


def _object_key(run: datetime, hour: int) -> str:
    return f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{hour:03d}"


def _forecast_hour(valid: datetime, run: datetime) -> int:
    hours = round((valid - run).total_seconds() / 3600)
    return 3 * round(hours / 3) if hours > 120 else hours   # beyond f120 the bucket only has 3-hourly files


_https_broken = False   # set once HTTPS to the bucket has failed: later requests go straight to plain HTTP


def _get(url, headers=None, attempts=3, **kwargs):
    """GET with retries.  If HTTPS to the bucket fails (connection / TLS errors), the same request is repeated over
    plain HTTP (the data is public and not secret), and the rest of the run stays on HTTP."""
    global _https_broken
    if _https_broken and url.startswith("https://"):
        url = "http://" + url[len("https://"):]
    try:
        return _get_with_retries(url, headers, attempts, **kwargs)
    except requests.RequestException as exc:
        if not url.startswith("https://"):
            raise
        print(f"HTTPS download from AWS failed ({type(exc).__name__}); trying plain HTTP ...")
        response = _get_with_retries("http://" + url[len("https://"):], headers, attempts, **kwargs)
        _https_broken = True
        return response


def _get_with_retries(url, headers=None, attempts=3, **kwargs):
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=(10, 60), **kwargs)
            if response.status_code in (200, 206, 404):
                return response
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
        time.sleep(2 * (attempt + 1))
    response.raise_for_status()
    return response


def find_run(valid: datetime, now: datetime | None = None):
    """Newest run (<= `valid`, already published) that has the forecast hour closest to `valid`.

    Returns (run, forecast_hour, parsed .idx).  Steps back one cycle at a time while the file is missing."""
    now = now or datetime.now(timezone.utc)
    latest = min(valid, now).astimezone(timezone.utc)
    run = latest.replace(hour=latest.hour - latest.hour % CYCLE_HOURS, minute=0, second=0, microsecond=0)
    for _ in range(MAX_RUNS_BACK):
        hour = _forecast_hour(valid, run)
        if 0 <= hour <= MAX_FORECAST_HOUR:
            response = _get(f"{BUCKET}/{_object_key(run, hour)}.idx")
            if response.status_code == 200:
                rows = [line.split(":") for line in response.text.splitlines() if line]
                return run, hour, rows
        run -= timedelta(hours=CYCLE_HOURS)
    raise RuntimeError(f"no GFS run with a forecast for {valid:%Y-%m-%d %H:%M} UTC in the AWS bucket")


def _ranges(rows, wanted):
    """{(name, level): (start byte, end byte)} for the wanted messages, from the .idx rows."""
    found = {}
    for i, row in enumerate(rows):
        key = (row[3], row[4])
        if key in wanted and i + 1 < len(rows):
            found[key] = (int(row[1]), int(rows[i + 1][1]) - 1)
    return found


class GfsAwsBackend(WeatherBackend):
    """Open-Meteo-shaped payloads built from GFS GRIB2 messages on AWS (bilinear sampling)."""

    def __init__(self):
        self.model_run: datetime | None = None   # init time (UTC) of the run actually used
        self.forecast_hour: int | None = None

    def fetch(self, request) -> list[dict | None]:
        if request._target_time is None:
            raise ValueError("the AWS GFS backend needs the target time: call valid_at() first")
        unknown = [v for v in request._variables if v not in VARIABLES]
        if unknown:
            raise ValueError(f"AWS GFS backend has no variable {', '.join(unknown)}")
        valid = request._target_time.astimezone(timezone.utc)
        run, hour, rows = find_run(valid)
        self.model_run, self.forecast_hour = run, hour
        valid_model = run + timedelta(hours=hour)

        wanted = {(VARIABLES[v][0], f"{level} mb") for v in request._variables for level in request._pressure_levels}
        wanted.add(("HGT", "surface"))     # the model's own terrain height
        ranges = _ranges(rows, wanted)
        missing = wanted - set(ranges)
        if len(missing) == len(wanted):
            raise RuntimeError("the .idx file lists none of the wanted fields")
        fields = self._download(run, hour, ranges)

        lats = np.asarray(request._latitudes, dtype=float)
        lons = np.asarray(request._longitudes, dtype=float)
        local_time = valid_model.astimezone(ZoneInfo(request._timezone)).replace(tzinfo=None)
        hourly = {"time": [local_time.isoformat(timespec="minutes")]}
        for variable in request._variables:
            name, convert = VARIABLES[variable]
            for level in request._pressure_levels:
                field = fields.get((name, f"{level} mb"))
                values = np.full(len(lats), np.nan) if field is None else convert(_sample(field, lats, lons))
                hourly[f"{variable}_{level}hPa"] = [[None if np.isnan(v) else float(v)] for v in values]
        surface = fields.get(("HGT", "surface"))
        elevation = np.full(len(lats), np.nan) if surface is None else _sample(surface, lats, lons)
        payloads = []
        for j in range(len(lats)):
            point_hourly = {"time": hourly["time"]}
            point_hourly.update({key: series[j] for key, series in hourly.items() if key != "time"})
            payloads.append({"hourly": point_hourly,
                             "elevation": None if np.isnan(elevation[j]) else float(elevation[j])})
        return payloads

    def _download(self, run, hour, ranges):
        """Fetch (or load from the cache) every message and decode it; returns {(name, level): Grib2Field}."""
        folder = os.path.join(CACHE_DIR, f"{run:%Y%m%d%H}_f{hour:03d}")
        os.makedirs(folder, exist_ok=True)
        url = f"{BUCKET}/{_object_key(run, hour)}"
        label = f"GFS {run:%Y-%m-%d %Hz} f{hour:03d}"
        status = _StatusLine(period=5.0)
        lock = threading.Lock()
        state = {"done": 0, "bytes": 0}
        started = time.time()

        def load(item):
            key, (start, end) = item
            path = os.path.join(folder, f"{key[0]}_{key[1].replace(' ', '')}.grb2")
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    blob = handle.read()
                fresh = 0
            else:
                response = _get(url, headers={"Range": f"bytes={start}-{end}"})
                if response.status_code != 206:
                    raise RuntimeError(f"AWS answered {response.status_code} for {key[0]} {key[1]}")
                blob = response.content
                with open(path + ".part", "wb") as handle:
                    handle.write(blob)
                os.replace(path + ".part", path)
                fresh = len(blob)
            field = next(grib2.read_messages(blob))
            with lock:
                state["done"] += 1
                state["bytes"] += fresh
                elapsed = max(time.time() - started, 1e-3)
                status.show(f"{label}  {state['done']}/{len(ranges)} fields  {_size(state['bytes'])} {_unit(state['bytes'])}  "
                            f"{_rate(state['bytes'] / elapsed)}")
            return key, field

        with ThreadPoolExecutor(WORKERS) as pool:
            fields = dict(pool.map(load, ranges.items()))
        status.clear()
        elapsed = time.time() - started
        print(f"{label}: {len(ranges)} fields from AWS, {_size(state['bytes'])} {_unit(state['bytes'])} downloaded in {elapsed:.0f} s")
        return fields


def _sample(field: grib2.Grib2Field, lats, lons) -> np.ndarray:
    """Bilinear interpolation of a regular lat/lon grid at the given points (longitude wraps around)."""
    n_lat, n_lon = field.values.shape
    row = np.clip((lats - field.lat0) / field.dlat, 0, n_lat - 1)
    col = ((lons - field.lon0) / field.dlon) % n_lon
    r0 = np.minimum(np.floor(row).astype(int), n_lat - 2)
    c0 = np.floor(col).astype(int) % n_lon
    c1 = (c0 + 1) % n_lon
    fr, fc = row - r0, col - np.floor(col)
    v = field.values
    return ((1 - fr) * ((1 - fc) * v[r0, c0] + fc * v[r0, c1])
            + fr * ((1 - fc) * v[r0 + 1, c0] + fc * v[r0 + 1, c1]))
