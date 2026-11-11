"""Open-Meteo backend that spreads a big multi-point download over many minutes and keeps what it got.

A vividness map needs thousands of profiles, but the free Open-Meteo API limits the *weighted* number of calls
(600 / minute, 5 000 / hour, 10 000 / day, see MINUTE_LIMIT below; a location costs max(1, variables / 10) calls,
so a profile of 4 variables x 16 pressure levels is 6.4).  `PacedOpenMeteoBackend` is a drop-in `OpenMeteoBackend` that

  * snaps the points to the model grid (GFS / ECMWF / CMA: 0.25 / 0.25 / 0.125 deg) and drops duplicates, since
    neighbouring transects share model columns - the same grid cell is never paid for twice;
  * answers from an on-disk cache first (one gzip file per finished request in `om_cache/`), so an interrupted or
    quota-limited run is resumed by running the same command again, and a re-run costs nothing;
  * downloads the rest in requests sized to the per-minute budget, one request whenever the sliding minute / hour /
    day windows (persisted in `usage.json`, so they hold across runs) have room;
  * if a window would make it wait longer than `max_wait_s` (typically the daily quota), stops with
    `DownloadIncomplete`; everything fetched so far is kept and the next run continues.

Cached data older than `max_age_h` is ignored so that a map is not built from very different model runs.

    backend = PacedOpenMeteoBackend()                  # then hand it to WeatherData.source(..., backend=backend)
    backend = PacedOpenMeteoBackend(plan_only=True)    # print the estimate (points, calls, minutes), download nothing
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from datetime import date

import requests

from weather_data import DataSource, OpenMeteoBackend

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "om_cache")

# Native grid spacing (deg) of models whose grid is aligned to multiples of it; query points are snapped to cell
# centres.  Other models (JMA MSM, HRRR: finer, offset grids) are only rounded to 4 decimals.
GRID_DEG: dict[DataSource, float] = {DataSource.GFS: 0.25, DataSource.ECMWF: 0.25, DataSource.CMA_GFS: 0.125}

# Free (non-commercial) tier limits, in weighted API calls; applied per IP address.
# Source: https://open-meteo.com/en/pricing (comparison table) and https://open-meteo.com/en/terms ("Less than
# 10'000 API calls per day, 5'000 per hour and 600 per minute").  The 300 000 / month limit is not enforced.
MINUTE_LIMIT = 600.0
HOUR_LIMIT = 5000.0
DAY_LIMIT = 10000.0
# Fraction of each limit this module uses, as a margin for other uses of the same IP and because the terms say "less
# than" for the hourly / daily limits.
MINUTE_SAFETY, HOUR_SAFETY, DAY_SAFETY = 0.8, 0.95, 0.95
DEFAULT_LIMITS = (MINUTE_LIMIT * MINUTE_SAFETY, HOUR_LIMIT * HOUR_SAFETY, DAY_LIMIT * DAY_SAFETY)

# Weight of one request, from the FAQ "How is one API call defined?" at https://open-meteo.com/en/pricing: "Requests
# for data covering more than 10 weather variables or extending over a period of more than 2 weeks for a single
# location are considered multiple API calls ... fractional counts are used. For example, a request for 2 weeks of
# data with 15 weather variables will be calculated as 1.5 API calls, while 4 weeks of data equals 3.0".
# Hence weight per location = max(1, variables / 10) * max(1, days / 14).  The page does not say how several
# locations in one request count; we assume they add up (the safe reading).
CALL_VARIABLES_UNIT = 10
CALL_DAYS_UNIT = 14
WINDOWS_S = (60.0, 3600.0, 86400.0)


class DownloadIncomplete(RuntimeError):
    """The quota would make the download wait too long; the data fetched so far is saved, run again later."""


class PlanOnly(SystemExit):
    """Raised after printing the download plan when `plan_only` is set."""


# ==================== quota accounting ====================
class UsageLedger:
    """Weighted calls made in the last 24 h, persisted so the sliding windows also hold across runs."""

    def __init__(self, path: str | None, limits=DEFAULT_LIMITS, clock=time.time):
        self.path, self.limits, self.clock = path, list(limits), clock
        self.entries: list[tuple[float, float]] = []   # (time, weight)
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self.entries = [(float(t), float(w)) for t, w in json.load(f)]
            except (ValueError, OSError):
                self.entries = []
        self._prune(self.clock())

    def _prune(self, now: float) -> None:
        self.entries = [(t, w) for t, w in self.entries if now - t < WINDOWS_S[-1]]

    def record(self, weight: float, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self.entries.append((now, weight))
        self._prune(now)
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.entries, f)
            os.replace(tmp, self.path)

    def wait_for(self, weight: float, now: float | None = None, entries=None) -> float:
        """Seconds until `weight` fits in every window (0: now; inf: it never fits)."""
        now = self.clock() if now is None else now
        entries = sorted(self.entries if entries is None else entries)
        wait = 0.0
        for window, limit in zip(WINDOWS_S, self.limits):
            if weight > limit:
                return math.inf
            live = [(t, w) for t, w in entries if now - t < window]
            total = sum(w for _, w in live)
            for t, w in live:   # oldest first: the moment enough has expired
                if total + weight <= limit + 1e-9:
                    break
                total -= w
                wait = max(wait, t + window - now)
        return wait

    def simulate(self, weights: list[float]) -> float:
        """Seconds the given requests take when sent as early as the quota allows (inf if impossible)."""
        entries, now = list(self.entries), self.clock()
        start = now
        for w in weights:
            wait = self.wait_for(w, now, entries)
            if math.isinf(wait):
                return math.inf
            now += wait
            entries.append((now, w))
        return now - start


# ==================== on-disk store ====================
class ChunkStore:
    """One gzip JSON file per finished request: {"fetched": unix time, "points": {key: payload}}."""

    def __init__(self, directory: str):
        self.directory = directory

    def load(self, max_age_s: float, now: float) -> dict[str, dict]:
        points: dict[str, dict] = {}
        if not os.path.isdir(self.directory):
            return points
        for name in sorted(os.listdir(self.directory)):
            if not name.endswith(".json.gz"):
                continue
            try:
                with gzip.open(os.path.join(self.directory, name), "rt") as f:
                    chunk = json.load(f)
            except (OSError, ValueError):   # a half-written / corrupt file is just refetched
                continue
            if now - chunk["fetched"] <= max_age_s:
                points.update(chunk["points"])
        return points

    def save(self, points: dict[str, dict], now: float) -> None:
        os.makedirs(self.directory, exist_ok=True)
        name = os.path.join(self.directory, f"chunk_{int(now * 1000)}.json.gz")
        with gzip.open(name + ".tmp", "wt") as f:
            json.dump({"fetched": now, "points": points}, f)
        os.replace(name + ".tmp", name)


# ==================== backend ====================
class PacedOpenMeteoBackend(OpenMeteoBackend):
    RATE_LIMIT_RETRIES = 0     # limits are handled here, by pacing, not by `_query`'s blind one-minute retry
    NETWORK_RETRIES = 3

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, limits=DEFAULT_LIMITS, max_age_h: float = 6.0,
                 max_wait_s: float = 2 * 3600.0, plan_only: bool = False, log=print,
                 clock=time.time, sleep=time.sleep):
        self.cache_dir, self.max_age_s, self.max_wait_s = cache_dir, max_age_h * 3600.0, max_wait_s
        self.plan_only, self.log, self.clock, self.sleep = plan_only, log, clock, sleep
        os.makedirs(cache_dir, exist_ok=True)
        self.ledger = UsageLedger(os.path.join(cache_dir, "usage.json"), limits, clock)

    # ----- keys -----
    @staticmethod
    def _snap(source: DataSource, lat: float, lon: float) -> tuple[float, float]:
        step = GRID_DEG.get(source)
        if step:
            lat, lon = round(lat / step) * step, round(lon / step) * step
        return round(lat, 4) + 0.0, round(lon, 4) + 0.0   # + 0.0: no "-0.0"

    @staticmethod
    def _point_key(lat: float, lon: float) -> str:
        return f"{lat:.4f},{lon:.4f}"

    def _store_for(self, request, variables: list[str]) -> ChunkStore:
        ident = json.dumps([request._source.value, variables, request._start_date, request._end_date,
                            request._forecast_days, request._timezone])
        tag = hashlib.sha1(ident.encode()).hexdigest()[:10]
        return ChunkStore(os.path.join(self.cache_dir, f"{request._source.name}_{request._start_date or 'now'}_{tag}"))

    # ----- main entry (called by OpenMeteoBackend.fetch for the in-domain points) -----
    def _query_points(self, request, latitudes, longitudes) -> list[dict]:
        variables = request._expanded_hourly_variables()
        if request._start_date is not None:
            days = (date.fromisoformat(request._end_date or request._start_date)
                    - date.fromisoformat(request._start_date)).days + 1
        else:
            days = request._forecast_days
        per_point = max(1.0, len(variables) / CALL_VARIABLES_UNIT) * max(1.0, days / CALL_DAYS_UNIT)

        snapped = [self._snap(request._source, lat, lon) for lat, lon in zip(latitudes, longitudes)]
        keys = [self._point_key(*p) for p in snapped]
        unique = dict.fromkeys(keys)                                  # ordered, de-duplicated
        coords = {k: p for k, p in zip(keys, snapped)}

        store = self._store_for(request, variables)
        have = store.load(self.max_age_s, self.clock())
        missing = [k for k in unique if k not in have]
        self._report(len(keys), len(unique), len(unique) - len(missing), missing, per_point)
        if missing:
            have.update(self._download(request, store, missing, coords, per_point))
        return [have[k] for k in keys]

    # ----- planning -----
    def _chunk_size(self, per_point: float) -> int:
        return max(1, min(self.MAX_POINTS_PER_REQUEST, int(self.ledger.limits[0] // per_point)))

    def _report(self, n_requested: int, n_unique: int, n_cached: int, missing: list[str], per_point: float) -> None:
        size = self._chunk_size(per_point)
        chunks = [min(size, len(missing) - i) * per_point for i in range(0, len(missing), size)]
        self.log(f"Open-Meteo: {n_requested} points -> {n_unique} distinct model columns, {n_cached} cached, "
                 f"{len(missing)} to download")
        if missing:
            eta = self.ledger.simulate(chunks)
            when = "never (a request exceeds a quota; lower the number of points)" if math.isinf(eta) else \
                f"~{eta / 60:.0f} min if the quota windows allow" + \
                (" - the day limit is hit, so it will stop and resume on the next run"
                 if eta > self.max_wait_s else "")
            self.log(f"  {sum(chunks):.0f} weighted calls in {len(chunks)} requests of <= {size} points "
                     f"({per_point:.1f} calls/point): {when}")
        if self.plan_only:
            raise PlanOnly(0)

    # ----- downloading -----
    def _download(self, request, store: ChunkStore, missing: list[str], coords, per_point: float) -> dict[str, dict]:
        got: dict[str, dict] = {}
        total = math.ceil(len(missing) / self._chunk_size(per_point))
        done = 0
        while missing:
            size = self._chunk_size(per_point)   # shrinks when Open-Meteo reports the minute budget too high
            chunk, weight = missing[:size], min(size, len(missing)) * per_point
            wait = self.ledger.wait_for(weight)
            if wait > self.max_wait_s:
                raise DownloadIncomplete(
                    f"Open-Meteo quota: next request must wait {wait / 3600:.1f} h; {len(got)} points of this "
                    f"download are saved in {store.directory}, run the same command again to continue")
            if wait > 0:
                self.log(f"  waiting {wait:.0f} s for the quota window ({done}/{total} requests done) ...")
                self.sleep(wait)
            lats = [coords[k][0] for k in chunk]
            lons = [coords[k][1] for k in chunk]
            payloads = self._send(request, lats, lons, weight)
            if payloads is None:      # a quota error: the ledger now reflects it, so wait again
                continue
            now = self.clock()
            self.ledger.record(weight, now)
            part = dict(zip(chunk, payloads))
            store.save(part, now)
            got.update(part)
            missing = missing[size:]
            done += 1
            self.log(f"  request {done}/{total}: {len(chunk)} points saved ({len(got)} of {len(got) + len(missing)})")
        return got

    def _send(self, request, lats, lons, weight: float):
        """One request; None when Open-Meteo reported a quota limit (the ledger is adjusted to match)."""
        for attempt in range(self.NETWORK_RETRIES + 1):
            try:
                payloads = self._as_points(self._query(request, lats, lons))
            except requests.RequestException as exc:
                if attempt == self.NETWORK_RETRIES:
                    raise
                self.log(f"  network problem ({exc}); retrying in 10 s")
                self.sleep(10)
                continue
            except RuntimeError as exc:
                reason = str(exc)
                now = self.clock()
                if "Minutely" in reason:
                    self.ledger.limits[0] = max(weight / len(lats), self.ledger.limits[0] * 0.75)
                    self.ledger.record(self.ledger.limits[0], now)    # the minute is spent: wait for it to pass
                    self.log(f"  Open-Meteo says the minutely limit is reached; minute budget lowered to "
                             f"{self.ledger.limits[0]:.0f}")
                    return None
                if "Hourly" in reason or "Daily" in reason:
                    index = 1 if "Hourly" in reason else 2
                    self.ledger.record(self.ledger.limits[index], now)
                    self.log(f"  Open-Meteo says the {'hourly' if index == 1 else 'daily'} limit is reached")
                    return None
                raise
            if len(payloads) != len(lats):
                raise RuntimeError(f"Open-Meteo returned {len(payloads)} points for {len(lats)} requested")
            return payloads
