"""Observers laid out along sun lines, so that neighbours can share one cross-section.

An observer's transect runs from the observer towards the sun (the azimuth of the observer's own sunset or
sunrise).  Observers on one such line see the same atmosphere, only from different starting points, so one transect
can serve all of them: its weather and terrain are fetched / read once and each observer takes the slice that
starts at the observer's own position.  This module builds that layout; `vividness_map` scores it.

Layout
  * Lines start on the edges of the region that face the sun (an edge is a start edge when the sun direction at its
    middle points into the region), one line every `cross_km` measured across the sun direction.  Each line follows
    the sun azimuth of its start point, so the lines are "rotated" with the sun and bend with it over the region.
  * Observers sit every `along_km` along a line until it leaves the region.  `along_km` must be a multiple of the
    column step (sunset_core.STEP_KM), so that every observer falls on a column of the shared transect and the
    weather slices are exact.  (The shared terrain is read on a finer grid, see `vividness_map._SharedTerrain`.)
  * A group of consecutive observers shares one transect while (a) they lie within `share_km` of the group's first
    observer (the azimuth drifts over distance, so the sharing range is limited) and (b), for single-time weather
    (AWS GFS: one forecast hour per request), their events fall into the same forecast hour; weather of different
    snapshots is never shared.  The shared transect starts at the group's first observer and is laid in the
    direction of the sun azimuth of the group's middle observer (the mean of the two middle ones for an even
    count), which halves the azimuth drift to the ends compared with using the first one's.  The first observer
    outside the group starts the next group, one step further along the final line, with a transect of its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

import sunset_core as core
from solar import SunEvent, sun_event

KM_PER_DEG = 111.195          # km per degree of latitude (mean Earth radius 6371 km)
MIN_ENTRY = 0.05              # an edge is entered by the sun direction if its component across the edge exceeds this


@dataclass
class PointCity:
    """The few city attributes sunset_core needs, for an arbitrary grid point."""
    name: str
    latitude: float
    longitude: float
    timezone: str


@dataclass
class Observer:
    lat: float
    lon: float
    sun: SunEvent             # this observer's own event (time, azimuth, declination)
    line: int                 # index of the sun line
    along_km: float           # distance from the start of the line
    offset_km: float = 0.0    # distance from the anchor of the group (0 for the anchor itself)


@dataclass
class Group:
    """Observers sharing one transect; `transect` starts at the anchor (the first observer) at x = 0."""
    transect: core.Transect
    observers: list[Observer] = field(default_factory=list)
    snapshot: datetime | None = None     # UTC hour of the forecast the group needs (single-time weather)


def snapshot_hour(sun_time: datetime) -> datetime:
    """UTC hour of the forecast hour nearest to an event time (the hour an AWS GFS request is made for)."""
    return (sun_time.astimezone(timezone.utc) + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)


# ==================== start points ====================
def _inside(lat: float, lon: float, bbox, tolerance: float = 1e-9) -> bool:
    lon_min, lon_max, lat_min, lat_max = bbox
    return (lat_min - tolerance <= lat <= lat_max + tolerance) and (lon_min - tolerance <= lon <= lon_max + tolerance)


def start_points(bbox, cross_km: float, azimuth_at) -> list[tuple[float, float]]:
    """Start points of the sun lines on the region's sun-facing edges, `cross_km` apart across the sun direction.

    `bbox` is (lon_min, lon_max, lat_min, lat_max); `azimuth_at(lat, lon)` the sun azimuth in degrees.  The edges
    facing the sun are those whose inward normal has a positive component along the sun direction (the east edge
    when the sun is in the west, and so on); one such edge, or two for a diagonal sun."""
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_mid = 0.5 * (lat_min + lat_max)
    km_per_deg_lon = KM_PER_DEG * math.cos(math.radians(lat_mid))
    # (edge start, edge end, inward normal as (east, north))
    edges = [((lon_max, lat_min), (lon_max, lat_max), (-1.0, 0.0)),    # east edge
             ((lon_min, lat_min), (lon_min, lat_max), (1.0, 0.0)),     # west edge
             ((lon_min, lat_max), (lon_max, lat_max), (0.0, -1.0)),    # north edge
             ((lon_min, lat_min), (lon_max, lat_min), (0.0, 1.0))]     # south edge
    points = []
    for (lon0, lat0), (lon1, lat1), normal in edges:
        mid_lon, mid_lat = 0.5 * (lon0 + lon1), 0.5 * (lat0 + lat1)
        az = math.radians(azimuth_at(mid_lat, mid_lon))
        direction = (math.sin(az), math.cos(az))                      # (east, north)
        entry = direction[0] * normal[0] + direction[1] * normal[1]
        if entry <= MIN_ENTRY:
            continue
        length_km = math.hypot((lon1 - lon0) * km_per_deg_lon, (lat1 - lat0) * KM_PER_DEG)
        spacing_km = cross_km / entry                                  # along the edge, for cross_km across the sun
        k = 0
        while (k + 0.5) * spacing_km < length_km:
            u = (k + 0.5) * spacing_km / length_km
            points.append((lat0 + u * (lat1 - lat0), lon0 + u * (lon1 - lon0)))
            k += 1
    return points


# ==================== lines and groups ====================
def _observer(lat, lon, day, event, tz, line, along_km, offset_km, cache) -> Observer:
    key = (round(lat, 6), round(lon, 6))
    sun = cache.get(key)
    if sun is None:
        sun = cache[key] = sun_event(lat, lon, day, event, tz)
    return Observer(lat, lon, sun, line, along_km, offset_km)


def _make_group(members: list[Observer], day, event, tz, x_max, single_time) -> Group:
    """One transect from the first member through all of them, laid in the direction of the sun azimuth of the
    middle member (of the two middle ones, averaged, for an even count), so that the azimuth drift to either end of
    the group is as small as it can be.  The members then stand on the columns of this final line."""
    first = members[0]
    azimuth = 0.5 * (members[(len(members) - 1) // 2].sun.position.azimuth
                     + members[len(members) // 2].sun.position.azimuth)
    city = PointCity(f"{first.lat:.2f},{first.lon:.2f}", first.lat, first.lon, tz)
    length = members[-1].offset_km + x_max
    transect = core.make_transect(city, day, event, x_min=0, x_max=length, step_km=core.STEP_KM, sun=first.sun,
                                  azimuth=azimuth)
    for member in members:
        column = int(round(member.offset_km / core.STEP_KM))
        member.lat, member.lon = transect.lats[column], transect.lons[column]
    return Group(transect, members, snapshot_hour(first.sun.time) if single_time else None)


def _collect(start: Observer, bbox, day, event, tz, line, along_km, share_km, single_time, cache, log):
    """The observers that will share a transect with `start` (a provisional walk along his own azimuth: the final
    line differs from it by a fraction of a km), and the first observer after them, who starts the next group
    (None when the line leaves the region)."""
    members = [start]
    r_earth = core.earth_radius_km(start.lat)
    azimuth = start.sun.position.azimuth
    for k in range(1, int(2.5 * 20000 / along_km)):          # the bound only guards against a line that never ends
        offset = k * along_km
        lat, lon = core.get_point_at_distance(start.lat, start.lon, offset, azimuth, r_earth)
        if not _inside(lat, lon, bbox):
            return members, None
        try:
            candidate = _observer(lat, lon, day, event, tz, line, start.along_km + offset, offset, cache)
        except Exception as exc:
            log(f"line {line} ends at ({lat:.2f}, {lon:.2f}): {exc}")
            return members, None
        same_snapshot = (not single_time) or snapshot_hour(candidate.sun.time) == snapshot_hour(start.sun.time)
        if offset <= share_km and same_snapshot:
            members.append(candidate)
            continue
        candidate.offset_km = 0.0
        return members, candidate
    return members, None


def march_line(start, line: int, bbox, day: str, event: str, tz: str, along_km: float, share_km: float,
               x_max: float, single_time: bool, cache: dict, log=print) -> list[Group]:
    """Observers every `along_km` from `start` until the line leaves the region, grouped by sharing range/snapshot."""
    try:
        first = _observer(start[0], start[1], day, event, tz, line, 0.0, 0.0, cache)
    except Exception as exc:   # e.g. no sunset at this latitude
        log(f"line {line} skipped at ({start[0]:.2f}, {start[1]:.2f}): {exc}")
        return []
    groups: list[Group] = []
    while first is not None and len(groups) < int(2.5 * 20000 / along_km):
        members, following = _collect(first, bbox, day, event, tz, line, along_km, share_km, single_time, cache, log)
        group = _make_group(members, day, event, tz, x_max, single_time)
        groups.append(group)
        if following is not None:   # continue where the final line puts the next observer
            following.lat, following.lon = core.get_point_at_distance(
                first.lat, first.lon, members[-1].offset_km + along_km, group.transect.azimuth, group.transect.r_earth)
        first = following
    return groups


def build_layout(bbox, day: str, event: str, tz: str, cross_km: float = 20.0, along_km: float = 20.0,
                 share_km: float = 100.0, x_max: float = core.X_MAX, single_time: bool = True,
                 log=print) -> list[Group]:
    """All sun lines of a region as groups of observers; see the module docstring."""
    if along_km <= 0 or along_km % core.STEP_KM:
        raise ValueError(f"along_km must be a positive multiple of {core.STEP_KM} km (the step of the model columns "
                         f"of a shared transect), got {along_km}")
    if cross_km <= 0 or share_km < 0:
        raise ValueError("cross_km must be positive and share_km not negative")
    cache: dict = {}
    azimuth_at = lambda lat, lon: sun_event(lat, lon, day, event, tz).position.azimuth
    starts = start_points(bbox, cross_km, azimuth_at)
    groups: list[Group] = []
    for line, start in enumerate(starts):
        groups.extend(march_line(start, line, bbox, day, event, tz, along_km, share_km, x_max, single_time, cache, log))
    return groups
