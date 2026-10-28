"""Small dependency-free solar position / sunrise-sunset calculator.

Uses the NOAA solar position equations, which are accurate to a few minutes of
time and a fraction of a degree in azimuth -- plenty for selecting a transect
along the sunrise/sunset direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

# Standard sunrise/sunset is when the geometric centre of the Sun is 0.833
# degrees below the horizon (solar semi-diameter + typical refraction).
_SUN_RISE_SET_GEOMETRIC_ELEVATION = -0.833


@dataclass(frozen=True)
class SolarPosition:
    azimuth: float
    elevation: float              # apparent (refraction-corrected) centre
    geometric_elevation: float    # geometric centre
    declination: float = 0.0      # degrees; latitude of the sub-solar point


@dataclass(frozen=True)
class SunEvent:
    event: str                    # "sunrise" or "sunset"
    time: datetime                # aware datetime in `timezone_name`
    time_utc: datetime
    position: SolarPosition


def _julian_day(when: datetime) -> float:
    """Return the Julian Day for a UTC `datetime`."""
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc).replace(tzinfo=None)

    year, month = when.year, when.month
    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4
    day_fraction = (
        when.day
        + (when.hour + when.minute / 60.0 + when.second / 3600.0) / 24.0
    )
    return (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day_fraction
        + b
        - 1524.5
    )


def solar_position(latitude: float, longitude: float, when: datetime) -> SolarPosition:
    """Return the Sun's azimuth and elevation at a UTC `datetime`."""
    julian_day = _julian_day(when)
    julian_century = (julian_day - 2451545.0) / 36525.0

    # Geometric mean longitude and anomaly of the Sun.
    l0 = (280.46646 + julian_century * (36000.76983 + julian_century * 0.0003032)) % 360
    mean_anomaly = 357.52911 + julian_century * (35999.05029 - 0.0001537 * julian_century)
    eccentricity = 0.016708634 - julian_century * (0.000042037 + 0.0000001267 * julian_century)

    # Equation of centre and true longitude.
    m_rad = math.radians(mean_anomaly)
    equation_of_centre = (
        math.sin(m_rad) * (1.914602 - julian_century * (0.004817 + 0.000014 * julian_century))
        + math.sin(2 * m_rad) * (0.019993 - 0.000101 * julian_century)
        + math.sin(3 * m_rad) * 0.000289
    )
    true_longitude = l0 + equation_of_centre

    # Apparent longitude and obliquity of the ecliptic.
    omega = 125.04 - 1934.136 * julian_century
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    mean_obliquity = (
        23.0
        + 26.0 / 60.0
        + 21.448 / 3600.0
        - julian_century * (46.8150 / 3600.0)
        + julian_century**2 * (0.00059 / 3600.0)
        - julian_century**3 * (0.001813 / 3600.0)
    )
    obliquity = mean_obliquity + 0.00256 * math.cos(math.radians(omega))

    # Declination.
    declination = math.degrees(
        math.asin(math.sin(math.radians(obliquity)) * math.sin(math.radians(apparent_longitude)))
    )

    # Equation of time.
    y = math.tan(math.radians(obliquity / 2.0)) ** 2
    equation_of_time = 4.0 * math.degrees(
        y * math.sin(math.radians(2 * l0))
        - 2 * eccentricity * math.sin(m_rad)
        + 4 * eccentricity * y * math.sin(m_rad) * math.cos(math.radians(2 * l0))
        - 0.5 * y * y * math.sin(math.radians(4 * l0))
        - 1.25 * eccentricity * eccentricity * math.sin(math.radians(2 * mean_anomaly))
    )

    # Hour angle from true solar time.
    utc_minutes = when.hour * 60 + when.minute + when.second / 60.0
    true_solar_time = (utc_minutes + equation_of_time + 4 * longitude) % 1440
    hour_angle = true_solar_time / 4.0 - 180.0

    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)
    hour_angle_rad = math.radians(hour_angle)

    cos_zenith = (
        math.sin(lat_rad) * math.sin(dec_rad)
        + math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_angle_rad)
    )
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    geometric_elevation = 90.0 - math.degrees(math.acos(cos_zenith))

    # Atmospheric refraction correction (NOAA).
    if geometric_elevation > 85:
        refraction = 0.0
    elif geometric_elevation > 5:
        tan_e = math.tan(math.radians(geometric_elevation))
        refraction = (
            58.1 / tan_e
            - 0.07 / tan_e**3
            + 0.000086 / tan_e**5
        )
    elif geometric_elevation > -0.575:
        e = geometric_elevation
        refraction = 1735 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    else:
        refraction = -20.772 / math.tan(math.radians(geometric_elevation))
    elevation = geometric_elevation + refraction / 3600.0

    azimuth = math.degrees(
        math.atan2(
            math.sin(hour_angle_rad),
            math.cos(hour_angle_rad) * math.sin(lat_rad)
            - math.tan(dec_rad) * math.cos(lat_rad),
        )
    )
    azimuth = (azimuth + 180.0) % 360.0

    return SolarPosition(azimuth=azimuth, elevation=elevation, geometric_elevation=geometric_elevation,
                         declination=declination)


EARTH_ANGULAR_VELOCITY = 2.0 * math.pi / 86164.0905  # rad/s (sidereal rotation)


def last_light_delay(latitude: float, declination: float, r_earth: float, h: float, ha: float) -> timedelta:
    """Time after (ground) sunset at which the last ray still reaches a cloud base at height `h`.

    The last ray grazes the top of the aerosol layer (height `ha`, radius R + ha) before reaching the
    cloud base (radius R + h), so the Sun is then arccos(x) below the horizontal at the cloud, with
    x = (R + ha) / (R + h). With phi = latitude and alpha = declination (the sub-solar latitude):

        delay = [asin(tan(phi) tan(alpha) + sqrt(1 - x^2) sec(phi) sec(alpha)) - asin(tan(phi) tan(alpha))] / omega
    """
    phi, alpha = math.radians(latitude), math.radians(declination)
    x = (r_earth + ha) / (r_earth + h)
    base = math.tan(phi) * math.tan(alpha)
    late = base + math.sqrt(max(0.0, 1.0 - x * x)) / (math.cos(phi) * math.cos(alpha))
    if abs(base) > 1.0 or late > 1.0:  # polar day/night: the Sun never reaches that depression
        return timedelta(0)
    return timedelta(seconds=(math.asin(late) - math.asin(base)) / EARTH_ANGULAR_VELOCITY)


def grazing_height(latitude: float, declination: float, r_earth: float, h: float, delay: timedelta) -> float:
    """Lowest height (same unit as `r_earth`) of the ray reaching a cloud base at `h`, `delay` after sunset.

    Inverse of `last_light_delay`: that ray is tangent to the sphere of radius R + g with
    g = (R + h) cos(theta), where theta is the Sun's depression below the horizontal at the cloud.
    """
    phi, alpha = math.radians(latitude), math.radians(declination)
    base = math.tan(phi) * math.tan(alpha)
    if abs(base) > 1.0:
        return 0.0
    angle = math.asin(base) + EARTH_ANGULAR_VELOCITY * delay.total_seconds()
    sin_theta = (math.sin(min(angle, math.pi / 2)) - base) * math.cos(phi) * math.cos(alpha)
    return (r_earth + h) * math.sqrt(max(0.0, 1.0 - sin_theta * sin_theta)) - r_earth


def sun_event(
    latitude: float,
    longitude: float,
    day: date | str,
    event: str,
    timezone_name: str = "Asia/Hong_Kong",
) -> SunEvent:
    """Return the requested sunrise or sunset for a local calendar day.

    `event` must be `"sunrise"` or `"sunset"`.
    """
    if isinstance(day, str):
        day = date.fromisoformat(day)

    if event not in {"sunrise", "sunset"}:
        raise ValueError("event must be 'sunrise' or 'sunset'")

    tz = ZoneInfo(timezone_name)
    local_start = datetime.combine(day, time.min, tzinfo=tz)
    start_utc = local_start.astimezone(timezone.utc)
    end_utc = start_utc + timedelta(days=1)

    step = timedelta(minutes=1)
    previous_time: datetime | None = None
    previous_elevation: float | None = None
    candidates: list[tuple[datetime, bool]] = []  # (crossing_utc, rising)

    current = start_utc
    while current <= end_utc:
        geometric_elevation = solar_position(latitude, longitude, current).geometric_elevation
        if previous_elevation is not None:
            crosses = (
                previous_elevation <= _SUN_RISE_SET_GEOMETRIC_ELEVATION
                <= geometric_elevation
            ) or (
                previous_elevation >= _SUN_RISE_SET_GEOMETRIC_ELEVATION
                >= geometric_elevation
            )
            if crosses:
                span = geometric_elevation - previous_elevation
                if span != 0:
                    fraction = (_SUN_RISE_SET_GEOMETRIC_ELEVATION - previous_elevation) / span
                else:
                    fraction = 0.5
                crossing = previous_time + (current - previous_time) * fraction
                rising = geometric_elevation > previous_elevation
                candidates.append((crossing, rising))

        previous_time = current
        previous_elevation = geometric_elevation
        current += step

    want_rising = event == "sunrise"
    for crossing, rising in candidates:
        if rising == want_rising:
            position = solar_position(latitude, longitude, crossing)
            return SunEvent(
                event=event,
                time=crossing.astimezone(tz),
                time_utc=crossing,
                position=position,
            )

    raise RuntimeError(f"could not determine {event} for {day} at lat={latitude}, lon={longitude}")
