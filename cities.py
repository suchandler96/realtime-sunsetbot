"""Major-city coordinates and IANA time zones.

A small lookup table so `main.py` (or any other script) can pick a known city
without hard-coding latitude/longitude/timezone.

    from cities import get_city

    city = get_city("Hong Kong")
    print(city.latitude, city.longitude, city.timezone)
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    timezone: str


CITIES = [
    City("Hong Kong", 22.3193, 114.1694, "Asia/Hong_Kong"),
    City("Shanghai", 31.2304, 121.4737, "Asia/Shanghai"),
    City("Beijing", 39.9042, 116.4074, "Asia/Shanghai"),
    City("Nanjing", 32.0603, 118.7969, "Asia/Shanghai"),
    City("Tokyo", 35.6762, 139.6503, "Asia/Tokyo"),
    City("Taipei", 25.0330, 121.5654, "Asia/Taipei"),
    City("Seoul", 37.5665, 126.9780, "Asia/Seoul"),
    City("Singapore", 1.3521, 103.8198, "Asia/Singapore"),
    City("Bangkok", 13.7563, 100.5018, "Asia/Bangkok"),
    City("London", 51.5074, -0.1278, "Europe/London"),
    City("New York", 40.7128, -74.0060, "America/New_York"),
    City("Los Angeles", 34.0522, -118.2437, "America/Los_Angeles"),
]

CITIES_BY_NAME = {city.name: city for city in CITIES}


def get_city(name: str) -> City:
    """Return a `City` by exact name (case-insensitive)."""
    key = name.strip().lower()
    for city in CITIES:
        if city.name.lower() == key:
            return city
    raise KeyError(f"unknown city '{name}' (available: {', '.join(c.name for c in CITIES)})")


def lookup_timezone(latitude: float, longitude: float) -> str:
    """IANA time zone of a point, from Open-Meteo; falls back to a fixed offset from the longitude.

    The fallback (``Etc/GMT-8`` for 120 E, ...) has no daylight saving and ignores political
    borders, so pass an explicit zone when it matters.
    """
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, "timezone": "auto",
                    "daily": "sunrise", "forecast_days": 1},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["timezone"]
    except (requests.RequestException, KeyError, ValueError):
        hours = round(longitude / 15.0)
        return f"Etc/GMT{-hours:+d}" if hours else "UTC"


def custom_city(latitude: float, longitude: float, name: str | None = None,
                timezone: str | None = None) -> City:
    """An observer at arbitrary coordinates, shaped like the entries of ``CITIES``.

    ``name`` is used in the figure's title and file name (default: the coordinates);
    ``timezone`` is looked up from the coordinates when omitted.
    """
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude {latitude} is outside -90..90")
    if not -180.0 <= longitude <= 180.0:
        longitude = ((longitude + 180.0) % 360.0) - 180.0  # accept 0..360 too
    if name is None:
        name = (f"{abs(latitude):.2f}{'N' if latitude >= 0 else 'S'}_"
                f"{abs(longitude):.2f}{'E' if longitude >= 0 else 'W'}")
    return City(name, latitude, longitude, timezone or lookup_timezone(latitude, longitude))
