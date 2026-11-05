"""Major-city coordinates and IANA time zones.

A small lookup table so `main.py` (or any other script) can pick a known city
without hard-coding latitude/longitude/timezone.

    from cities import get_city

    city = get_city("Hong Kong")
    print(city.latitude, city.longitude, city.timezone)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class City:
    name: str
    latitude: float
    longitude: float
    timezone: str
    name_zh: str = ""   # Chinese name, accepted as input as well

    @property
    def title_name(self) -> str:
        """Name for figure titles: Chinese characters for cities in China / Hong Kong / Macao / Taiwan."""
        in_china = self.timezone in ("Asia/Shanghai", "Asia/Hong_Kong", "Asia/Macau", "Asia/Taipei", "Asia/Urumqi")
        return self.name_zh if in_china and self.name_zh else self.name


class CityLookupError(ValueError):
    """The city is unknown, ambiguous or could not be looked up; the message is meant for the user."""


CITIES = [
    City("Hong Kong", 22.3193, 114.1694, "Asia/Hong_Kong", "香港"),
    City("Shanghai", 31.2304, 121.4737, "Asia/Shanghai", "上海"),
    City("Beijing", 39.9042, 116.4074, "Asia/Shanghai", "北京"),
    City("Nanjing", 32.0603, 118.7969, "Asia/Shanghai", "南京"),
    City("Chengdu", 30.6667, 104.0667, "Asia/Shanghai", "成都"),
    City("Tokyo", 35.6762, 139.6503, "Asia/Tokyo", "东京"),
    City("Taipei", 25.0330, 121.5654, "Asia/Taipei", "台北"),
    City("Seoul", 37.5665, 126.9780, "Asia/Seoul", "首尔"),
    City("Singapore", 1.3521, 103.8198, "Asia/Singapore", "新加坡"),
    City("Bangkok", 13.7563, 100.5018, "Asia/Bangkok", "曼谷"),
    City("London", 51.5074, -0.1278, "Europe/London", "伦敦"),
    City("New York", 40.7128, -74.0060, "America/New_York", "纽约"),
    City("Los Angeles", 34.0522, -118.2437, "America/Los_Angeles", "洛杉矶"),
]

CITIES_BY_NAME = {city.name: city for city in CITIES}


def get_city(name: str) -> City:
    """Return a table `City` by English (case-insensitive) or Chinese name."""
    key = name.strip().lower()
    for city in CITIES:
        if city.name.lower() == key or (city.name_zh and city.name_zh == name.strip()):
            return city
    raise KeyError(f"unknown city '{name}' (available: {', '.join(c.name for c in CITIES)})")


# ---- cities outside the table: Open-Meteo geocoding (Chinese cities for now) ----
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
GEOCODING_BY_ID_URL = "https://geocoding-api.open-meteo.com/v1/get"
CHINESE_REGIONS = ("CN", "HK", "MO", "TW")
CITY_FEATURES = ("PPLC", "PPLA", "PPLA2", "PPLA3")   # capital, provincial / prefecture / county-level seat
MIN_CITY_POPULATION = 300_000   # drops same-named districts and villages (e.g. Xi'an in Heilongjiang)
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geocode_cache.json")  # UTF-8 JSON


def _is_chinese(text: str) -> bool:
    return re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text) is not None


def _fold(text: str) -> str:
    text = text.replace("\u2019", "'").strip().lower()   # Xi’an == Xi'an
    return text[:-1] if len(text) > 2 and text.endswith("\u5e02") else text   # 泰州市 == 泰州


def _ascii_name(name: str) -> str:
    """Name for the figure title / file name (matplotlib's default font has no Chinese glyphs)."""
    return re.sub(r"[^A-Za-z0-9 ]", "", name).strip()


def _geocode(url: str, **params) -> list[dict]:
    try:
        resp = requests.get(url, params={**params, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise CityLookupError(f"cannot look up the city (geocoding service unreachable: {exc}); "
                              "use --lat/--lon instead") from exc
    return data.get("results", [data] if "id" in data else [])


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    except OSError:
        pass


def _describe(record_id: int, fallback: dict) -> str:
    """'福建 福州' for an ambiguity message, in Chinese when the service has it."""
    zh = (_geocode(GEOCODING_BY_ID_URL, id=record_id, language="zh") or [fallback])[0]
    return f"{zh.get('name')} ({zh.get('admin1', '')}, {zh['latitude']:.2f}, {zh['longitude']:.2f})"


def geocode_chinese_city(query: str) -> City:
    """A Chinese city by Chinese or English name.  English names that fit several cities
    (Fuzhou, Suzhou, Taizhou ...) are refused: the message lists them so the Chinese name can be used."""
    chinese = _is_chinese(query)
    wanted = _fold(query)
    names = [query.strip()]
    if chinese and not names[0].endswith("\u5e02"):
        names.append(names[0] + "\u5e02")   # the service stores prefecture-level cities as 泰州市, not 泰州
    found = {}
    for name in names:
        for r in _geocode(GEOCODING_URL, name=name, count=30, language="zh" if chinese else "en"):
            found[r["id"]] = r
    results = [r for r in found.values()
               if r.get("country_code") in CHINESE_REGIONS and _fold(r.get("name", "")) == wanted]
    candidates = [r for r in results if r.get("feature_code") in CITY_FEATURES]
    pool = [r for r in candidates if (r.get("population") or 0) >= MIN_CITY_POPULATION] or candidates
    if not pool:  # a small town: accept it only when the name is unique
        pool = [r for r in results if r.get("feature_code", "").startswith("PPL")]
    places = {}
    for r in pool:   # one entry per province/region: the service lists some cities twice
        places.setdefault(r.get("admin1") or r["id"], r)
    if not places:
        raise CityLookupError(f"unknown city {query!r}: not in cities.py and not found as a Chinese city; "
                              "use --lat/--lon instead")
    if len(places) > 1:
        options = "; ".join(_describe(r["id"], r) for r in places.values())
        if chinese:
            raise CityLookupError(f"{query!r} matches several places: {options}. Use --lat/--lon.")
        raise CityLookupError(f"the English name {query!r} matches several Chinese cities: {options}. "
                              "Use the Chinese name (or --lat/--lon).")
    hit = next(iter(places.values()))
    english = hit if not chinese else (_geocode(GEOCODING_BY_ID_URL, id=hit["id"], language="en") or [hit])[0]
    name = _ascii_name(english.get("name", "")) or custom_city(hit["latitude"], hit["longitude"]).name
    timezone = hit.get("timezone") or lookup_timezone(hit["latitude"], hit["longitude"])
    zh = hit if chinese else (_geocode(GEOCODING_BY_ID_URL, id=hit["id"], language="zh") or [hit])[0]
    name_zh = zh.get("name", "")
    name_zh = name_zh[:-1] if len(name_zh) > 2 and name_zh.endswith("\u5e02") else name_zh   # 福州市 -> 福州
    return City(name, hit["latitude"], hit["longitude"], timezone, name_zh if _is_chinese(name_zh) else "")


def resolve_city(query: str) -> City:
    """City table first (English case-insensitive, or Chinese), then the geocoding service (cached).
    Raises CityLookupError when unknown or ambiguous."""
    query = query.strip()
    if not query:
        raise CityLookupError("empty city name")
    try:
        return get_city(query)
    except KeyError:
        pass
    key = _fold(query)
    cache = _load_cache()
    if key in cache:
        return City(**cache[key])
    city = geocode_chinese_city(query)
    cache[key] = {"name": city.name, "latitude": city.latitude, "longitude": city.longitude,
                  "timezone": city.timezone, "name_zh": city.name_zh}
    _save_cache(cache)
    return city


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
