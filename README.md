# realtime-sunsetbot

Plots a vertical cross-section of the atmosphere along the direction of the sunrise / sunset
(relative humidity, cloud cover, temperature, optional CAMS aerosol, terrain, FY-4B satellite and
NMC radar strips) and traces the sunlight rays that reach the cloud base above the observer.

## Run

```bash
uv sync
uv run main.py -h
uv run main.py --city Shanghai --event sunset --date 2026-11-06
```

Optional features need extra packages that are not in `pyproject.toml` yet: `cdsapi` (`--aerosol`),
`opencv-python`, `pytesseract` plus the system `tesseract` binary (automatic FY-4B captcha login).

## Choosing the place

- `--city NAME` accepts the English name (any case) or the Chinese name: `Shanghai`, `shanghai`, `上海`.
  Cities in `cities.py` are used directly. Other **Chinese** cities (mainland, Hong Kong, Macao, Taiwan)
  are looked up online (Open-Meteo geocoding) and remembered in `geocode_cache.json` (UTF-8).
- An English name shared by several Chinese cities (e.g. Fuzhou, Suzhou, Taizhou) is refused and the
  candidates are listed: use the Chinese name instead (`福州`, `苏州`). An ambiguous Chinese name is
  also refused; use `--lat/--lon`.
- `--lat/--lon [--place-name] [--timezone]` for any other place.
- Figure title uses the Chinese name for Chinese cities (a CJK font is needed; Noto Sans CJK, Microsoft YaHei,
  SimHei ... are tried). The file name always keeps the English name.

## Output

`{date}_{city}_{event}_{model}_{update}.png`; the title adds the Sun's azimuth after the event, e.g.
`20261104_杭州_sunset(252.6°)_GFS_2026110500z`.

## Light rays and transmittance

Rays reach the local cloud base (automatically detected from the forecast, or `--cloud-base KM`). With no
cloud above the observer and no `--cloud-base`, no rays are drawn and nothing is integrated.

- The earliest ray grazes the ground, the latest grazes the aerosol top (or terrain, if that blocks it);
  intermediate rays are drawn every 5 minutes.
- `--transmittance` (default on) integrates Beer-Lambert extinction along each ray (`raytrace.py`):
  Rayleigh scattering (wavelength and density dependent), aerosol (CAMS, scaled as lambda^-1.2; needs
  `--aerosol`) and cloud. Water-bearing or mixed-phase cloud beyond the target cell blocks the direct beam
  (the ray is labelled "blocked" and drawn black); cold ice-only cloud has a finite extinction.
  This is direct sunlight only; scattered light is not modelled and the forecast cloud cover is
  an occupancy estimate, not a water-content field.
- The square swatch at the end of each ray is the colour of the light reaching the cloud (R/G/B transmittance).
- `--ray-min-transmittance` (default 0.001) is the direct-beam threshold below which a ray is reported as
  "<0.1%" in the labels and the console table. It is a reporting threshold, not a sunset-glow boundary;
  the first and last ray are always drawn.

## Other options

`--data-source` (GFS, ECMWF, HRRR, ...), `--aerosol` / `--aerosol-wavelength`, `--radar`, `--satellite`
(FY-4B cloud mask and optical thickness strip), `--cth-method`, `--save`, `--silent`; see `-h`.

## E-mail service

`email/` answers forecast requests sent by e-mail; see `email/README.md`. `email/config.toml` holds
credentials and must not be committed; the same goes for `nsmc_credentials.txt` and `.nsmc_session.json`.
