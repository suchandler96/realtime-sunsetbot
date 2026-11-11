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

`--data-source` is case-insensitive. `GFS` uses the Open-Meteo `gfs_seamless` blend. `GFS_AWS` downloads the plain
GFS run from NOAA's public AWS bucket (only the needed GRIB2 fields, cached in `gfs_aws/`); that data has cloud
liquid and ice water, so the figure shows the cloud phase (liquid / mixed / ice) and the light-ray transmittance
uses it. With Open-Meteo data (all other sources, or `GFS_AWS` when the download fails) there is no cloud phase and
the old temperature rule decides which cloud blocks light.

## Code layout

`main.py` is the command line and the figure. Everything that computes (transect geometry, the weather fetch, the
profile arrays, cloud-base detection, light-ray tracing and transmittance) lives in `sunset_core.py` and has no
plotting, so other tools can reuse it. `fetch_weather_batch()` fetches the profiles of one or many transects in one
batched request (Open-Meteo: chunked requests that wait out the minutely quota; AWS GFS: one set of GRIB fields
sampled at all points) and returns one `WeatherResult` per transect; a batch of one is the plain single fetch.

## Vividness map

`vividness_map.py` scores observers over a region (0-100) with the same ray tracing as `main.py` and overlays the
result on an `images/base_*.png` radar base map:

```
python vividness_map.py --region AECN --event sunset --date 2026-11-12 --data-source GFS_AWS --save
```

A point's score = local cloud cover x best 3-minute sliding mean (rays 0.5 min apart) of ray brightness
(green transmittance, logarithmic) x ray colour (red/blue transmittance); the thresholds are constants at the top of
the file. It says how well the cloud base over each point is lit; there is no smoothing over neighbouring points.
An observer with no level reaching the `CLOUD_PRESENT_*` constants (cloud cover / relative humidity) is skipped
(score 0) before anything else is computed; the cloud base of the others is found with `main.py`'s 50 % / 99 %
thresholds.

**Layout (`--layout`).** The default, `sunlines`, puts the observers on lines that follow the sun azimuth, so that
neighbours share one cross-section (`sunlines.py` describes it in detail):
lines start on the region's sun-facing edges every `--cross-km` (20), observers sit every `--along-km` (20, a
multiple of the 20 km model-column step) along each line, and observers within `--share-km` (100) of a group's
first observer use that observer's cross-section: weather and terrain are fetched once and each observer takes the
slice from its own position (terrain is read on a 1 km grid, so the slices are exact). A group never mixes two
forecast hours (AWS GFS gives one hour per request). The azimuth drifts with distance, so sharing is an
approximation: it is exact for the same points (checked). The shared line follows the azimuth of the group's middle
observer; against each observer's own azimuth (measured at 60 km spacing, no terrain) 93 % of the scores are
identical at `--share-km 100` and 2 of 790 differ by more than 5 points; at 300 km 30 of 791 do (a cloud edge can fall
on either side of a ray). On AECN at the defaults, 6900 observers need 5.0x fewer model
columns than one cross-section each (about 3 min in total, of which the sun-event calculation for every observer
takes about 80 s).
The cells are drawn rotated to the azimuth; where neighbouring lines converge they overlap (the higher score on top)
and where they diverge there are gaps. `--layout grid` is the older lat/lon grid (`--step-deg`), one cross-section
per point.

`--save` also writes the scores to a `.npz` that `--from-npz` redraws. Use `GFS_AWS`
for maps: Open-Meteo rate-limits big batches.

## E-mail service

`email/` answers forecast requests sent by e-mail; see `email/README.md`. `email/config.toml` holds
credentials and must not be committed; the same goes for `nsmc_credentials.txt` and `.nsmc_session.json`.
