"""Direct-beam transmittance through the forecast cross-section.

The ray is a curve y(x) in the flattened (distance, altitude) plane.  Rayleigh, aerosol and cold-ice-cloud
optical depths are integrated with Beer-Lambert at representative red, green and blue wavelengths.  NWP cloud
cover is an occupancy estimate rather than cloud-water concentration: the model cell containing the target
cloud base is excluded, later water-bearing cloud blocks the direct beam, and cold ice-only cloud has a
finite wavelength-neutral extinction.  When the forecast carries cloud liquid and ice mixing ratios (GFS from
AWS) the phase comes from them instead of from temperature: each cloud's extinction is computed from its liquid
and ice mass, and only liquid cloud dense enough to be opaque blocks the beam.  This computes direct sunlight only, not diffuse light produced by
cloud multiple scattering.
"""
from dataclasses import dataclass

import numpy as np

AIR_REFRACTIVE_INDEX = 1.00029   # sea-level air, visible light
AIR_NUMBER_DENSITY = 2.504e25    # molecules per m^3 at standard sea level


def rayleigh_sea_level_per_km(wavelength_nm):
    """Rayleigh extinction at standard sea level (km^-1): 8 pi^3 (n^2 - 1)^2 / (3 N lambda^4)."""
    wavelength_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    per_m = 8 * np.pi**3 * (AIR_REFRACTIVE_INDEX**2 - 1) ** 2 / (3 * AIR_NUMBER_DENSITY * wavelength_m**4)
    return per_m * 1e3


RAYLEIGH_SCALE_HEIGHT_KM = 8.0
AEROSOL_ANGSTROM = 1.2         # aerosol extinction ~ lambda^-1.2 (observed exponent range 0.07 .. 3.803)
RGB_NM = (650.0, 550.0, 450.0)  # wavelengths standing in for the red, green and blue primaries
REFERENCE = 1                  # index of the wavelength (green, 550 nm) used for the headline T and its threshold
DEFAULT_CLOUD_BLOCK_COVER_PERCENT = 50.0
ICE_ONLY_TEMP_C = -20.0        # colder cloud is treated as ice-only; warmer/mixed cloud blocks direct light
ICE_EXTINCTION_PER_KM = 0.05   # optically thin cirrus fallback when the forecast has no ice water content

# Cloud-water path used when the forecast has liquid/ice mixing ratios: extinction = mass extinction coefficient
# (3 / (2 rho r_eff)) x water content.  Effective radii: 10 um for droplets, 40 um for ice crystals.
LIQUID_MASS_EXTINCTION_M2_PER_KG = 150.0
ICE_MASS_EXTINCTION_M2_PER_KG = 40.0
LIQUID_BLOCK_PER_KM = 1.0      # liquid-cloud extinction (km^-1) from which the direct beam counts as blocked
DRY_AIR_GAS_CONSTANT = 287.05


def condensate_extinction(pressure_hpa, temperature_c, liquid, ice, cloud_cover):
    """In-cloud (liquid, ice) extinction in km^-1 from grid-mean mixing ratios (kg/kg).

    The grid-box mean is divided by the cloud fraction (at least 0.5: blocking needs >= 50 % cover anyway) to
    get the in-cloud value.  All arguments share one shape; pressure may broadcast against it."""
    density = np.asarray(pressure_hpa, dtype=float) * 100.0 / (
            DRY_AIR_GAS_CONSTANT * (np.nan_to_num(temperature_c, nan=0.0) + 273.15))
    in_cloud = 1.0 / np.clip(np.nan_to_num(cloud_cover, nan=0.0) / 100.0, 0.5, 1.0)
    scale = 1e3 * density * in_cloud     # kg/kg -> kg/m^3 in cloud, m^-1 -> km^-1
    return (np.clip(np.nan_to_num(liquid, nan=0.0), 0.0, None) * LIQUID_MASS_EXTINCTION_M2_PER_KG * scale,
            np.clip(np.nan_to_num(ice, nan=0.0), 0.0, None) * ICE_MASS_EXTINCTION_M2_PER_KG * scale)


@dataclass
class RayTransmittance:
    rgb_transmittance: tuple     # direct-beam fractions at the red / green / blue wavelengths
    tau_cloud: float             # ice-cloud optical depth, or infinity when water-bearing cloud blocks the ray
    tau_aerosol: float
    tau_rayleigh: float
    stopped_at_km: float | None  # x where cloud blocked the ray or clear-air tau reached tau_stop
    blocked_by_cloud: bool = False

    @property
    def transmittance(self) -> float:
        return self.rgb_transmittance[REFERENCE]

    @property
    def tau(self) -> float:
        return self.tau_cloud + self.tau_aerosol + self.tau_rayleigh

    @property
    def light_colour(self) -> tuple:
        """Approximate sRGB including direct-beam dimming; a blocked ray is black."""
        t = np.clip(np.asarray(self.rgb_transmittance, dtype=float), 0.0, 1.0)
        return tuple(float(c) for c in t ** (1 / 2.2))

    @property
    def normalized_light_colour(self) -> tuple:
        """Exposure-normalized chromaticity for colour comparisons, independent of brightness."""
        t = np.clip(np.asarray(self.rgb_transmittance, dtype=float), 0.0, 1.0)
        peak = float(t.max())
        if peak == 0.0:
            return 0.0, 0.0, 0.0
        return tuple(float(c) for c in (t / peak) ** (1 / 2.2))


class ProfileExtinction:
    """Cloud occupancy and clear-air extinction fields on the forecast cross-section.

    heights_km: (levels, points) heights of the cloud-cover and temperature fields.
    aerosol_*: optional CAMS extinction (levels, points) at `wavelength_nm` with its own heights (levels, points).
    liquid_extinction / ice_extinction: optional in-cloud extinction (levels, points, km^-1) of cloud liquid and ice
    from `condensate_extinction`; with them the cloud phase is not guessed from temperature."""

    def __init__(self, distances, heights_km, cloud_cover, temperature=None, underground=None,
                 aerosol_extinction=None, aerosol_heights_km=None, wavelength_nm=532,
                 cloud_block_cover_percent=DEFAULT_CLOUD_BLOCK_COVER_PERCENT,
                 ice_extinction_per_km=ICE_EXTINCTION_PER_KM, liquid_extinction=None, ice_extinction=None):
        self.distances = np.asarray(distances, dtype=float)
        cloud = np.clip(np.nan_to_num(cloud_cover, nan=0.0), 0.0, 100.0)
        if underground is not None:  # extrapolated data below the model terrain is not real cloud
            cloud = np.where(underground, 0.0, cloud)
        self._cloud_cover = self._columns(heights_km, cloud)
        if temperature is None:
            temperature = np.full_like(cloud, 0.0)
        self._temperature = self._columns(heights_km, np.nan_to_num(temperature, nan=0.0))
        self.cloud_block_cover_percent = float(cloud_block_cover_percent)
        self.ice_extinction_per_km = float(ice_extinction_per_km)
        self._liquid = self._ice = None
        if liquid_extinction is not None and ice_extinction is not None:
            self._liquid = self._columns(heights_km, np.nan_to_num(liquid_extinction, nan=0.0))
            self._ice = self._columns(heights_km, np.nan_to_num(ice_extinction, nan=0.0))
        self._aerosol = None
        if aerosol_extinction is not None:
            self._aerosol = self._columns(aerosol_heights_km, np.nan_to_num(aerosol_extinction, nan=0.0))
        wavelengths = np.asarray(RGB_NM)
        self._aerosol_scale = (wavelength_nm / wavelengths) ** AEROSOL_ANGSTROM
        self._rayleigh_scale = rayleigh_sea_level_per_km(wavelengths)

    @staticmethod
    def _columns(heights, values):
        out = []
        for j in range(np.shape(heights)[1]):
            order = np.argsort(heights[:, j])
            out.append((np.asarray(heights)[order, j], np.asarray(values)[order, j]))
        return out

    def _at(self, columns, x, y):
        """Field value at (x, y): linear in x between neighbouring columns, linear in height in each column;
        zero outside the model's height range."""
        f = (x - self.distances[0]) / (self.distances[1] - self.distances[0])
        j = int(np.clip(np.floor(f), 0, len(self.distances) - 2))
        w = float(np.clip(f - j, 0.0, 1.0))
        a, b = columns[j], columns[j + 1]
        return (1 - w) * np.interp(y, a[0], a[1], left=0.0, right=0.0) + w * np.interp(y, b[0], b[1], left=0.0, right=0.0)

    def _nearest_column_at(self, columns, x, y):
        """A field in the nearest horizontal model cell, interpolated only in height."""
        j = int(np.argmin(np.abs(self.distances - x)))
        heights, values = columns[j]
        return float(np.interp(y, heights, values, left=0.0, right=0.0))

    def _target_cell_exit(self, x):
        """Right-hand boundary of the horizontal model cell containing the ray endpoint."""
        midpoints = 0.5 * (self.distances[:-1] + self.distances[1:])
        j = int(np.searchsorted(midpoints, x, side="right"))
        return float(midpoints[j]) if j < len(midpoints) else float(self.distances[-1])

    def extinction(self, x, y):
        """Fields at one point as (cloud cover percent, temperature C, aerosol[rgb], Rayleigh[rgb])."""
        cloud = self._nearest_column_at(self._cloud_cover, x, y)
        temperature = self._nearest_column_at(self._temperature, x, y)
        aerosol = self._at(self._aerosol, x, y) * self._aerosol_scale if self._aerosol is not None else np.zeros(3)
        rayleigh = self._rayleigh_scale * np.exp(-max(y, 0.0) / RAYLEIGH_SCALE_HEIGHT_KM)
        return cloud, temperature, aerosol, rayleigh

    def integrate(self, ray_x, ray_y, x_max=None, step_km=1.0, tau_stop=4.6, top_km=None):
        """Optical depth along a ray y(x), starting at x = ray_x[0] (the cloud base) outwards.

        ray_x, ray_y: the ray on any grid (interpolated to `step_km`).  The walk ends at `x_max` or when the ray
        leaves the model above `top_km` (default the highest model level).  The first position where reference
        optical depth exceeds tau_stop is recorded, but cloud checks continue over the remaining path.

        The horizontal model cell containing the starting point is the target cloud and is ignored.  In later
        cells, cloud cover at or above `cloud_block_cover_percent` blocks the ray when warmer than
        `ICE_ONLY_TEMP_C`; colder cloud contributes the finite `ice_extinction_per_km` instead.  With liquid / ice
        extinction fields, such a cell blocks the ray when its liquid extinction reaches `LIQUID_BLOCK_PER_KM` and
        otherwise adds its liquid + ice extinction."""
        ray_x, ray_y = np.asarray(ray_x, dtype=float), np.asarray(ray_y, dtype=float)
        x_end = min(ray_x[-1], x_max if x_max is not None else ray_x[-1], self.distances[-1])
        if top_km is None:
            top_km = max(c[0][-1] for c in self._cloud_cover)
        xs = np.arange(ray_x[0], x_end, step_km)
        if len(xs) == 0 or xs[-1] < x_end:
            xs = np.append(xs, x_end)
        ys = np.interp(xs, ray_x, ray_y)
        tau_cloud, tau_aerosol, tau_rayleigh = 0.0, np.zeros(3), np.zeros(3)
        target_cell_exit = self._target_cell_exit(float(xs[0]))

        def at(i):
            cloud, temperature, aerosol, rayleigh = self.extinction(xs[i], ys[i])
            occupied = xs[i] > target_cell_exit and cloud >= self.cloud_block_cover_percent
            if self._liquid is not None:   # phase and amount from the model's liquid / ice water
                beta_liquid = self._nearest_column_at(self._liquid, xs[i], ys[i])
                beta_ice = self._nearest_column_at(self._ice, xs[i], ys[i])
                water_block = occupied and beta_liquid >= LIQUID_BLOCK_PER_KM
                ice_beta = beta_liquid + beta_ice if occupied and not water_block else 0.0
            else:
                ice_beta = self.ice_extinction_per_km if occupied and temperature <= ICE_ONLY_TEMP_C else 0.0
                water_block = occupied and temperature > ICE_ONLY_TEMP_C
            return ice_beta, water_block, aerosol, rayleigh

        def result(stopped_at, blocked_by_cloud=False):
            tau = tau_cloud + tau_aerosol + tau_rayleigh
            transmittance = np.zeros(3) if blocked_by_cloud else np.exp(-tau)
            return RayTransmittance(
                tuple(float(t) for t in transmittance),
                float("inf") if blocked_by_cloud else float(tau_cloud),
                float(tau_aerosol[REFERENCE]),
                float(tau_rayleigh[REFERENCE]),
                stopped_at,
                blocked_by_cloud,
            )

        prev = at(0)
        threshold_at = None
        for i in range(1, len(xs)):
            if ys[i] > top_km:
                break
            cur = at(i)
            ds = np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            tau_cloud += 0.5 * (prev[0] + cur[0]) * ds
            tau_aerosol += 0.5 * (prev[2] + cur[2]) * ds
            tau_rayleigh += 0.5 * (prev[3] + cur[3]) * ds
            prev = cur
            if cur[1]:
                return result(float(xs[i]), blocked_by_cloud=True)
            if threshold_at is None and tau_cloud + tau_aerosol[REFERENCE] + tau_rayleigh[REFERENCE] > tau_stop:
                threshold_at = float(xs[i])
        return result(threshold_at)
