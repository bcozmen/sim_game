import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from .helper import (fbm, ridged_fbm, anisotropic_ridged_fbm, domain_warp,
                     thermal_erosion, hydraulic_erosion, particle_hydraulic_erosion,
                     flow_accumulation, carve_rivers,
                     widen_valleys, alluvial_deposition, rain_shadow)

class Terrain:
    def __init__(self,scale=8.0, octaves=6, persistence=0.5, lacunarity=2.0, seed=0):
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed

    def grid(self, lim, w, h):
        xmin, xmax, ymin, ymax = lim
        X, Y = np.meshgrid(np.linspace(xmin, xmax, w), np.linspace(ymin, ymax, h))
        return X, Y

    def slope_based_height_reduction(self, terrain, lim, alpha=1.0):
        """
        Damp high-frequency ridge peaks using local slope.
        Gradient is computed with the actual coordinate spacing derived from `lim`
        so the damping is identical regardless of LOD zoom level.
        """
        xmin, xmax, ymin, ymax = lim
        h, w = terrain.shape
        pixel_dx = (xmax - xmin) / w
        pixel_dy = (ymax - ymin) / h

        # pass real coordinate spacings so gradient is in height-per-coord-unit
        grad_y, grad_x = np.gradient(terrain, pixel_dy, pixel_dx)
        slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
        return terrain * np.exp(-alpha * slope)

    def generate(self,
                 lim=(0, 1, 0, 1),
                 width=512, height=512,
                 warp_strength=0,
                 thermal_iters=0,
                 talus=0.02,
                 thermal_strength=0.5,
                 hydraulic_iters=0,
                 rain=0.01,
                 evap=0.02,
                 erode=0.1,
                 deposit=0.1,
                 # --- particle erosion ---
                 particle_droplets=0,
                 particle_max_path=0.3,
                 particle_inertia=0.4,
                 particle_capacity=8.0,
                 particle_min_slope=0.002,
                 particle_erode_speed=0.3,
                 particle_deposit_speed=0.3,
                 particle_evaporate=0.02,
                 particle_gravity=10.0,
                 # --- layered generation ---
                 continent_weight=0.45,
                 mountain_weight=0.45,
                 detail_weight=0.10,
                 continent_freq_factor=0.15,
                 # --- tectonic ridge direction ---
                 # ridge_angle    : strike of mountain chains in radians (0 = E-W chains).
                 # ridge_anisotropy > 1 creates elongated chains; 1 = isotropic (= ridged_fbm).
                 ridge_angle=0.0,
                 ridge_anisotropy=1.0,
                 # --- drainage / river carving ---
                 flow_threshold=0,
                 river_depth=0.06,
                 river_slope_power=0.45,
                 # --- valley widening (lateral erosion) ---
                 # valley_sigma=0 disables it.
                 valley_sigma=0.0,
                 valley_depth=0.025,
                 # --- alluvial deposition ---
                 # alluvial_amount=0 disables it.
                 alluvial_amount=0.0,
                 alluvial_lowland_cutoff=0.35,
                 # --- post-process ---
                 # gaussian_sigma destroys erosion detail; keep at 0 or very small (≤0.5).
                 gaussian_sigma=0.0,
                 # slope_reduction_alpha=0 disables legacy ridge damping.
                 slope_reduction_alpha=0.0
                 ):
        """
        Generate a terrain heightmap for the coordinate region `lim`.

        HEIGHT-SCALE CONSISTENCY
        ------------------------
        All noise layers are normalised using *fixed theoretical bounds* derived
        from the layer weights and gradient-noise range (≈ ±0.7071), not from the
        per-grid min/max.  This guarantees that the same world coordinate always
        maps to the same base height regardless of the resolution or extent of the
        grid being generated — a 10 km sub-patch will have values consistent with
        the 100 km world map that contains it.

        Erosion is a local, non-linear process that can shift values slightly
        (typically ±0.02 – 0.05) but the coordinate-dependent base is preserved.

        EROSION ORDER  (matches geological time-ordering)
        --------------------------------------------------
          1. Thermal erosion       — scree / talus
          2. Particle hydraulic    — channel incision + fine detail
          3. Flow accumulation     — drainage network
          4. River carving         — V-valleys / canyons (stream-power law)
          5. Valley widening       — lateral erosion / floodplain
          6. Alluvial deposition   — fans, deltas, valley fill
          (gaussian_sigma should remain 0 or ≤ 0.5 to preserve erosion detail)
        """
        xmin, xmax, ymin, ymax = lim
        pixel_dx = (xmax - xmin) / width
        pixel_dy = (ymax - ymin) / height

        X, Y = self.grid(lim, width, height)

        if warp_strength > 0:
            X, Y = domain_warp(
                X, Y,
                warp_strength,
                self.octaves,
                self.persistence,
                self.lacunarity,
                self.scale,
                self.seed
            )

        # ----------------------------------------------------------
        # LAYERED GENERATION
        #
        # All layers are normalised to [0, 1] using FIXED bounds so
        # that any sub-region produces values consistent with the full
        # world map (no per-grid min/max normalisation).
        #
        # gradient_noise output range ≈ ±0.7071 (1/√2 scaling).
        # fbm normalises by the sum of octave amplitudes, so the
        # output is also bounded by ±0.7071.
        # ----------------------------------------------------------
        _GN_MAX = 0.7071   # theoretical half-range of gradient_noise

        # 1. Continent mask — very-low-freq gradient fBm, smooth plate shape.
        continent = fbm(
            X, Y,
            max(3, self.octaves - 2),
            self.persistence * 1.1,
            self.lacunarity,
            self.scale * continent_freq_factor,
            self.seed + 9999
        )
        # Fixed-range normalisation: maps [-0.7071, 0.7071] → [0, 1].
        # A sub-region with the same seed/scale/coords gives the same values
        # as the corresponding pixels in the full world.
        continent = (continent + _GN_MAX) / (2.0 * _GN_MAX)
        continent = continent.clip(0.0, 1.0)

        # 2. Mountain ridges — ridged multifractal, masked by continent.
        if ridge_anisotropy > 1.0:
            mountains = anisotropic_ridged_fbm(
                X, Y,
                self.octaves,
                self.persistence,
                self.lacunarity,
                self.scale,
                self.seed,
                angle=ridge_angle,
                anisotropy=ridge_anisotropy
            )
        else:
            mountains = ridged_fbm(
                X, Y,
                self.octaves,
                self.persistence,
                self.lacunarity,
                self.scale,
                self.seed
            )
        # ridged_fbm already returns [0, 1]; mask by continent² for sharp shores.
        mountains *= continent ** 2

        # 3. Fine-scale detail — higher-freq gradient fBm.
        detail = fbm(
            X, Y,
            self.octaves,
            self.persistence,
            self.lacunarity,
            self.scale * 2.5,
            self.seed + 7777
        )
        # detail in [-0.7071, 0.7071]; keep centred at 0 for additive blending.

        # Combine.  Theoretical range:
        #   [- detail_weight * _GN_MAX,
        #    continent_weight + mountain_weight + detail_weight * _GN_MAX]
        terrain = (continent_weight * continent
                   + mountain_weight * mountains
                   + detail_weight   * detail)

        # Normalise to [0, 1] using the same theoretical bounds every time.
        t_lo = -detail_weight * _GN_MAX
        t_hi = continent_weight + mountain_weight + detail_weight * _GN_MAX
        terrain = (terrain - t_lo) / (t_hi - t_lo + 1e-8)
        terrain = terrain.clip(0.0, 1.0)

        # Legacy slope-based damping (disabled by default; destroys ridges).
        if slope_reduction_alpha > 0.0:
            terrain = self.slope_based_height_reduction(
                terrain, lim=lim, alpha=slope_reduction_alpha)

        before = terrain.copy()

        # ----------------------------------------------------------
        # EROSION  (geological ordering)
        # ----------------------------------------------------------

        # STEP 1 — Thermal erosion (talus / scree).
        if thermal_iters > 0:
            talus_px = talus * pixel_dx
            terrain = thermal_erosion(
                terrain,
                thermal_iters,
                talus_px,
                thermal_strength
            )

        # STEP 2 — Legacy grid hydraulic erosion (usually replaced by particles).
        if hydraulic_iters > 0:
            terrain = hydraulic_erosion(
                terrain,
                hydraulic_iters,
                rain,
                evap,
                erode,
                deposit
            )

        # STEP 3 — Particle hydraulic erosion (channel incision, fine detail).
        if particle_droplets > 0:
            max_path_px  = max(1, int(particle_max_path / pixel_dx))
            min_slope_px = particle_min_slope * pixel_dx

            ref_pixels   = 512 * 512
            area_ratio   = (width * height) / ref_pixels
            scaled_drops = max(1, int(particle_droplets * area_ratio))

            terrain = particle_hydraulic_erosion(
                terrain,
                num_droplets=scaled_drops,
                max_path=max_path_px,
                inertia=particle_inertia,
                capacity_factor=particle_capacity,
                min_slope=min_slope_px,
                erode_speed=particle_erode_speed,
                deposit_speed=particle_deposit_speed,
                evaporate_speed=particle_evaporate,
                gravity=particle_gravity,
                seed=self.seed,
            )

        # STEPS 4-6 — Drainage network, river carving, valley widening,
        #             alluvial deposition.  All share the same flow-accum map.
        if flow_threshold > 0:
            norm_t = self.normalize(terrain)
            accum  = flow_accumulation(norm_t)

            # 4. V-valley incision (stream-power law).
            norm_t = carve_rivers(
                norm_t, accum,
                threshold   = flow_threshold,
                max_depth   = river_depth,
                slope_power = river_slope_power,
            )

            # 5. Valley widening — lateral erosion / floodplains.
            if valley_sigma > 0.0:
                norm_t = widen_valleys(
                    norm_t, accum,
                    threshold    = flow_threshold,
                    valley_sigma = valley_sigma,
                    max_depth    = valley_depth,
                )

            # 6. Alluvial deposition — fans, deltas, valley fill.
            if alluvial_amount > 0.0:
                norm_t = alluvial_deposition(
                    norm_t, accum,
                    threshold      = flow_threshold,
                    deposit_amount = alluvial_amount,
                    lowland_cutoff = alluvial_lowland_cutoff,
                )

            # Re-blend into original height range.
            mn, mx  = terrain.min(), terrain.max()
            terrain = norm_t * (mx - mn) + mn

        # Gaussian smoothing — use sparingly (≤0.5 sigma); blurs erosion detail.
        if gaussian_sigma > 0:
            terrain = gaussian_filter(terrain, sigma=gaussian_sigma)

        return terrain

    def generate_climate(self, terrain, lim=(0, 1, 0, 1),
                         wind_angle_deg=270.0, rain_shadow_passes=40,
                         rain_shadow_lift=2.5):
        """
        Derive a simple climate layer from a finished terrain heightmap.

        Returns a dict with:
          'rainfall'    : float32 [0,1] — 1 = wet windward coast, 0 = dry shadow.
          'temperature' : float32 [0,1] — 1 = hot lowlands, 0 = cold peaks.

        Parameters
        ----------
        wind_angle_deg : prevailing wind FROM direction (degrees CW from North).
        rain_shadow_passes / rain_shadow_lift : see helper.rain_shadow().
        """
        norm = self.normalize(terrain)

        rainfall = rain_shadow(
            norm,
            wind_angle_deg  = wind_angle_deg,
            num_passes      = rain_shadow_passes,
            lift_factor     = rain_shadow_lift,
        )

        # Simple lapse-rate temperature: cool with altitude, using fixed coefficients
        # so values are consistent between world-scale and sub-patch calls.
        temperature = (1.0 - norm).astype(np.float32)

        return {'rainfall': rainfall, 'temperature': temperature}

    def create_sea(self, terrain, percentile=0.3):
        sea_level = np.percentile(terrain, percentile * 100)
        sea_mask = terrain < sea_level
        return sea_mask

    def normalize(self, terrain):
        mn, mx = terrain.min(), terrain.max()
        return (terrain - mn) / (mx - mn + 1e-8)

    def plot_slope_angle_histogram(self, terrain, lim=(0, 1, 0, 1), world_size_m=100_000, max_altitude=1000):
        """
        lim          : coordinate limits used to generate this terrain patch.
        world_size_m : physical size (metres) of the full lim=(0,1) domain.
        max_altitude : physical altitude range (metres) represented by [0, 1].
        Cell size is derived automatically so this is correct at every LOD level.
        """
        xmin, xmax, ymin, ymax = lim
        h, w = terrain.shape
        cell_x = (xmax - xmin) * world_size_m / w
        cell_y = (ymax - ymin) * world_size_m / h

        terrain_m = terrain * max_altitude
        grad_y, grad_x = np.gradient(terrain_m, cell_y, cell_x)
        slope = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2)) * (180.0 / np.pi)

        plt.hist(slope.flatten(), bins=50, color='tan', edgecolor='black')
        plt.title('Slope Angle Distribution')
        plt.xlabel('Slope Angle (degrees)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def plot(self, terrain):
        fig, ax = plt.subplots(figsize=(13, 13))
        plt.imshow(terrain, cmap='terrain', vmin=0, vmax=1)
        plt.colorbar()
        plt.show()

    def plot3D(self, terrain, zlim=(-1, 2)):
        h, w = terrain.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        Z = terrain

        fig = plt.figure(figsize=(13, 13))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(zlim[0], zlim[1])
        cbar = ax.plot_surface(X, Y, Z, cmap='terrain', edgecolor='none', vmin=0, vmax=1)
        fig.colorbar(cbar, ax=ax, shrink=0.5, aspect=5)
        plt.show()
