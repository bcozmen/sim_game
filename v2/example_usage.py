"""
example_usage.py
================
Demonstrates the full Terrain pipeline:
  1. Generate a 100 km × 100 km world at 1 000 × 1 000 resolution.
  2. Zoom into a 10 km × 10 km sub-region at the SAME grid resolution.
  3. Show that raw terrain values are consistent between world and sub-region
     (no re-normalisation needed — the sub-patch values fall inside the world range).
  4. Generate and plot a climate layer (rainfall / temperature).

Run from the repo root:
    python -m v2.example_usage
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from v2.map.LOD import Terrain

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

terrain_gen = Terrain(
    scale       = 6.0,
    octaves     = 7,
    persistence = 0.50,
    lacunarity  = 2.0,
    seed        = 42,
)

# Generation parameters shared between world and sub-region calls.
# Because continent / mountain / detail layers now use fixed-range
# normalisation, the same coord always maps to the same base height —
# sub-region values will automatically fall within the world value range.
GEN_PARAMS = dict(
    warp_strength        = 0.35,
    thermal_iters        = 30,
    talus                = 0.025,
    thermal_strength     = 0.45,
    particle_droplets    = 120_000,
    particle_max_path    = 0.25,
    particle_inertia     = 0.40,
    particle_capacity    = 8.0,
    particle_erode_speed = 0.35,
    particle_deposit_speed = 0.30,
    particle_evaporate   = 0.025,
    particle_gravity     = 10.0,
    # River system
    flow_threshold       = 400,
    river_depth          = 0.06,
    river_slope_power    = 0.45,
    valley_sigma         = 2.5,    # lateral valley widening
    valley_depth         = 0.025,
    alluvial_amount      = 0.02,   # sediment fans / delta fill
    # Layer weights
    continent_weight     = 0.45,
    mountain_weight      = 0.45,
    detail_weight        = 0.10,
    continent_freq_factor= 0.15,
    # Tectonic ridge direction — slight NW-SE orientation like the Andes
    ridge_angle          = 0.4,    # radians (~23°)
    ridge_anisotropy     = 2.5,    # chain elongation factor
    # Keep gaussian off to preserve erosion detail
    gaussian_sigma       = 0.0,
    slope_reduction_alpha= 0.0,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2.  WORLD  (lim [0,1]×[0,1] = 100 km × 100 km)
# ──────────────────────────────────────────────────────────────────────────────
WORLD_LIM  = (0.0, 1.0, 0.0, 1.0)
WORLD_SIZE = 1000

print("Generating world (1000×1000)…")
st = time.time()
world = terrain_gen.generate(
    lim    = WORLD_LIM,
    width  = WORLD_SIZE,
    height = WORLD_SIZE,
    **GEN_PARAMS,
)

world_min = float(world.min())
world_max = float(world.max())
time_taken = time.time() - st

print(f"World raw range: [{world_min:.4f}, {world_max:.4f}]")
print(f"Time taken: {time_taken:.2f} seconds")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  SUB-REGION  (lim [0,0.1]×[0,0.1] = 10 km × 10 km)
#
# With fixed-range normalisation the sub-patch is generated completely
# independently and its raw values sit inside the world range — no
# post-hoc rescaling is required.
# ──────────────────────────────────────────────────────────────────────────────
SUB_FRAC = 0.1
SUB_LIM  = (0.0, SUB_FRAC, 0.0, SUB_FRAC)
SUB_SIZE = 1000

print("\nGenerating sub-region (1000×1000 @ 10 km)…")
st = time.time()
sub = terrain_gen.generate(
    lim    = SUB_LIM,
    width  = SUB_SIZE,
    height = SUB_SIZE,
    **GEN_PARAMS,
)
time_taken = time.time() - st
print(f"Time taken: {time_taken:.2f} seconds")

sub_min = float(sub.min())
sub_max = float(sub.max())
print(f"Sub-region raw range: [{sub_min:.4f}, {sub_max:.4f}]")
print(f"World range:          [{world_min:.4f}, {world_max:.4f}]")
consistent = sub_min >= world_min - 0.05 and sub_max <= world_max + 0.05
print(f"Height-scale consistent: {consistent}")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  CLIMATE LAYER
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerating world climate…")
st = time.time()
climate = terrain_gen.generate_climate(
    world,
    lim             = WORLD_LIM,
    wind_angle_deg  = 270.0,   # westerlies
    rain_shadow_passes = 50,
    rain_shadow_lift   = 2.5,
)
time_taken = time.time() - st
print(f"Time taken: {time_taken:.2f} seconds")
rainfall    = climate['rainfall']
temperature = climate['temperature']

# ──────────────────────────────────────────────────────────────────────────────
# 5.  PLOTS
# ──────────────────────────────────────────────────────────────────────────────

# Normalise world to [0,1] for display.
world_disp = (world - world_min) / (world_max - world_min + 1e-8)

# --- World 2-D ---
fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(world_disp, cmap='terrain', vmin=0, vmax=1, origin='upper')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Normalised elevation')
sub_px = int(WORLD_SIZE * SUB_FRAC)
ax.add_patch(plt.Rectangle(
    (0, 0), sub_px, sub_px,
    linewidth=2, edgecolor='red', facecolor='none', label='10 km sub-region'
))
ax.legend(loc='upper right')
ax.set_title('World — 100 km × 100 km  (1 000 × 1 000)')
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
plt.tight_layout()
plt.show()

# --- World 3-D (downsampled) ---
ds  = 4
W3  = world_disp[::ds, ::ds]
h3, w3 = W3.shape
Xg, Yg = np.meshgrid(np.arange(w3), np.arange(h3))
fig3 = plt.figure(figsize=(12, 10))
ax3  = fig3.add_subplot(111, projection='3d')
ax3.set_zlim(0, 1)
surf = ax3.plot_surface(Xg, Yg, W3, cmap='terrain',
                        edgecolor='none', vmin=0, vmax=1, rcount=250, ccount=250)
fig3.colorbar(surf, ax=ax3, shrink=0.45, aspect=10, label='Normalised elevation')
ax3.set_title('World 3-D — 100 km × 100 km')
plt.tight_layout()
plt.show()

# --- Sub-region 2-D displayed in world-consistent scale ---
# Raw values are now height-consistent; reuse the world min/max directly.
fig2, ax2 = plt.subplots(figsize=(9, 9))
sub_disp = (sub - world_min) / (world_max - world_min + 1e-8)
im2 = ax2.imshow(sub_disp, cmap='terrain', vmin=0, vmax=1, origin='upper')
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.02)
cbar2.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar2.set_ticklabels([f'{world_min + t*(world_max-world_min):.3f}'
                      for t in [0.0, 0.25, 0.5, 0.75, 1.0]])
cbar2.set_label('World-scale elevation')
ax2.set_title(
    f'Sub-region — 10 km × 10 km  (1 000 × 1 000,  10× zoom)\n'
    f'Raw range [{sub_min:.3f}, {sub_max:.3f}] inside world range '
    f'[{world_min:.3f}, {world_max:.3f}]'
)
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
plt.tight_layout()
plt.show()

# --- Sub-region 3-D ---
ds2 = 4
S3  = sub_disp[::ds2, ::ds2]
hs3, ws3 = S3.shape
Xs, Ys = np.meshgrid(np.arange(ws3), np.arange(hs3))
fig4 = plt.figure(figsize=(12, 10))
ax4  = fig4.add_subplot(111, projection='3d')
ax4.set_zlim(0, 1)
surf2 = ax4.plot_surface(Xs, Ys, S3, cmap='terrain',
                         edgecolor='none', vmin=0, vmax=1, rcount=250, ccount=250)
cbar4 = fig4.colorbar(surf2, ax=ax4, shrink=0.45, aspect=10)
cbar4.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cbar4.set_ticklabels([f'{world_min + t*(world_max-world_min):.3f}'
                      for t in [0.0, 0.25, 0.5, 0.75, 1.0]])
cbar4.set_label('World-scale elevation')
ax4.set_title(f'Sub-region 3-D — 10 km × 10 km  '
              f'(raw [{sub_min:.3f}, {sub_max:.3f}])')
plt.tight_layout()
plt.show()

# --- Climate maps ---
fig5, axes = plt.subplots(1, 2, figsize=(16, 7))

im_r = axes[0].imshow(rainfall, cmap='Blues', vmin=0, vmax=1, origin='upper')
plt.colorbar(im_r, ax=axes[0], fraction=0.03, pad=0.02, label='Precipitation [0–1]')
axes[0].set_title('Rainfall / orographic precipitation\n(westerlies from 270°)')
axes[0].set_xlabel('X (pixels)')
axes[0].set_ylabel('Y (pixels)')

im_t = axes[1].imshow(temperature, cmap='RdYlBu_r', vmin=0, vmax=1, origin='upper')
plt.colorbar(im_t, ax=axes[1], fraction=0.03, pad=0.02,
             label='Temperature [0=cold peak, 1=warm lowland]')
axes[1].set_title('Temperature  (lapse-rate proxy)')
axes[1].set_xlabel('X (pixels)')

plt.suptitle('World Climate Layer', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
