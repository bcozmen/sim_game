import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_biomes(terrain, rainfall, temperature):
    """
    Parameterised biome classifier.

    Returns a uint8 mask with biome indices. Default thresholds are tuned for
    a simple large-scale world, but can be overridden by passing `thresholds`.

    New biome indices:
      0: ocean/sea
      1: desert
      2: grassland
      3: forest
      4: mountains
      5: snow
      6: wetlands
      7: tundra
      8: savanna

    Inputs expected normalised to [0,1].
    """
    return compute_biomes_with_params(terrain, rainfall, temperature)


def compute_biomes_with_params(terrain, rainfall, temperature, *, thresholds=None):
    h = np.asarray(terrain)
    r = np.asarray(rainfall)
    t = np.asarray(temperature)

    # default thresholds
    defaults = {
        'sea_percentile': 30,
        'mountain_h': 0.75,
        'snow_h': 0.9,
        'snow_temp': 0.25,
        'desert_r': 0.18,
        'forest_r': 0.55,
        'wetland_r': 0.45,
        'wetland_h_max': 0.35,
        'tundra_h_min': 0.65,
        'tundra_temp_max': 0.35,
        'savanna_r_min': 0.25,
        'savanna_r_max': 0.45,
    }
    if thresholds is None:
        thresholds = defaults
    else:
        # fill missing
        for k, v in defaults.items():
            thresholds.setdefault(k, v)

    biome = np.zeros_like(h, dtype=np.uint8)

    sea = h < np.percentile(h, thresholds['sea_percentile'])
    biome[sea] = 0

    # Mountains and snow (highest override)
    mountain = h > thresholds['mountain_h']
    snow = (h > thresholds['snow_h']) | ((h > thresholds['mountain_h']) & (t < thresholds['snow_temp']))
    biome[mountain] = 4
    biome[snow] = 5

    # Wetlands: relatively low elevation, moderate-high rainfall
    wetland = (r >= thresholds['wetland_r']) & (h <= thresholds['wetland_h_max']) & (~sea)
    biome[wetland] = 6

    # Tundra: cold and relatively high, but below permanent snow
    tundra = (h >= thresholds['tundra_h_min']) & (t <= thresholds['tundra_temp_max']) & (~snow)
    biome[tundra] = 7

    # Desert: arid lowlands
    desert = (r < thresholds['desert_r']) & (~sea) & (~mountain) & (~wetland)
    biome[desert] = 1

    # Forest: wet lowlands / midlands
    forest = (r >= thresholds['forest_r']) & (~sea) & (~mountain) & (~wetland)
    biome[forest] = 3

    # Savanna: intermediate rainfall with warm temperatures
    savanna = (r >= thresholds['savanna_r_min']) & (r < thresholds['savanna_r_max']) & (t > 0.45) & (~sea) & (~mountain)
    biome[savanna] = 8

    # Grassland: all remaining non-sea/mountain cells
    others = (~sea) & (~mountain) & (~desert) & (~forest) & (~snow) & (~wetland) & (~tundra) & (~savanna)
    biome[others] = 2

    return biome


def biome_colors():
    # RGB tuples for each biome index 0..5
    return np.array([
        [0.0, 0.16, 0.6],   # ocean
        [0.87, 0.79, 0.65], # desert (sand)
        [0.7, 0.85, 0.55],  # grassland
        [0.0, 0.5, 0.0],    # forest
        [0.5, 0.5, 0.5],    # mountains (rock)
        [1.0, 1.0, 1.0],    # snow
        [0.45, 0.65, 0.45], # wetlands (marsh green)
        [0.8, 0.85, 0.9],   # tundra (pale)
        [0.86, 0.75, 0.4],  # savanna (dry grass)
    ])


def plot_map(terrain, sea_mask=None, biomes=None, title=None, figsize=(8,8)):
    """Plot 2D map with optional sea overlay and biome overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(terrain, cmap='terrain', vmin=-0.2, vmax=1, origin='upper')
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Normalised elevation')

    legend_items = []
    legend_labels = []
    if biomes is not None:
        # overlay biome colors with alpha
        colors = biome_colors()
        rgba = np.zeros((biomes.shape[0], biomes.shape[1], 4), dtype=np.float32)
        for i in range(colors.shape[0]):
            mask = biomes == i
            if not np.any(mask):
                continue
            rgba[mask, :3] = colors[i]
            rgba[mask, 3] = 0.28
            legend_items.append(colors[i])
        ax.imshow(rgba, origin='upper', vmin=-0.2, vmax=1)
        # build legend labels mapping index->name
        biome_names = {
            0: 'Ocean', 1: 'Desert', 2: 'Grassland', 3: 'Forest', 4: 'Mountains', 5: 'Snow',
            6: 'Wetlands', 7: 'Tundra', 8: 'Savanna'
        }
        # create legend handles
        from matplotlib.patches import Patch
        legend_handles = []
        for i, col in enumerate(biome_colors()):
            # only include biomes present
            if np.any(biomes == i):
                legend_handles.append(Patch(facecolor=col, edgecolor='k', label=biome_names.get(i, str(i))))
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', title='Biomes')

    if sea_mask is not None:
        sea_overlay = np.zeros((sea_mask.shape[0], sea_mask.shape[1], 4), dtype=np.float32)
        sea_overlay[sea_mask, :3] = np.array([0.0, 0.16, 0.6])
        sea_overlay[sea_mask, 3] = 0.8
        ax.imshow(sea_overlay, origin='upper')

    if title:
        ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.tight_layout()
    plt.show()


def plot_3d_map(terrain, sea_mask=None, biomes=None, title=None, zlim=(0,1), downsample=4, figsize=(12,10)):
    h, w = terrain.shape
    ds = max(1, downsample)
    Z = terrain
    Zs = Z[::ds, ::ds]
    hs, ws = Zs.shape
    Xg, Yg = np.meshgrid(np.arange(ws), np.arange(hs))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(zlim[0], zlim[1])

    # color by biome if available else terrain colormap
    if biomes is not None:
        colors_rgb = biome_colors()
        Bds = biomes[::ds, ::ds]
        facecolors = colors_rgb[Bds.flatten()].reshape(hs, ws, 3)
    else:
        cmap = plt.get_cmap('terrain')
        facecolors = cmap(Zs)

    if sea_mask is not None:
        sea_ds = sea_mask[::ds, ::ds]
        sea_overlay = np.zeros((hs, ws, 4), dtype=np.float32)
        sea_overlay[sea_ds, :3] = np.array([0.0, 0.16, 0.6])
        sea_overlay[sea_ds, 3] = 0.75
        facecolors = np.where(sea_ds[..., None], sea_overlay, facecolors)

    surf = ax.plot_surface(Xg, Yg, Zs, facecolors=facecolors, rcount=200, ccount=200, linewidth=0, antialiased=True, vmin=-0.2, vmax=1)
    ax.view_init(elev=35, azim=-45)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_slope_histogram(terrain, lim = (0,1,0,1), total_size = 100000, max_height = 1000):
    """
    Compute and plot a physically-correct slope histogram.

    This computes dz/dx and dz/dy in metres/metre using the provided
    `total_size` (metres for the full [0,1] domain) and `lim` window,
    then converts the gradient magnitude to degrees.

    Returns the slope map in degrees (2D array) for further inspection.
    """
    H, W = terrain.shape

    # physical span of the lim window in metres (assume total_size is the
    # size of the full [0,1] domain in metres). Support non-square lim.
    x_span = total_size * (lim[1] - lim[0])
    y_span = total_size * (lim[3] - lim[2])

    # metres per pixel
    dx = x_span / max(W, 1)
    dy = y_span / max(H, 1)

    # compute gradients of height in metres
    terrain_m = terrain * max_height
    dz_dy, dz_dx = np.gradient(terrain_m, dy, dx)  # order: (row_spacing, col_spacing)

    # slope magnitude and conversion to degrees
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad)

    plt.figure(figsize=(8, 4))
    plt.hist(slope_deg.flatten(), bins=100, color='tan', edgecolor='black')
    plt.title('Slope distribution (degrees)')
    plt.xlabel('Slope (degrees)')
    plt.ylabel('Density')
    plt.grid()
    plt.show()

    # Print useful percentiles so the user can see whether extremes are outliers
    for p in (50, 90, 95, 99, 100):
        print(f"{p}th percentile slope: {np.percentile(slope_deg, p):.2f}°")

    return slope_deg