from .gpu import binary_dilation

type_to_cmap = {
    "height": "terrain",
    "fertility": "YlGn",
    "forest": "Greens",
    "humidity": "PuBuGn",
    "husbandry": "Oranges",
    "habitability": "coolwarm"
}


def plot_map(self, maps, ax, map_type = "height", title = None):
    this_map = maps[map_type].cpu().numpy()

    cmap = type_to_cmap.get(map_type, "viridis")
    vmin, vmax = 0, 1
    if map_type == "height":
        vmin, vmax = -0.2, 1

    #add colorbar per axis
    ax.imshow(this_map, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(ax.imshow(this_map, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax)
    ax.set_title(title if title else f"{map_type.capitalize()} Map")

    overlay = _get_overlay(this_map.shape, color=(0.0, 0.0, 0.0))
    overlay = plot_sea_and_river_overlay(maps, overlay)
    overlay = plot_cities(maps, overlay)
    for city in self.cities:
        ax.scatter(city.pos[1], city.pos[0], c='white',s=25, edgecolors = 'black')

    overlay = plot_roads(maps, overlay)
    ax.imshow(overlay)

def plot_sea_and_river_overlay(maps, overlay):
    sea = maps["sea"].cpu().numpy()
    river = maps["river"].cpu().numpy()
    H, W = sea.shape

    # sea mask
    sea_mask = np.logical_or(sea, river)
    overlay = _mask_overlay(overlay, sea_mask, color=(0.0, 0.5, 1.0), alpha=0.9)  # Semi-transparent blue for sea and river
    return overlay

def plot_cities(maps, overlay):
    H, W = maps["height"].shape
    city = maps["city"].cpu().numpy()
    overlay = _plot_rural_farmland_overlay(city, overlay)
    overlay = _plot_rural_forest_overlay(city, overlay)
    overlay = _plot_urban_overlay(city, overlay)
    return overlay
def _plot_rural_farmland_overlay(city, overlay):
    mask = (city[:, :, 1] == 0) & (city[:, :, 2] == 0) & (city[:, :, 0] != 0)
    overlay = _mask_overlay(overlay, mask, color=(1.0, 1.0, 0.0), alpha=0.7)   # Yellow for rural farmland
    return _darken_border(overlay, mask, border_color=(0.5, 0.5, 0.0))  # Darker yellow for borders
def _plot_rural_forest_overlay(city, overlay):
    mask = (city[:, :, 1] == 0) & (city[:, :, 2] == 1) & (city[:, :, 0] != 0)
    overlay = _mask_overlay(overlay, mask, color=(0.0, 0.5, 0.0), alpha=0.7)   # Dark Green for rural forest
    return _darken_border(overlay, mask, border_color=(0.0, 0.25, 0.0))  # Darker green for borders
def _plot_urban_overlay(city, overlay):
    mask = (city[:, :, 1] == 1) & (city[:, :, 0] != 0)
    overlay = _mask_overlay(overlay, mask, color=(1.0, 0.0, 0.0), alpha=0.7)   # Red for urban areas
    return _darken_border(overlay, mask, border_color=(0.5, 0.0, 0.0))  # Darker red for borders

def plot_roads(maps, overlay):
    road_mask = maps["road"].cpu().numpy()
    H, W = road_mask.shape
    overlay = _mask_overlay(overlay, road_mask, color=(0, 0, 0), alpha=1.0)  # Grey color for roads
    return overlay

