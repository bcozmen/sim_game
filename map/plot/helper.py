import numpy as np
from ..functions.gpu import binary_erosion

def get_overlay(shape, color = (1.0, 1.0, 0.0)):
    H, W = shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA overlay
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    return overlay

def mask_overlay(overlay, mask, color = (1.0, 1.0, 0.0), alpha = 1.0):
    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = alpha
    return overlay

def darken_border(overlay, mask, old_color=(0.0, 0.0, 0.0), alpha=1.0):
    border_mask = mask & ~binary_erosion(mask, iterations=1)
    #darken color by 50%
    darkened_color = tuple([c * 0.5 for c in old_color])
    overlay = mask_overlay(overlay, border_mask, color=darkened_color, alpha=alpha)
    return overlay

def plot_rural_farmland_overlay(city, overlay, color ):
    mask = (city[:, :, 1] == 0) & (city[:, :, 2] == 0) & (city[:, :, 0] != 0)
    overlay = mask_overlay(overlay, mask, color=color[:3], alpha=color[3])   # Yellow for rural farmland
    return darken_border(overlay, mask, old_color =color[:3])  # Darker yellow for borders

def plot_rural_forest_overlay(city, overlay, color):
    mask = (city[:, :, 1] == 0) & (city[:, :, 2] == 1) & (city[:, :, 0] != 0)
    overlay = mask_overlay(overlay, mask, color=color[:3], alpha=color[3])   # Dark Green for rural forest
    return darken_border(overlay, mask, old_color=color[:3])  # Darker green for borders

def plot_urban_overlay(city, overlay, color):
    mask = (city[:, :, 1] == 1) & (city[:, :, 0] != 0)
    overlay = mask_overlay(overlay, mask, color=color[:3], alpha=color[3])   # Red for urban areas
    return darken_border(overlay, mask, old_color=color[:3])  # Darker red for borders

def plot_city_edges_overlay(city, overlay, color):
    edge_mask = city[:, :, 3] == 1
    overlay = mask_overlay(overlay, edge_mask, color=color[:3], alpha=color[3])   # Grey for city edges
    return overlay
def get_sea_and_river_mask(maps):
        sea = maps["sea"].cpu().numpy()
        river = maps["river"].cpu().numpy()
        H, W = sea.shape

        # sea mask
        sea_mask = np.logical_or(sea, river)
        return sea_mask