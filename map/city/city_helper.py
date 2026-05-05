import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label



def choose_from_pdf(pdf, top_k=100):
    flat = pdf.flatten()
    valid_indices = np.where(flat != 0)[0]

    if len(valid_indices) == 0:
        raise ValueError("PDF has no valid entries to sample from.")

    top_k = min(top_k, len(valid_indices))
    top_indices = valid_indices[np.argsort(flat[valid_indices])[-top_k:]]
    top_probs = flat[top_indices]
    top_probs /= top_probs.sum()  # Normalize

    chosen_flat_index = np.random.choice(top_indices, p=top_probs)
    return np.unravel_index(chosen_flat_index, pdf.shape)

def remove_small_islands(mask, min_size):
    mask = mask.astype(bool)

    # Explicit 4-connectivity
    structure = np.array([[0,1,0],
                          [1,1,1],
                          [0,1,0]])

    labeled, _ = label(mask, structure=structure)

    counts = np.bincount(labeled.ravel())

    remove = counts < min_size
    remove[0] = False

    cleaned = ~remove[labeled]
    cleaned[labeled == 0] = False

    return cleaned

def get_box_indices(array, radius):
    #get a box around true values in the array with a margin of radius
    indices = np.argwhere(array)
    if len(indices) == 0:
        return array, (0, array.shape[0], 0, array.shape[1])
    min_y, min_x = indices.min(axis=0)
    max_y, max_x = indices.max(axis=0)
    min_y = max(0, min_y - radius)
    max_y = min(array.shape[0], max_y + radius)
    min_x = max(0, min_x - radius)
    max_x = min(array.shape[1], max_x + radius)
    return (min_y, max_y, min_x, max_x)

def cut_box(array, indices):
    min_y, max_y, min_x, max_x = indices
    return array[min_y:max_y, min_x:max_x]

#local index is numpy array of shape (N, 2)
def convert_local_to_global(local_index, cut_indices):
    if (type(local_index) == tuple):
        local_index = np.array(local_index)
    min_y, max_y, min_x, max_x = cut_indices
    global_index = local_index + np.array([min_y, min_x])
    return global_index

def plot_fn(array, title = ""):
    plt.imshow(array)
    plt.title(title)
    plt.colorbar()
    plt.show()


def soft_max(array, beta = 1.0):
    coef = np.exp(beta * array)
    coef_sum = np.sum(coef)
    return coef / (coef_sum + 1e-8)
def choose_indices(array, amount, minimize=False):
    flat = array.flatten()

    # Get indices of valid (non-inf) entries
    valid_indices = np.where(flat != np.inf)[0]

    # Sort only valid values
    sorted_valid = valid_indices[np.argsort(flat[valid_indices])]

    # Select desired amount
    if minimize:
        chosen_indices = sorted_valid[:amount]
    else:
        chosen_indices = sorted_valid[-amount:]

    # Convert back to coordinates
    coords = np.unravel_index(chosen_indices, array.shape)
    return np.column_stack(coords)
def sample_index_from_pdf(pdf, percentile = 0.5):
    # zero out values below the percentile
    threshold = np.percentile(pdf, percentile * 100)
    pdf = np.where(pdf >= threshold, pdf, 0)
    flat = pdf.flatten()
    prob = flat / flat.sum()  # normalize

    chosen_flat_index = np.random.choice(len(flat), p=prob)
    return np.unravel_index(chosen_flat_index, pdf.shape)

def normalize_inverted(arr, inf_threshold=1e18):
    """Normalize arr to [0,1] and invert, zeroing out inf values."""
    inf_mask = arr >= inf_threshold
    arr[inf_mask] = 0
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    arr = 1 - arr
    arr[inf_mask] = 0
    return arr

def circle_mask(shape, center, radius):
    """Return a boolean mask of a filled circle."""
    y, x = center
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2

def build_city_exclusion_mask(city_map, radius, factor=4):
    """Dilate each existing city position by max_radius * factor."""
    mask = city_map > 0
    for y, x in np.argwhere(mask):
        mask |= circle_mask(city_map.shape, (y, x), radius * factor)
    return mask

def get_box_indices_smart(mask, growth_mask, radius):
    H, W = mask.shape

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, H, 0, W)

    min_y = max(0, ys.min() - radius)
    max_y = min(H, ys.max() + radius)
    min_x = max(0, xs.min() - radius)
    max_x = min(W, xs.max() + radius)

    sub = growth_mask[min_y:max_y, min_x:max_x]

    # row/col projections (VERY fast, C-optimized)
    row_any = sub.any(axis=1)
    col_any = sub.any(axis=0)

    # find bounds in O(n)
    rows = np.where(row_any)[0]
    cols = np.where(col_any)[0]

    if len(rows) == 0 or len(cols) == 0:
        return (0, H, 0, W)

    min_y += rows[0]
    max_y = min_y + rows[-1] + 1

    min_x += cols[0]
    max_x = min_x + cols[-1] + 1

    return (min_y, max_y, min_x, max_x)

