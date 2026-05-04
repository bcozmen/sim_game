import heapq
import numpy as np
from numba import njit
@njit
def grow_region(score, blocked_mask, chosen_index, amount = 1):
    h, w = score.shape
    visited = np.zeros(score.shape, dtype=np.bool_)
    heap = [(-score[chosen_index], chosen_index)]  # max heap based on score
    count = 0

    while heap and count < amount:
        neg_score, index = heapq.heappop(heap)
        if visited[index]:
            continue
        visited[index] = True
        count += 1

        # Add neighbors to the heap
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny = index[0] + dy
                nx = index[1] + dx
                if (0 <= ny < h and 0 <= nx < w and
                    not visited[ny, nx] and blocked_mask[ny, nx] > 0):
                    heapq.heappush(heap, (-score[ny, nx], (ny, nx)))
    # Convert visited boolean array to (count, 2) array of indices
    visited_indices = np.argwhere(visited)
    return visited_indices

def _normalize(array):
        return (array - array.min()) / (array.max() - array.min() + 1e-5)

def get_bounding_box_indices(array):
    #given a 2d boolean array, find the bounding box of the True values in the array and return the indices of the box
    rows = np.any(array, axis=1)
    cols = np.any(array, axis=0)
    if not np.any(rows) or not np.any(cols):
        return (0, array.shape[0], 0, array.shape[1])
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    return (min_y, max_y + 1, min_x, max_x + 1)



def cut_box(array, margin = 2):
    #find the bounding box of the True values in the array and cut a box around it with the given margin, return the cut array and the indices of the cut
    indices = np.argwhere(array)
    if len(indices) == 0:
        return array, (0, array.shape[0], 0, array.shape[1])
    min_y, min_x = indices.min(axis=0)
    max_y, max_x = indices.max(axis=0)
    min_y = max(0, min_y - margin)
    max_y = min(array.shape[0], max_y + margin)
    min_x = max(0, min_x - margin)
    max_x = min(array.shape[1], max_x + margin)
    return array[min_y:max_y, min_x:max_x], (min_y, max_y, min_x, max_x)

def convert_local_to_global(local_index, cut_indices):
    min_y, max_y, min_x, max_x = cut_indices
    global_index = (local_index[0] + min_y, local_index[1] + min_x)
    return global_index
def convert_local_to_global_vectorized(local_indices, cut_indices):
    min_y, max_y, min_x, max_x = cut_indices
    global_indices = local_indices + np.array([min_y, min_x])
    return global_indices

def cut_array(array, indices):
    min_y, max_y, min_x, max_x = indices
    return array[min_y:max_y, min_x:max_x]


import numpy as np
from numba import njit

@njit
def compute_distance_numba(city_urban_mask, impassable_mask, max_radius = -1):
    h, w = city_urban_mask.shape
    size = h * w

    dist = np.full((h, w), np.inf, dtype=np.float32)

    qy = np.empty(size, dtype=np.int32)
    qx = np.empty(size, dtype=np.int32)

    head = 0
    tail = 0

    # Initialize sources
    for y in range(h):
        for x in range(w):
            if city_urban_mask[y, x]:
                dist[y, x] = 0.0
                qy[tail] = y
                qx[tail] = x
                tail += 1

    # Convert optional radius to a usable value
    use_cutoff = max_radius >= 0  # pass -1 to disable cutoff

    while head < tail:
        y = qy[head]
        x = qx[head]
        head += 1

        d = dist[y, x]

        if use_cutoff and d >= max_radius:
            continue

        nd = d + 1.0

        # up
        if y > 0:
            if not impassable_mask[y-1, x] and nd < dist[y-1, x]:
                dist[y-1, x] = nd
                qy[tail] = y-1
                qx[tail] = x
                tail += 1

        # down
        if y < h-1:
            if not impassable_mask[y+1, x] and nd < dist[y+1, x]:
                dist[y+1, x] = nd
                qy[tail] = y+1
                qx[tail] = x
                tail += 1

        # left
        if x > 0:
            if not impassable_mask[y, x-1] and nd < dist[y, x-1]:
                dist[y, x-1] = nd
                qy[tail] = y
                qx[tail] = x-1
                tail += 1

        # right
        if x < w-1:
            if not impassable_mask[y, x+1] and nd < dist[y, x+1]:
                dist[y, x+1] = nd
                qy[tail] = y
                qx[tail] = x+1
                tail += 1

    return dist


import numpy as np
from numba import njit

SQRT2 = np.float32(1.41421356237)

# ---------------------------------------------------------------------------
# Minimal binary min-heap for (priority, y, x) triples, numba-compatible
# ---------------------------------------------------------------------------
@njit
def _heap_push(heap_p, heap_y, heap_x, size, p, y, x):
    heap_p[size] = p
    heap_y[size] = y
    heap_x[size] = x
    i = size
    size += 1
    while i > 0:
        parent = (i - 1) >> 1
        if heap_p[parent] > heap_p[i]:
            heap_p[parent], heap_p[i] = heap_p[i], heap_p[parent]
            heap_y[parent], heap_y[i] = heap_y[i], heap_y[parent]
            heap_x[parent], heap_x[i] = heap_x[i], heap_x[parent]
            i = parent
        else:
            break
    return size

@njit
def _heap_pop(heap_p, heap_y, heap_x, size):
    top_p = heap_p[0]
    top_y = heap_y[0]
    top_x = heap_x[0]
    size -= 1
    heap_p[0] = heap_p[size]
    heap_y[0] = heap_y[size]
    heap_x[0] = heap_x[size]
    i = 0
    while True:
        left  = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left  < size and heap_p[left]  < heap_p[smallest]:
            smallest = left
        if right < size and heap_p[right] < heap_p[smallest]:
            smallest = right
        if smallest == i:
            break
        heap_p[i], heap_p[smallest] = heap_p[smallest], heap_p[i]
        heap_y[i], heap_y[smallest] = heap_y[smallest], heap_y[i]
        heap_x[i], heap_x[smallest] = heap_x[smallest], heap_x[i]
        i = smallest
    return top_p, top_y, top_x, size


@njit
def terrain_distance(height, source_mask, impassable, road_map, river_map, scale=10):
    h, w = height.shape
    INF = np.float32(1e20)
    dist   = np.full((h, w), INF,   dtype=np.float32)
    parent = np.full((h, w), -1,    dtype=np.int32)
    visited = np.zeros((h, w),      dtype=np.bool_)

    dy   = np.array([-1,  1,  0,  0, -1, -1,  1,  1], dtype=np.int16)
    dx   = np.array([ 0,  0, -1,  1, -1,  1, -1,  1], dtype=np.int16)
    base = np.array([ 1,  1,  1,  1, SQRT2, SQRT2, SQRT2, SQRT2], dtype=np.float32)

    max_heap = h * w * 8 + h * w
    heap_p = np.empty(max_heap, dtype=np.float32)
    heap_y = np.empty(max_heap, dtype=np.int32)
    heap_x = np.empty(max_heap, dtype=np.int32)
    hsize  = 0

    # Seed ALL source cells with dist=0
    for y in range(h):
        for x in range(w):
            if source_mask[y, x]:
                dist[y, x] = np.float32(0.0)
                hsize = _heap_push(heap_p, heap_y, heap_x, hsize,
                                   np.float32(0.0), y, x)

    while hsize > 0:
        d0, y, x, hsize = _heap_pop(heap_p, heap_y, heap_x, hsize)

        if visited[y, x]:
            continue
        visited[y, x] = True

        h0 = height[y, x]

        for k in range(8):
            ny = y + dy[k]
            nx = x + dx[k]

            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if impassable[ny, nx] and not river_map[ny, nx]:
                continue
            if visited[ny, nx]:
                continue

            dh   = float(height[ny, nx]) - float(h0)
            cost = base[k] + max(np.float32(0.0), np.float32(dh)) * np.float32(scale)

            if road_map[ny, nx]:
                cost *= np.float32(0.5)
            if river_map[ny, nx]:
                cost *= np.float32(5.0)

            nd = d0 + cost
            if nd < dist[ny, nx]:
                dist[ny, nx]   = nd
                parent[ny, nx] = y * w + x
                hsize = _heap_push(heap_p, heap_y, heap_x, hsize, nd, ny, nx)

    return dist, parent