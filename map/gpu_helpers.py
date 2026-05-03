import math
import time
import heapq
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit

# Set to False to disable per-function timing output
timer_flag = True


def timer(func):
    """Decorator: prints wall-clock time of *func* when timer_flag is True."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if timer_flag:
            t0     = time.perf_counter()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {time.perf_counter() - t0:.4f} s")
            return result
        return func(*args, **kwargs)
    return wrapper


def compute_on_gpu(func):
    """
    Decorator: moves a numpy array argument to the best available device,
    runs *func*, then returns the result as numpy.
    Supports (H, W) and (C, H, W) inputs.
    """
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x      = torch.from_numpy(x).to(device)
        result = func(x, *args, **kwargs)
        if isinstance(result, torch.Tensor):
            return result.cpu().numpy()
        tensor, *rest = result
        return (tensor.cpu().numpy(), *rest)
    return wrapper


def masked_avg(array, mask, radius):
    mask = mask.to(array.device, dtype=array.dtype)

    # Ensure 4D: (N, C, H, W)
    if array.dim() == 2:
        array = array.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif array.dim() == 3:
        array = array.unsqueeze(1)
        mask = mask.unsqueeze(1)

    k = 2 * radius + 1

    kernel = torch.ones((1, 1, k, k), device=array.device, dtype=array.dtype)

    masked_array = array * mask

    numerator = F.conv2d(masked_array, kernel, padding=radius)
    denominator = F.conv2d(mask, kernel, padding=radius)

    result = numerator / denominator.clamp(min=1e-8)
    result[denominator == 0] = 0

    return result.squeeze()




@compute_on_gpu
def binary_dilation(x, k=3, iterations=1):
    """
    GPU binary dilation using max pooling.

    Args:
        x: numpy array (2D or 3D)
        k: kernel size (odd)
        iterations: number of dilation steps

    Returns:
        numpy array (same 2D shape as input)
    """

   
    # ensure shape (1,1,H,W)
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)

    x = x.float()

    for _ in range(iterations):
        x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    x = (x > 0)

    return x.squeeze()

@njit(cache=True)
def trace_river_dijkstra(start_x, start_y, height_map, sea_mask):
    """
    Numba-compiled Dijkstra core for river tracing.
    Returns (end_x, end_y, parent) where parent is a flat int32 array
    encoding the predecessor of each cell as (row*w + col), or -1 if unvisited.
    Returns end_x == -1 when no path to sea was found.
    """
    h, w = height_map.shape
    cost = np.full((h, w), np.inf, dtype=np.float32)
    cost[start_x, start_y] = np.float32(0.0)

    # flat parent array: parent[x*w+y] = predecessor flat index, -1 = none
    parent = np.full(h * w, -1, dtype=np.int32)

    # priority queue as list of (cost, x, y)
    heap = [(np.float32(0.0), np.int32(start_x), np.int32(start_y))]

    end_x = np.int32(-1)
    end_y = np.int32(-1)

    while len(heap) > 0:
        c, x, y = heapq.heappop(heap)

        if sea_mask[x, y]:
            end_x = x
            end_y = y
            break

        if c > cost[x, y]:
            continue

        current_h = height_map[x, y]

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= h or ny < 0 or ny >= w:
                    continue
                nh = height_map[nx, ny]
                step_cost = nh - current_h
                if step_cost < np.float32(0.0):
                    step_cost = np.float32(0.0)
                new_cost = c + step_cost
                if new_cost < cost[nx, ny]:
                    cost[nx, ny] = new_cost
                    parent[nx * w + ny] = x * w + y
                    heapq.heappush(heap, (new_cost, np.int32(nx), np.int32(ny)))

    return end_x, end_y, parent





@njit(cache=True)
def _fast_sigmoid(x, scale=10.0):
    """Altitude penalty multiplier: maps height diff to a cost scale in [0.5, 2.0]."""
    v = 1.0 + scale * x
    if v < 0.5:
        return 0.5
    if v > 2.0:
        return 2.0
    return v


@njit(cache=True, nogil=True)
def _astar_core(sy, sx, gy, gx, H, W, cost_map, height_map, use_height, cell_size, max_altitude):
    """
    Numba-compiled A* core.
    Returns (came_from_y, came_from_x, found) where came_from_* are flat int32 arrays.
    """
    g_score = np.full((H, W), np.inf, dtype=np.float32)
    came_from_y = np.full(H * W, -1, dtype=np.int32)
    came_from_x = np.full(H * W, -1, dtype=np.int32)
    visited = np.zeros((H, W), dtype=np.bool_)

    g_score[sy, sx] = np.float32(0.0)

    # heap entries: (f, g, y, x)
    heap = [(np.float32(math.sqrt((sy - gy) ** 2 + (sx - gx) ** 2)), np.float32(0.0), np.int32(sy), np.int32(sx))]

    dy_arr = np.array([-1, 1, 0, 0], dtype=np.int32)
    dx_arr = np.array([0, 0, -1, 1], dtype=np.int32)

    while len(heap) > 0:
        f, base_g, cy, cx = heapq.heappop(heap)

        if visited[cy, cx]:
            continue
        visited[cy, cx] = True

        if cy == gy and cx == gx:
            return came_from_y, came_from_x, True

        for k in range(4):
            ny = cy + dy_arr[k]
            nx = cx + dx_arr[k]

            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue

            step_cost = cost_map[ny, nx]
            if math.isinf(step_cost):
                continue

            if use_height:
                height_diff = (height_map[ny, nx] - height_map[cy, cx]) * max_altitude / cell_size
                
                #scale from [0 0.1] to [1 10]
                step_cost = step_cost * (1 + 10000 * max(height_diff, 0))


            new_g = base_g + step_cost
            if new_g < g_score[ny, nx]:
                g_score[ny, nx] = new_g
                came_from_y[ny * W + nx] = cy
                came_from_x[ny * W + nx] = cx
                h = math.sqrt((ny - gy) ** 2 + (nx - gx) ** 2)
                heapq.heappush(heap, (np.float32(new_g + h), np.float32(new_g), np.int32(ny), np.int32(nx)))

    return came_from_y, came_from_x, False


def astar(start, goal, cost_map, height_map=None, cell_size=1.0, max_altitude=1.0):
    H, W = cost_map.shape
    sy, sx = int(start[0]), int(start[1])
    gy, gx = int(goal[0]), int(goal[1])

    cost_map = cost_map.astype(np.float32)
    use_height = height_map is not None
    if not use_height:
        height_map = np.zeros((H, W), dtype=np.float32)
    else:
        height_map = height_map.astype(np.float32)

    came_from_y, came_from_x, found = _astar_core(
        sy, sx, gy, gx, H, W,
        cost_map, height_map, use_height,
        np.float32(cell_size), np.float32(max_altitude)
    )

    if not found:
        return None, np.inf

    # Reconstruct path
    path = []
    y, x = gy, gx
    while y != -1:
        path.append((int(y), int(x)))
        py = came_from_y[y * W + x]
        px = came_from_x[y * W + x]
        y, x = int(py), int(px)
    path.reverse()
    return path, float(cost_map[gy, gx])