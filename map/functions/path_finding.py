import numpy as np
from numba import njit
from .heap import MinHeap


def _points_to_array(points, W):
    if isinstance(points, np.ndarray) and points.dtype == bool:
        yx = np.argwhere(points).astype(np.int32)  # (N, 2)
    else:
        yx = np.array(points, dtype=np.int32).reshape(-1, 2)

    flat = (yx[:, 0] * W + yx[:, 1]).reshape(-1, 1)
    return np.concatenate([yx, flat], axis=1)  # (N, 3)


def reconstruct_path(parent, goal_flat):
    path = []
    cur = int(goal_flat)
    while cur != -1:
        path.append(cur)
        cur = int(parent[cur])
    path.reverse()
    return path


def dijkstra(cost_map, starts, goals=None, max_amount=None, max_cost=None):
    """
    Multi-source / multi-goal Dijkstra.

    Returns
    -------
    parent : int64 (H*W,)
    dist   : float64 (H*W,)   # full map, INF = unreachable
    """
    W = cost_map.shape[1]
    starts_arr = _points_to_array(starts, W)
    goals_arr = _points_to_array(goals, W) if goals is not None else np.empty((0, 3), dtype=np.int32)
    
    parent, dist = _pathfind(cost_map, starts_arr, goals_arr, False, max_amount = max_amount, max_cost = max_cost)
    dist[dist >= 1e18] = np.inf  # convert INF to np.inf for better readability

    is_single_source = starts_arr.shape[0] == 1
    is_single_goal = goals_arr.shape[0] == 1

    path = None
    
    if is_single_source and goals_arr.shape[0] > 0:
        cheapest_goal_idx = np.argmin(dist[goals_arr[:, 2]])
        cheapest_goal_flat = goals_arr[cheapest_goal_idx, 2]
        if dist[cheapest_goal_flat] < np.inf:
            path = reconstruct_path(parent, cheapest_goal_flat)
            path = np.array(path, dtype=np.int32)
            path_y = path // W
            path_x = path % W
            path = np.stack([path_y, path_x], axis=1)  # (N, 2)

    
    dist = dist.reshape(cost_map.shape[0], cost_map.shape[1])  # (H, W)
    return parent, dist, path


def astar(cost_map, starts, goals):
    """
    Multi-source / multi-goal A*.

    Returns
    -------
    parent : int64 (H*W,)
    dist   : float64 (H*W,)
    """
    W = cost_map.shape[1]
    starts_arr = _points_to_array(starts, W)
    goals_arr = _points_to_array(goals, W)
    parent, dist = _pathfind(cost_map, starts_arr, goals_arr, True)
    dist[dist >= 1e18] = np.inf  # convert INF to np.inf for better readability
    #find the path (its always single source single goal)
    path = None
    goal_flat = goals_arr[0, 2]
    if dist[goal_flat] < np.inf:
        path = reconstruct_path(parent, goal_flat)
        path = np.array(path, dtype=np.int32)
        path_y = path // W
        path_x = path % W
        path = np.stack([path_y, path_x], axis=1)  # (N, 2)
    dist = dist.reshape(cost_map.shape[0], cost_map.shape[1])  # (H, W)
    return parent, dist, path



# ---------------------------------------------------------------------------
# Numba core
# ---------------------------------------------------------------------------

@njit
def _pathfind(cost_map, starts, goals, use_astar, max_amount=None, max_cost=None):
    # cost_map: (H, W, 4) with [north, south, west, east] costs
    H = cost_map.shape[0]
    W = cost_map.shape[1]
    N = H * W
    INF = 1e18

    g = np.full(N, INF)
    parent = np.full(N, -1, dtype=np.int64)
    heap = MinHeap(N)

    visited = 0

    # Build O(1) goal lookup to avoid O(goals) scan on every pop
    is_goal = np.zeros(N, dtype=np.bool_)
    for i in range(goals.shape[0]):
        is_goal[goals[i, 2]] = True

    # ---- init sources ----
    for i in range(starts.shape[0]):
        s = starts[i, 2]
        g[s] = 0.0
        f = _h(s, goals, W) if use_astar else 0.0
        heap.push(s, f)

    # ---- main loop ----
    while heap.size > 0:
        if max_amount is not None and visited >= max_amount:
            break

        u, fu = heap.pop()

        if max_cost is not None and g[u] > max_cost:
            break
        if goals.shape[0] > 0:
            if is_goal[u]:
                return parent, g
        
        visited += 1
        f_u = g[u] + (_h(u, goals, W) if use_astar else 0.0)
        if fu > f_u:
            continue

        uy = u // W
        ux = u % W

        # north
        if uy > 0:
            v = u - W
            w = cost_map[uy - 1, ux, 1]
            if not np.isinf(w):
                alt = g[u] + w
                if alt < g[v]:
                    g[v] = alt
                    parent[v] = u
                    fv = alt + (_h(v, goals, W) if use_astar else 0.0)
                    heap.push(v, fv)

        # south
        if uy < H - 1:
            v = u + W
            w = cost_map[uy + 1, ux, 0]
            if not np.isinf(w):
                alt = g[u] + w
                if alt < g[v]:
                    g[v] = alt
                    parent[v] = u
                    fv = alt + (_h(v, goals, W) if use_astar else 0.0)
                    heap.push(v, fv)

        # west
        if ux > 0:
            v = u - 1
            w = cost_map[uy, ux - 1, 3]
            if not np.isinf(w):
                alt = g[u] + w
                if alt < g[v]:
                    g[v] = alt
                    parent[v] = u
                    fv = alt + (_h(v, goals, W) if use_astar else 0.0)
                    heap.push(v, fv)

        # east
        if ux < W - 1:
            v = u + 1
            w = cost_map[uy, ux + 1, 2]
            if not np.isinf(w):
                alt = g[u] + w
                if alt < g[v]:
                    g[v] = alt
                    parent[v] = u
                    fv = alt + (_h(v, goals, W) if use_astar else 0.0)
                    heap.push(v, fv)

    return parent, g


@njit
def _h(node, goals, W):
    if goals.shape[0] == 0:
        return 0.0

    y = node // W
    x = node % W

    best = 1e18
    for i in range(goals.shape[0]):
        dy = abs(y - goals[i, 0])
        dx = abs(x - goals[i, 1])
        d = dy + dx
        if d < best:
            best = d

    return float(best)