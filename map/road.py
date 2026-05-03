import numpy as np
from scipy.spatial import Delaunay
from concurrent.futures import ThreadPoolExecutor, as_completed

from .gpu_helpers import timer, astar

# ── Cost-map constants ────────────────────────────────────────────────────────
BASE_COST        = 10.0
RIVER_COST       = 20.0
MOUNTAIN_THRESH  = 0.75
MOUNTAIN_SLOPE   = 100.0
ROAD_COST_FACTOR = 0.1   # existing-road discount applied during iterative build
EXTRA_EDGE_RATIO = 0.4   # fraction of non-MST Delaunay edges added for loops


class Road:
    def __init__(self, maps, cities):
        self.maps     = maps
        self.cities   = cities
        self.road_map = maps["road"]

        # Cache numpy views reused across many A* calls
        self._height_np             = maps["height"].cpu().numpy()
        cell_size, max_altitude, *_ = maps["info"].cpu().numpy()
        self._cell_size             = float(cell_size)
        self._max_altitude          = float(max_altitude)

        self._init_roads()

    # ── Public entry point ────────────────────────────────────────────────────
    def _init_roads(self):
        cost_map   = self._build_cost_map()
        candidates = self._delaunay_candidates()
        self._build_network(candidates, cost_map)

    # ── Cost map ──────────────────────────────────────────────────────────────

    def _build_cost_map(self) -> np.ndarray:
        sea_mask   = self.maps["sea"].cpu().numpy()
        river_mask = self.maps["river"].cpu().numpy()
        height_map = self._height_np

        cost_map             = np.full_like(height_map, BASE_COST, dtype=np.float32)
        cost_map[sea_mask]   = np.inf
        cost_map[river_mask] = RIVER_COST

        high = height_map > MOUNTAIN_THRESH
        cost_map[high] = RIVER_COST + MOUNTAIN_SLOPE * (height_map[high] - MOUNTAIN_THRESH)

        return cost_map

    # ── Graph construction ────────────────────────────────────────────────────

    def _delaunay_candidates(self) -> list[tuple[int, int]]:
        """Return (i, j) city-index pairs from a Delaunay triangulation (O(n log n))."""
        positions = np.array([c.pos for c in self.cities], dtype=float)
        n = len(positions)
        if n < 2:
            return []
        if n == 2:
            return [(0, 1)]

        edges: set[tuple[int, int]] = set()
        for simplex in Delaunay(positions).simplices:
            for k in range(3):
                i, j = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(i, j), max(i, j)))
        return list(edges)

    # ── A* helpers ───────────────────────────────────────────────────────────

    def _compute_bounds(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        margin: int = 50,
    ) -> tuple[int, int, int, int]:
        """Compute crop bounds that tightly cover start→goal + margin."""
        H, W  = self._height_np.shape
        sy, sx = start
        gy, gx = goal
        return (
            max(0, min(sy, gy) - margin),
            min(H, max(sy, gy) + margin),
            max(0, min(sx, gx) - margin),
            min(W, max(sx, gx) + margin),
        )

    def _run_astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        cost_map: np.ndarray,
    ) -> tuple[list | None, float, tuple[int, int]]:
        """Run A* in a cropped window. Returns (path, dist, (min_y, min_x))."""
        min_y, max_y, min_x, max_x = self._compute_bounds(start, goal)

        local_cost   = cost_map[min_y:max_y, min_x:max_x]
        local_height = self._height_np[min_y:max_y, min_x:max_x]
        local_start  = (start[0] - min_y, start[1] - min_x)
        local_goal   = (goal[0]  - min_y, goal[1]  - min_x)

        path, dist = astar(
            local_start, local_goal,
            local_cost, local_height,
            self._cell_size, self._max_altitude,
        )
        return path, dist, (min_y, min_x)

    # ── Kruskal's MST ─────────────────────────────────────────────────────────

    @staticmethod
    def _kruskal_mst(
        n: int,
        edge_costs: dict[tuple[int, int], float],
        extra_edge_ratio: float,
    ) -> list[tuple[int, int]]:
        """
        Return MST edges (Kruskal) + cheapest non-MST extras for loop roads.
        Uses path-halving union-find.
        """
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True

        mst_edges:  list[tuple[int, int]] = []
        extra_pool: list[tuple[int, int]] = []
        for edge in sorted(edge_costs, key=edge_costs.__getitem__):
            (mst_edges if union(*edge) else extra_pool).append(edge)

        n_extra = max(1, int(len(extra_pool) * extra_edge_ratio))
        return mst_edges + extra_pool[:n_extra]

    # ── Path stamping ─────────────────────────────────────────────────────────

    def _stamp_path(
        self,
        path: list[tuple[int, int]],
        offset: tuple[int, int],
        cost_map: np.ndarray,
    ) -> None:
        """Mark road cells on road_map and apply cost discount in cost_map."""
        min_y, min_x = offset
        pts = np.array(path, dtype=np.int32)
        ys  = pts[:, 0] + min_y
        xs  = pts[:, 1] + min_x
        self.road_map[ys, xs]  = True
        cost_map[ys, xs]      *= ROAD_COST_FACTOR

    def _path_touches_roads(
        self,
        path: list[tuple[int, int]],
        offset: tuple[int, int],
    ) -> bool:
        """Return True if any cell in *path* already has a road stamped."""
        min_y, min_x = offset
        pts = np.array(path, dtype=np.int32)
        return bool(self.road_map[pts[:, 0] + min_y, pts[:, 1] + min_x].any())

    # ── Main network builder ──────────────────────────────────────────────────

    def _build_network(
        self,
        candidates: list[tuple[int, int]],
        cost_map: np.ndarray,
        extra_edge_ratio: float = EXTRA_EDGE_RATIO,
    ) -> list:
        """
        Two-phase terrain-aware road network construction.

        PLAN (frozen snapshot, parallelised)
          1. A* every Delaunay candidate concurrently on the frozen cost_map.
          2. Kruskal MST on those weights → optimal backbone order.
          3. Append cheapest non-MST edges for loops.

        BUILD (live cost_map, mutated in-place)
          4. For each selected edge, reuse the frozen A* path if no existing road
             intersects it (live result would be identical). Otherwise re-run A*
             on the live map so the new road merges onto shared corridors.
          5. Stamp each path immediately so the next edge can benefit.
        """
        frozen = cost_map.copy()

        # ── Phase 1: parallel A* on frozen snapshot ───────────────────────────
        # _astar_core has nogil=True so threads truly run in parallel.
        frozen_results: dict[tuple[int, int], tuple] = {}

        def _astar_edge(edge):
            i, j = edge
            return edge, self._run_astar(self.cities[i].pos, self.cities[j].pos, frozen)

        with ThreadPoolExecutor() as pool:
            for edge, result in pool.map(_astar_edge, candidates):
                _, dist, _ = result
                if not np.isinf(dist):
                    frozen_results[edge] = result   # (path, dist, offset)

        if not frozen_results:
            return []

        edge_costs = {e: r[1] for e, r in frozen_results.items()}
        selected   = self._kruskal_mst(len(self.cities), edge_costs, extra_edge_ratio)

        # ── Phase 2: build iteratively on the live cost_map ───────────────────
        built_paths = []
        for i, j in selected:
            frozen_path, _, frozen_offset = frozen_results[(i, j)]

            # Reuse frozen path when no prior road intersects it — the live A*
            # would return the same result, so we skip the redundant call.
            if self._path_touches_roads(frozen_path, frozen_offset):
                path, _, offset = self._run_astar(
                    self.cities[i].pos, self.cities[j].pos, cost_map
                )
            else:
                path, offset = frozen_path, frozen_offset

            if path is not None:
                self._stamp_path(path, offset, cost_map)
                built_paths.append((path, offset))

        return built_paths

