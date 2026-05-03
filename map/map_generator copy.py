import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter
import pickle

from .gpu_helpers import binary_dilation, trace_river_dijkstra, timer, binary_erosion

class MapGenerator:
    def __init__(self, size, scale = 1.0 , roughness = 0.45,
                 river_count = 8, cell_size = 10, max_altitude = 1000):
        self.size = size
        self.scale = scale 
        self.roughness = roughness 
        self.river_count = river_count
        self.cell_size = cell_size
        self.max_altitude = max_altitude
    def generate(self):
        height_map = self.generate_height_map()
        #cut height map 2nd dimention to 3/4 to make it more rectangular
        sea_mask, height_map = self.generate_sea_mask(height_map)
        river_mask = self.generate_rivers(height_map, sea_mask)
        river_mask = binary_dilation(river_mask, iterations=2)  # dilate rivers to make them wider
        fertility_map = self.generate_fertility_map(height_map, sea_mask, river_mask)
        forest_map = self.generate_forest_map(height_map, fertility_map, sea_mask, river_mask)
        humidity_map = self.generate_humidity_map(height_map, sea_mask, river_mask)
        slope_map = self.generate_slope_map(height_map)

        info = np.array([self.cell_size, self.max_altitude, self.sea_level], dtype=np.float32)
        maps = (height_map, sea_mask, river_mask, fertility_map, forest_map, humidity_map, slope_map, info)
        keys = ["height", "sea", "river", "fertility", "forest", "humidity", "slope", "info"]
        return keys, maps
    @timer
    def generate_height_map(self):
        assert (self.size - 1) & (self.size - 2) == 0, "size must be 2^n + 1"

        grid = np.zeros((self.size, self.size), dtype=np.float32)
        grid[0, 0] = np.random.rand() * self.scale
        grid[0, -1] = np.random.rand() * self.scale
        grid[-1, 0] = np.random.rand() * self.scale
        grid[-1, -1] = np.random.rand() * self.scale
        step = self.size - 1
        current = self.scale

        while step > 1:
            half = step // 2

            # square step – vectorized
            xs = np.arange(half, self.size, step)
            ys = np.arange(half, self.size, step)
            xv, yv = np.meshgrid(xs, ys, indexing='ij')
            grid[xv, yv] = (
                grid[xv - half, yv - half] +
                grid[xv - half, yv + half] +
                grid[xv + half, yv - half] +
                grid[xv + half, yv + half]
            ) / 4.0 + (np.random.rand(*xv.shape) - 0.5) * current

            # diamond step – vectorized
            # diamond points lie on the half-grid where (i_idx + j_idx) is odd
            coords = np.arange(0, self.size, half)
            ii, jj = np.meshgrid(np.arange(len(coords)), np.arange(len(coords)), indexing='ij')
            diamond_mask = (ii + jj) % 2 == 1
            dx = coords[ii[diamond_mask]]
            dy = coords[jj[diamond_mask]]

            vals   = np.zeros(len(dx), dtype=np.float32)
            counts = np.zeros(len(dx), dtype=np.float32)
            for ddx, ddy in ((-half, 0), (half, 0), (0, -half), (0, half)):
                nx, ny = dx + ddx, dy + ddy
                valid = (nx >= 0) & (nx < self.size) & (ny >= 0) & (ny < self.size)
                vals[valid]   += grid[nx[valid], ny[valid]]
                counts[valid] += 1
            grid[dx, dy] = vals / counts + (np.random.rand(len(dx)) - 0.5) * current

            step //= 2
            current *= self.roughness

        grid = gaussian_filter(grid, sigma=2.0 )
        grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-5)  # normalize to [0, 1]
        return grid

    def generate_humidity_map(self, height_map, sea_mask, river_mask):
        #humidity is higher near water and in low altitudes
        humidity = np.zeros_like(height_map, dtype=np.float32)

        #proximity to water (sea or river)
        water_mask = sea_mask | river_mask
        water_distance = distance_transform_edt(~water_mask)
        
        #normalize distance to water and invert so closer to water is higher humidity
        humidity = 1 - (water_distance / np.max(water_distance))
        humidity[water_mask] = 0  # water cells have max humidity
        return humidity
    
    def generate_sea_mask(self, height_map):
        #calculate sea level as the 30th percentile of height values
        sea_level = np.percentile(height_map, 30)
        sea_mask = height_map < sea_level
        

        #erase small land patches in the sea and small sea patches in the land by dilation and erosion
        sea_mask = binary_dilation(sea_mask, iterations=2)
        sea_mask = binary_erosion(sea_mask, iterations=2)

        height_map = height_map - sea_level  # adjust height map so sea level is at 0
        height_map[sea_mask] = 0  # set sea cells to exactly 0 height
        self.sea_level = 0  # store for later use in fertility and forest maps
        return sea_mask, height_map

    def _get_river_sources(self, height_map, land_mask):
        # get the top 15% highest land points and randomly pick river_count of them
        land_heights = height_map[land_mask]
        threshold = np.percentile(land_heights, 85)

        # np.where returns index arrays directly — faster than argwhere (no stacking)
        xs, ys = np.where((height_map >= threshold) & land_mask)

        # randomly sample without shuffling the whole array
        n = len(xs)
        k = min(self.river_count, n)
        chosen = np.random.choice(n, size=k, replace=False)
        sources = list(zip(xs[chosen].tolist(), ys[chosen].tolist()))
        return sources

    def _dilate_rivers(self, river_count):
        h, w = river_count.shape
        max_count = int(river_count.max())
        dilated = np.zeros((h, w), dtype=bool)
        for level in range(1, max_count + 1):
            # Every cell carrying at least `level` rivers contributes a dilation of `level` iterations
            mask = river_count >= level
            dilated |= binary_dilation(mask, iterations=level)
        return dilated

    def _dilate_all_rivers(self, river_mask):
        dilated = binary_dilation(river_mask, iterations=2)  # fixed dilation for all rivers
        return dilated

    def generate_slope_map(self, height_map):
        dz, dx = np.gradient(height_map)
        slope = np.sqrt(dz**2 + dx**2)
        #slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-5)  # normalize to [0, 1]
        return slope

    def generate_fertility_map(self, height_map, sea_mask, river_mask):
        river_effect = np.zeros_like(height_map, dtype=np.float32)

        #Give a boost to fertility based on proximity to rivers
        
        river_distance = distance_transform_edt(~river_mask)
        river_effect += np.exp(-river_distance / self.size)  # decay with distance

        #normalize fertility to [0, 1]
        river_effect = (river_effect - river_effect.min()) / (river_effect.max() - river_effect.min() + 1e-5)

        # Apply altitude effect: boost fertility at low to mid altitudes, reduce at high altitudes
        altitude = np.clip(height_map - self.sea_level, 0, 1)
        altitude_effect = (1 - altitude) ** 3  # quadratic boost for low altitudes, drops off at high altitudes
        river_effect *= altitude_effect

        #normalize again after altitude effect
        river_effect[sea_mask | river_mask] = 0  # sea and river have zero fertility
        river_effect = (river_effect - river_effect.min()) / (river_effect.max() - river_effect.min() + 1e-5)

        fertility_map = river_effect
        return fertility_map

    def generate_forest_map(self, height_map, fertility_map, sea_mask, river_mask):
        height = np.clip(height_map - self.sea_level, 0, 1)
        altitude_effect = np.exp(height)  # exponential boost for higher altitudes
        normalized_altitude_effect = (altitude_effect - altitude_effect.min()) / (altitude_effect.max() - altitude_effect.min() + 1e-5)

        forest_map = normalized_altitude_effect + normalized_altitude_effect * fertility_map
        forest_map = (forest_map - forest_map.min()) / (forest_map.max() - forest_map.min() + 1e-5)
        return forest_map
    @timer
    def generate_rivers(self, height_map, sea_mask):
        sources = self._get_river_sources(height_map, ~sea_mask)

        river_count = np.zeros_like(height_map, dtype=np.int32)
        hf = height_map.astype(np.float32)


        for s in sources:
            result = self._trace_river(s[0], s[1], height_map, sea_mask, river_count)
            if result is not None:
                river_count = result

        river_mask = self._dilate_rivers(river_count)
        return river_mask
        
    def _trace_river(self, start_x, start_y, height_map, sea_mask, river_count):
        h, w = height_map.shape

        end_x, end_y, parent = trace_river_dijkstra(
            np.int32(start_x), np.int32(start_y),
            height_map.astype(np.float32),
            sea_mask,
        )

        if end_x == -1:
            return None  # no path to sea found

        # reconstruct path using the flat parent array
        start_flat = start_x * w + start_y
        cur_flat = int(end_x) * w + int(end_y)
        path = []
        while cur_flat != start_flat:
            path.append((cur_flat // w, cur_flat % w))
            cur_flat = int(parent[cur_flat])
        path.append((start_x, start_y))

        # build river count map (increment for each river passing through)
        for x, y in path:
            river_count[x, y] += 1

        return river_count
    