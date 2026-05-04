import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, gaussian_filter, label
import pickle

from .gpu_helpers import binary_dilation, timer, binary_erosion, trace_river_fill

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
        sea_mask, height_map = self.generate_sea_mask(height_map)

        island_map = self.generate_island_map(sea_mask)
        height_map, sea_mask = self.delete_islands(height_map, sea_mask)
        sea_mask = self.delete_sea_lakes(sea_mask)


        river_mask = self.generate_rivers(height_map, sea_mask)
        slope_map = self.generate_slope_map(height_map)


        humidity_map = self.generate_humidity_map(height_map, sea_mask, river_mask)

        fertility_map = self.generate_fertility_map(height_map, humidity_map, slope_map)
        forest_map = self.generate_forest_map(height_map, fertility_map, sea_mask, river_mask)
        husbandry_map = self.generate_husbandry_map(height_map, humidity_map)

        habitability_map = self.generate_habitability_map(slope_map)
        city_map = self.generate_city_map(height_map)
        
        

        info = np.array([self.cell_size, self.max_altitude, self.sea_level], dtype=np.float32)
        maps = (height_map, sea_mask, river_mask, fertility_map, forest_map, humidity_map, slope_map, husbandry_map, habitability_map, island_map, city_map, info)
        keys = ["height", "sea", "river", "fertility", "forest", "humidity", "slope", "husbandry", "habitability", "island", "city", "info"]
        return keys, maps

    def generate_city_map(self, height_map):
        W,H = height_map.shape
        #city id, owner id, building type
        city_map = np.zeros((W, H, 3), dtype=np.int32)  # 0 means no city, positive integers are city IDs
        return city_map
    def delete_islands(self, height_map, sea_mask):
        #keep only the n largest islands and convert the rest to sea
        labeled, num_features = label(~sea_mask)
        for i in range(1, num_features + 1):
            mask = labeled == i
            if np.sum(mask) < 1000:  # if the island is smaller than 500 cells, delete it
                sea_mask[mask] = True
                height_map[mask] = 0  # set height to 0 for deleted islands
        return height_map, sea_mask

    def delete_sea_lakes(self, sea_mask):
        #delete small sea lakes that are completely surrounded by land
        labeled, num_features = label(sea_mask)
        for i in range(1, num_features + 1):
            mask = labeled == i
            if (np.sum(mask) < 100):  # if the lake is smaller than 100 cells, delete it
                sea_mask[mask] = False
        return sea_mask

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

    def generate_sea_mask(self, height_map):
        #calculate sea level as the 30th percentile of height values
        sea_level = np.percentile(height_map, 30)
        sea_mask = height_map < sea_level
        

        #erase small land patches in the sea and small sea patches in the land by dilation and erosion
        sea_mask = binary_dilation(sea_mask, iterations=4)
        sea_mask = binary_erosion(sea_mask, iterations=4)

        height_map = height_map - sea_level  # adjust height map so sea level is at 0
        height_map[sea_mask] = 0  # set sea cells to exactly 0 height
        self.sea_level = 0  # store for later use in fertility and forest maps
        return sea_mask, height_map
    
    
        
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
        river_mask = binary_dilation(river_mask, iterations=2)  # make rivers wider for better visibility and fertility effect
        return river_mask
        
    def _trace_river(self, start_x, start_y, height_map, sea_mask, river_count):
        h, w = height_map.shape

        path = trace_river_fill(
            np.int32(start_x), np.int32(start_y),
            height_map.astype(np.float32),
            sea_mask,
        )

        
        for x, y in path:
            river_count[x, y] += 1

        return river_count

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

    def generate_slope_map(self, height_map):
        dz, dx = np.gradient(height_map)
        slope = np.sqrt(dz**2 + dx**2)
        return slope

    def generate_humidity_map(self, height_map, sea_mask, river_mask):
        #humidity is higher near water and in low altitudes
        humidity = np.zeros_like(height_map, dtype=np.float32)

        #proximity to water (sea or river)
        water_mask = sea_mask | river_mask
        water_distance = distance_transform_edt(~water_mask)
        
        #normalize distance to water and invert so closer to water is higher humidity
        humidity = 1 - (water_distance / np.max(water_distance))
        humidity[water_mask] = 0  # water cells have max humidity
        humidity = humidity ** 0.4
        return humidity

    def generate_fertility_map(self, height_map, humidity_map, slope_map):
        slope = slope_map
        slope[slope < 0.0005] = 0 
        slope = 1 / np.exp(slope * 100)  # steep slopes have much lower fertility, gentle slopes are close to 1
        fertility = (humidity_map) * (1 - (height_map ** 1.5)) * slope  # combine humidity, altitude, and slope effects 
        fertility = (fertility - fertility.min()) / (fertility.max() - fertility.min() + 1e-5)  # normalize to [0, 1]
        return fertility

    def generate_forest_map(self, height_map, fertility_map, sea_mask, river_mask):
        height = np.clip(height_map - self.sea_level, 0, 1)
        altitude_effect = np.exp(height)  # exponential boost for higher altitudes
        normalized_altitude_effect = (altitude_effect - altitude_effect.min()) / (altitude_effect.max() - altitude_effect.min() + 1e-5)

        forest_map = normalized_altitude_effect * (1 + fertility_map)
        forest_map = (forest_map - forest_map.min()) / (forest_map.max() - forest_map.min() + 1e-5)
        return forest_map

    def generate_husbandry_map(self, height_map, humidity_map):
        husbandry = (humidity_map) * ((1 - height_map ) ** 0.25)  # husbandry is better in humid and low altitude areas
        husbandry = (husbandry - husbandry.min()) / (husbandry.max() - husbandry.min() + 1e-5)
        return husbandry
    
    def generate_habitability_map(self, slope_map):
        slope = slope_map
        slope[slope < 0.001] = 0 
        habitability = 1 / np.exp(slope * 1000)  # steep slopes are much less habitable, gentle slopes are close to 1
        habitability = (habitability - habitability.min()) / (habitability.max() - habitability.min() + 1e-5)  # normalize to [0, 1]
        return habitability

    def generate_island_map(self, sea_mask):
        #islands are land cells that are completely surrounded by sea
        labeled, num_features = label(~sea_mask)
        
        return labeled

    