import matplotlib.pyplot as plt
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .functions.gpu import binary_dilation
from .functions.helper import timer

from .nature.MapGenerator import MapGenerator
from .city.city import City
from .plot.plotter import Plotter
import pickle



map_generator_params = {
    "size": 2**11 + 1,  # Must be 2^n + 1 for diamond-square
    "scale": 1.0,
    "roughness": 0.45,
    "sea_level": 0.18,
    "river_count": 10
}

city_params = {
    "max_radius": 200,
    "growth_factor": {
        0: (1.0, -1.0),  # urban areas prefer high fertility and proximity
        1: (1.0, -0.5),  # farmland prefers high fertility but less proximity penalty
        2: (1.0, -0.2)   # industrial areas prefer high fertility but even less proximity penalty
    }
}

preference_index_to_name = {
    0 : "fertility",
    1 : "forest"
}



class WorldMap:
    def __init__(self, city_count = 5, device = "cuda", 
                map_generator_params = {} ,  city_params = {}, plotter_params = None, filename = None):
        self.city_count = city_count
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.map_generator_params = map_generator_params
        self.city_params = city_params
        self.plotter_params = plotter_params
        self.maps = None
        if filename:
            if filename == True:
                filename = "maps.pickle"
            self.load(filename)
        else:
            self.init()


    
    def init(self):
        self.maps = self.generate_maps()
        self.cities = self.generate_cities()
        self.plotter = Plotter(self.maps, self.cities, self.plotter_params)
        #self.generate_road_map()
    @timer
    def generate_maps(self):
        params = {**self.map_generator_params}
        generator = MapGenerator(**params)
        keys, maps = generator.generate()
        maps = self.convert_to_torch(keys, maps)
        return maps    
    @timer
    def generate_cities(self):
        cities = []
        for index in range(self.city_count):
            city = City(id = index+1, maps = self.maps, **self.city_params)
            if city.pos is not None:
                cities.append(city)
        return cities

    def convert_to_torch(self, keys, maps):
        #return a dict of torch tensors with keys "height", "sea", "river", "fertility", "forest", "humidity"
        return {key: torch.from_numpy(maps[i]) for i, key in enumerate(keys)}

    def grow_all(self, config):
        #config shape (city_count, 3) where each row is (amount, preference_index, land_type)
        #skip cities with amount = 0
        #do in parallel for each city:
        H, W = self.maps["city"].shape[:2]
        claimed = np.zeros((H, W), dtype=bool)

        plans = {}   # city → plan
        with ThreadPoolExecutor() as pool:
            futures = {
                pool.submit(city.plan_growth, int(config[i, 0]),
                            preference=preference_index_to_name[int(config[i, 1])], land_type=int(config[i, 2])): city
                for i, city in enumerate(self.cities) if config[i, 0] > 0
            }
            for fut in as_completed(futures):
                city = futures[fut]
                plan = fut.result()
                if plan is not None:
                    plans[city] = plan

        order = list(plans.keys())
        np.random.shuffle(order)

        for city in order:
            city.apply_growth(plans[city], already_claimed=claimed)
            gc = plans[city]["global_cells"]
            claimed[gc[:, 0], gc[:, 1]] = True

    @timer
    def grow_all_old(self, amount, preference="fertility", land_type=0, max_workers=None):
        """Parallel grow: all cities plan simultaneously, then conflicts are resolved
        in random order before writing to shared maps.

        Phase 1 (parallel): each city calls plan_growth() — pure read/compute, no writes.
        Phase 2 (serial):   plans are shuffled and applied; cells claimed by an earlier
                            city in the resolution order are skipped by later ones.
        """
        H, W = self.maps["city"].shape[:2]
        claimed = np.zeros((H, W), dtype=bool)

        # ── Phase 1: parallel planning ────────────────────────────────────
        plans = {}   # city → plan
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(city.plan_growth, amount,
                            preference=preference, land_type=land_type): city
                for city in self.cities
            }
            for fut in as_completed(futures):
                city = futures[fut]
                plan = fut.result()
                if plan is not None:
                    plans[city] = plan

        # ── Phase 2: serial conflict resolution ───────────────────────────
        # Shuffle so no city is systematically favoured
        order = list(plans.keys())
        np.random.shuffle(order)

        for city in order:
            city.apply_growth(plans[city], already_claimed=claimed)
            # Mark cells this city just claimed so later cities skip them
            gc = plans[city]["global_cells"]
            claimed[gc[:, 0], gc[:, 1]] = True

    def save(self, filename = "maps.pickle"):
        #save cities and maps
        with open(filename, "wb") as f:
            pickle.dump({
                "maps": self.maps,
                "cities": self.cities,
                "city_params": self.city_params,
                "plotter_params": self.plotter_params,
                "plotter" : self.plotter
            }, f)
    @timer
    def load(self, filename = "maps.pickle"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.maps = data["maps"]
            self.cities = data["cities"]
            self.plotter = data["plotter"]
            # Reassign each city's maps reference to the shared maps dict,
            # since pickle restores them as independent copies.
            for city in self.cities:
                city.maps = self.maps
                city.growth_factor = self.city_params["growth_factor"]
                city.max_radius = self.city_params["max_radius"]
                city.max_road_radius = self.city_params["max_road_radius"]


