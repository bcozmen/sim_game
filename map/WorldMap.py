import matplotlib.pyplot as plt
import torch
import numpy as np

from .gpu_helpers import timer, binary_dilation
from .map_generator import MapGenerator
from .land import City
from .road import Road

import pickle



map_generator_params = {
    "size": 2**11 + 1,  # Must be 2^n + 1 for diamond-square
    "scale": 1.0,
    "roughness": 0.45,
    "sea_level": 0.18,
    "river_count": 10
}

type_to_cmap = {
    "height": "terrain",
    "fertility": "YlGn",
    "forest": "Greens",
    "humidity": "PuBuGn",
    "husbandry": "Oranges",
    "habitability": "coolwarm"
}

city_params = {
    "max_radius": 200,
    "growth_factor": {
        0: (1.0, -1.0),  # urban areas prefer high fertility and proximity
        1: (1.0, -0.5),  # farmland prefers high fertility but less proximity penalty
        2: (1.0, -0.2)   # industrial areas prefer high fertility but even less proximity penalty
    }
}



class WorldMap:
    def __init__(self, city_count = 5, device = "cuda", map_generator_params = {} , filename = None, city_params = {}):
        self.city_count = city_count
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.map_generator_params = map_generator_params
        self.city_params = city_params
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
        self.generate_road_map()
    @timer
    def generate_maps(self):
        params = {**self.map_generator_params}
        generator = MapGenerator(**params)
        keys, maps = generator.generate()
        maps = self.convert_to_torch(keys, maps)
        return maps    

    @timer
    def generate_cities(self):
        city_count = self.city_count
        #find the top city with the highest fertility that is not sea or river
        fertility = self.maps["fertility"].cpu().numpy().copy()
        sea = self.maps["sea"].cpu().numpy()
        river = self.maps["river"].cpu().numpy()

        sea = binary_dilation(sea, iterations=2)
        river = binary_dilation(river, iterations=2)

        fertility[sea | river] = 0
        #check if fertility contains nan
        cities = []
        for index in range(city_count):
            #choose the point randomly wrt fertility as a probability distribution
            if np.isnan(fertility).any():
                print("Fertility map contains NaN values. Replacing with zeros.")
                fertility = np.nan_to_num(fertility)
            if fertility.sum() == 0:
                break
            probability = fertility.flatten() / fertility.sum()

            city_index = np.unravel_index(np.random.choice(fertility.size, p=probability), fertility.shape)

            
            
            city = City(id = index+1, maps = self.maps, 
                        pos =  city_index,  **self.city_params)
            cities.append(city)

            #make radius 200 around the city uninhabitable for other cities
            y, x = np.ogrid[:fertility.shape[0], :fertility.shape[1]]
            mask = (x - city_index[1]) ** 2 + (y - city_index[0]) ** 2 <= (city.max_radius * 4) ** 2
            fertility[mask] = 0
        return cities

    @timer
    def generate_road_map(self):
        W,H = self.maps["height"].shape
        road_map = torch.zeros((W, H), dtype=torch.bool)  # False means no road, True means road
        self.maps["road"] = road_map
        self.road = Road(self.maps, self.cities)
    def convert_to_torch(self, keys, maps):
        #return a dict of torch tensors with keys "height", "sea", "river", "fertility", "forest", "humidity"
        return {key: torch.from_numpy(maps[i]) for i, key in enumerate(keys)}
    
    def plot_all(self):
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        keys = ["height", "fertility", "forest", "humidity", "husbandry" ,"habitability"]
        for ax, key in zip(axs.flatten(), keys):
            self.plot_map(ax, key,title=key.capitalize())
        plt.tight_layout()
        plt.show()
    def plot(self, map_type = "height", ax = None, show = True):
        if ax is None:
            fig, ax = plt.subplots(figsize=(13, 13))
        self.plot_map(ax, map_type)

        if show:
            plt.show()

    def plot_map(self, ax, map_type = "height", title = None):
        this_map = self.maps[map_type].cpu().numpy()
        vmin, vmax = 0, 1
        if map_type == "height":
            vmin, vmax = -0.2, 1

        #add colorbar per axis
        ax.imshow(this_map, cmap=type_to_cmap.get(map_type, "viridis"), vmin=vmin, vmax=vmax)
        plt.colorbar(ax.imshow(this_map, cmap=type_to_cmap.get(map_type, "viridis"), vmin=vmin, vmax=vmax), ax=ax)
        ax.set_title(title if title else f"{map_type.capitalize()} Map")
       
       
        overlay = self.get_overlay(color=(0.0, 0.0, 0.0))  # Semi-transparent blue for sea and river
        overlay = self._plot_sea_and_river_overlay(overlay)
        overlay = self._plot_cities(overlay)
        
        

        for city in self.cities:
            #scatter the city position
            ax.scatter(city.pos[1], city.pos[0], c='white',s=25, edgecolors = 'black')
        if "road" in self.maps:
            overlay = self._plot_roads(overlay)
        ax.imshow(overlay)

    def _plot_sea_and_river_overlay(self, overlay):
        sea = self.maps["sea"].cpu().numpy()
        river = self.maps["river"].cpu().numpy()
        H, W = sea.shape



        # sea mask
        sea_mask = np.logical_or(sea, river)
        overlay = self.mask_overlay(overlay, sea_mask, color=(0.0, 0.5, 1.0), alpha=0.9)  # Semi-transparent blue for sea and river
        return overlay

    def _plot_rural_farmland_overlay(self, city, overlay):
        mask = (city[:, :, 1] == 0) & (city[:, :, 2] == 0) & (city[:, :, 0] != 0)
        overlay = self.mask_overlay(overlay, mask, color=(1.0, 1.0, 0.0), alpha=1.0)   # Yellow for rural farmland
        return overlay
         
    
    def _plot_rural_forest_overlay(self, city, overlay):
        mask = (city[:, :, 1] == 0) & (city[:, :, 2] == 1) & (city[:, :, 0] != 0)
        overlay = self.mask_overlay(overlay, mask, color=(0.0, 0.5, 0.0), alpha=1.0)   # Dark Green for rural forest
        return overlay

    def _plot_urban_overlay(self, city, overlay):
        mask = (city[:, :, 1] == 1) & (city[:, :, 0] != 0)
        overlay = self.mask_overlay(overlay, mask, color=(1.0, 0.0, 0.0), alpha=1.0)   # Red for urban areas
        return overlay

    def _plot_cities(self, overlay):
        H, W = self.maps["height"].shape
        
            
        city = self.maps["city"].cpu().numpy()
        overlay = self._plot_rural_farmland_overlay(city, overlay)
        overlay = self._plot_rural_forest_overlay(city, overlay)
        overlay = self._plot_urban_overlay(city, overlay)
        return overlay


    def _plot_roads(self, overlay):
        road_mask = self.maps["road"].cpu().numpy()
        H, W = road_mask.shape
        overlay = self.mask_overlay(overlay, road_mask, color=(0, 0, 0), alpha=1.0)  # Grey color for roads
        return overlay
    
    def get_overlay(self, color = (1.0, 1.0, 0.0)):
        H, W = self.maps["height"].shape
        overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA overlay
        overlay[:, :, 0] = color[0]
        overlay[:, :, 1] = color[1]
        overlay[:, :, 2] = color[2]
        return overlay

    def mask_overlay(self, overlay, mask, color = (1.0, 1.0, 0.0), alpha = 1.0):
        overlay[mask, 0] = color[0]
        overlay[mask, 1] = color[1]
        overlay[mask, 2] = color[2]
        overlay[mask, 3] = alpha
        return overlay


    def save(self, filename = "maps.pickle"):
        #save cities and maps
        with open(filename, "wb") as f:
            pickle.dump({
                "maps": self.maps,
                "cities": self.cities,
                "road": self.road,
            }, f)
    
    def load(self, filename = "maps.pickle"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.maps = data["maps"]
            self.cities = data["cities"]
            self.road = data.get("road", None)
            # Reassign each city's maps reference to the shared maps dict,
            # since pickle restores them as independent copies.
            for city in self.cities:
                city.maps = self.maps
                city.max_radius = self.city_params.get("max_radius", city.max_radius)
                city.growth_factor = self.city_params.get("growth_factor", city.growth_factor)
                city.max_road_radius = self.city_params.get("max_road_radius", getattr(city, "max_road_radius", 50))
                city.island_id = self.city_params.get("island_id", getattr(city, "island_id", None))