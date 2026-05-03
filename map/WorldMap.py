import matplotlib.pyplot as plt
import torch
import numpy as np

from .gpu_helpers import timer, binary_dilation
from .map_generator import MapGenerator
from .land import City
from .road import Road



map_generator_params = {
    "size": 2**11 + 1,  # Must be 2^n + 1 for diamond-square
    "scale": 1.0,
    "roughness": 0.45,
    "sea_level": 0.18,
    "river_count": 10
}

type_to_cmap = {
    "height": "terrain",
    "sea": "Blues",
    "river": "Blues",
    "fertility": "YlGn",
    "forest": "Greens",
    "humidity": "PuBuGn"
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
                filename = "maps.pt"
            self.load_maps(filename)
        else:
            self.init()


    
    def init(self):
        self.maps = self.generate_maps()
        self.generate_city_map()
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
    def generate_city_map(self):
        W,H = self.maps["height"].shape
        #city id, owner id, building type
        city_map = torch.zeros((W, H, 3), dtype=torch.int32)  # 0 means no city, positive integers are city IDs
        self.maps["city"] = city_map
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
            mask = (x - city_index[1]) ** 2 + (y - city_index[0]) ** 2 <= city.max_radius ** 2
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

    def plot_maps(self, map_type = "height"):
        H, W = self.maps["height"].shape
        
        fig, ax = plt.subplots(figsize=(10, 10))    

        this_map = self.maps[map_type].cpu().numpy()
        sea = self.maps["sea"].cpu().numpy()
        river = self.maps["river"].cpu().numpy()

        plt.imshow(this_map, cmap=type_to_cmap.get(map_type, "viridis"))
        plt.title(f"{map_type.capitalize()} Map")
        # Add colorbar for reference
        plt.colorbar()

        # Create sea and river overlays
        overlay = torch.zeros((H, W, 4), dtype=torch.float32)  # RGBA overlay
        #make a nice blue
        overlay[:, :, 0] = 0.0  # Red channel
        overlay[:, :, 1] = 0.0  # Green channel
        overlay[:, :, 2] = 1.0  # Blue channel
        overlay[:, :, 3] = 0.0  # Alpha channel (transparency

        # sea mask
        sea_mask = np.logical_or(sea, river)
        overlay[sea_mask, 3] = 1.0  # Semi-transparent blue for sea and river
        plt.imshow(overlay)

        # Plot cities as red dots
        for city in self.cities:
            plt.scatter(city.pos[1], city.pos[0], color='red', label='City Center')

        # plot roads as grey lines
        if "road" in self.maps:
            road_mask = self.maps["road"].cpu().numpy()
            overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA overlay for roads
            overlay[road_mask, :3] = 0.5  # Grey color for roads
            overlay[road_mask, 3] = 1.0  # Fully opaque
            plt.imshow(overlay)

        plt.show()


    def save_maps(self, filename = "maps.pt"):
        torch.save(self.maps, filename)
    
    def load_maps(self, filename = "maps.pt"):
        self.maps = torch.load(filename)