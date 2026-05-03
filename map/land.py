import numpy as np
import torch
import torch.nn.functional as F

from .gpu_helpers import binary_dilation, compute_on_gpu,  masked_avg,  timer
from scipy.ndimage import distance_transform_cdt, distance_transform_edt
import matplotlib.pyplot as plt


class City:
    def __init__(self, id = 1, maps = None, pos = (0,0), max_radius = 150, growth_factor = {}):
        self.id = id
        self.name = f"City {id}"
        self.pos = pos
        self.cities = []
        self.agents = []
        self.max_radius = max_radius
        self.growth_factor = growth_factor

        self.maps = maps #3d int array (x,y, [city_id, type])
        self.register_pos(pos, 1) #register initial position with land type 1 (urban)
    
    def register_pos(self, pos, land_type = 0):
        if type(pos) == tuple:
            pos = [pos]

        for p in pos:
            self.maps["city"][p[0], p[1], 0] = self.id
            self.maps["city"][p[0], p[1], 1] = land_type #owner id, for now just 0
    
    @timer 
    def grow(self, amount = 1, preference = "fertility", land_type = 0 ):
        map = self.maps[preference]
        growth_mask = self.calculate_growth_mask(land_type)
        growth_score = self.calculate_growth(map, growth_mask, amount)
        growth_score = self.specialize_growth(growth_score, preference, land_type)
        distance_map = self.calculate_distance_map()

        #plot distance map
        

        a, b = self.growth_factor[land_type]
        growth = (growth_score ** a) * (distance_map ** b)
        max_growth_index = np.unravel_index(np.argmax(growth), growth.shape)

        self.plot_helper(growth_score ** a, "Growth Score")
        self.plot_helper(distance_map ** b, "Distance Map")

        
        self.plot_helper(growth, "Combined Growth Potential", max_growth_index)
        
        
        #print distance of max growth point from city center
        distance = np.sqrt((max_growth_index[0] - self.pos[0]) ** 2 + (max_growth_index[1] - self.pos[1]) ** 2)
        print(f"Max growth point: {max_growth_index}, Distance from city center: {distance:.2f}")

    def plot_helper(self, array, title, max_growth_index = None):
        plt.imshow(array, cmap='viridis')
        plt.scatter(self.pos[1], self.pos[0], color='red', label='City Center')
        if max_growth_index is not None:
            plt.scatter(max_growth_index[1], max_growth_index[0], color='blue', label='Max Growth Point')
        plt.title(title)
        plt.colorbar()
        plt.show()
        
        #find the argmax
    @timer
    def calculate_growth_mask(self, land_type = 0):
        sea_mask, river_mask = self.maps["sea"], self.maps["river"]
        city_mask = self.maps["city"][:,:,0] > 0

        if land_type == 1:
            urban_mask = self.maps["city"][:,:,1] == 1 
            this_city_mask = self.maps["city"][:,:,0] == self.id
            city_mask = ~urban_mask & this_city_mask
        
        growth_mask = ~sea_mask & ~river_mask & ~city_mask
        return growth_mask

    def specialize_growth(self, score, preference, land_type):
        #for now just return the score, but in the future we can add more complex logic here
        return score
    @timer
    def calculate_growth(self, map, growth_mask, amount):
        radius = int(np.ceil(np.sqrt(amount))) + 1
        score = masked_avg(map, growth_mask, radius)
        return score
    
    @timer
    def get_effective_box_indices(self, array):
        #find the bounding box of the array (true false)
        rows = np.any(array, axis=1)
        cols = np.any(array, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]


        #expand the box by self.max_radius in all directions, but keep it within the bounds of the array
        rmin = max(0, rmin - self.max_radius)
        rmax = min(array.shape[0] - 1, rmax + self.max_radius)
        cmin = max(0, cmin - self.max_radius)
        cmax = min(array.shape[1] - 1, cmax + self.max_radius)

        return rmin, rmax, cmin, cmax

    @timer
    def calculate_distance_map(self):
        #calculate distance from urban land using a distance transform
        city_mask = self.maps["city"][:,:,0] == self.id
        urban_mask = self.maps["city"][:,:,1] == 1
        city_urban_mask = (city_mask & urban_mask).cpu().numpy()
        
        cut_box = self.get_effective_box_indices(city_urban_mask)
        cut_mask = city_urban_mask[cut_box[0]:cut_box[1]+1, cut_box[2]:cut_box[3]+1]
        
        
        distance_map = distance_transform_edt(~cut_mask)
        distance_map = 1 - (distance_map / np.max(distance_map))
        distance_map[cut_mask] = 0

        distance_map_full = np.zeros_like(city_urban_mask, dtype=np.float32)
        distance_map_full[cut_box[0]:cut_box[1]+1, cut_box[2]:cut_box[3]+1] = distance_map

        
        return distance_map_full
   

