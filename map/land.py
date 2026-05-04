import numpy as np
import torch
import torch.nn.functional as F
from .gpu_helpers import binary_dilation, timer, masked_avg
from .land_helper import *
from scipy.ndimage import distance_transform_cdt, distance_transform_edt
import matplotlib.pyplot as plt

growth_factor = {
    "preference": [1.0, 1.0, 1.0],  # all land types prefer high fertility equally for now
    "proximity": [-1.0, -0.5, -0.2],
    "distance": [-0.5, -0.3, -0.1],
}

preference_types = {
    "fertility": 0,
    "forest": 1,
}
class City:
    def __init__(self, id = 1, maps = None, pos = (0,0), max_radius = 150, max_road_radius = 10, growth_factor = {}):
        self.id = id
        self.name = f"City {id}"
        self.pos = pos
        self.cities = []
        self.agents = []
        self.max_radius = max_radius
        self.max_road_radius = max_road_radius
        self.growth_factor = growth_factor

        self.maps = maps #3d int array (x,y, [city_id, type])
        self.island_id = maps["island"][pos[0], pos[1]].item()
        self.register_pos(pos, 1) #register initial position with land type 1 (urban)
        self.debug = False

    def plot(self, array, title = "", cmap = "viridis"):
        if not self.debug:
            return
        plt.imshow(array, cmap=cmap)
        plt.title(title)
        plt.colorbar()
        plt.show()


    
    
    def gather_maps(self, cut_indices):
        keys = ["height","fertility", "forest", "habitability", "road", "sea", "river", "city", "island"]
        cut_maps = {}
        for key in keys:
            cut_maps[key] = cut_array(self.maps[key].numpy(), cut_indices)
        return cut_maps

    def grow(self, amount = 1, preference = "fertility", land_type = 0 ):
        city_urban_mask = self._get_city_urban_mask(self.maps)
        cut_box = self.get_cut_box_indices(city_urban_mask)
        cut_maps = self.gather_maps(cut_box)
        cut_city_urban_mask = cut_array(city_urban_mask, cut_box)
        blocked_mask = self._get_blocked_mask(cut_maps, land_type)
        

        cut_maps["city_urban_mask"] = cut_city_urban_mask
        cut_maps["blocked_mask"] = blocked_mask
           
        preference_score = cut_maps[preference]
        distance_score = self.calculate_distance_with_cost(cut_maps, land_type = land_type)
        self.plot(cut_maps["city_urban_mask"], title = "city urban mask")
        self.plot(cut_maps["blocked_mask"], title = "blocked mask")

        self.plot(preference_score, title = "preference score")
        self.plot(distance_score, title = "distance score")

        score = (preference_score ** self.growth_factor["preference"][land_type]) * \
                (distance_score ** self.growth_factor["distance"][land_type])  * blocked_mask

        self.plot(score, title = "final score")

        if np.all(score == 0):
            return
        best_location = np.unravel_index(np.argmax(score), score.shape)
        
        #chosen_locations = grow_region(score, blocked_mask, best_location, amount)
        chosen_locations = self.choose_n_best_locations(score, amount)
        global_locations = convert_local_to_global_vectorized(chosen_locations, cut_box)
        self.register_vectorized(global_locations, land_type, preference)
        if land_type == 1:
            #if urban, also register roads to connect to the city center
            # Refresh city_urban_mask to include the newly registered tiles
            updated_urban_mask = self._get_city_urban_mask(self.maps)
            cut_maps["city_urban_mask"] = cut_array(updated_urban_mask, cut_box)
            self.plot(self.maps["road"].numpy(), title = "road map before growth")
            while True:
                path = self.construct_roads(cut_maps)
                if path is False:
                    break
                #convert path to global coordinates
                global_path = [convert_local_to_global(p, cut_box) for p in path]
                self.register_roads(global_path)
            self.plot(self.maps["road"].numpy(), title = "road map after growth")

    def register_roads(self, path):
        for p in path:
            self.maps["road"][p[0], p[1]] = 1
    def construct_roads(self, maps):
        #distance to roads
        roads = maps["road"]
        city_urban_mask = maps["city_urban_mask"]
        distance_map = distance_transform_edt(roads == 0)

        
        #choose the most distant location in the city_urban_mask that is greater than max_road_radius
        distance_map[~city_urban_mask] = 0  # only consider locations within the city urban area
        #chose the location with the maximum distance to road
        best_location = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        if distance_map[best_location] < self.max_road_radius:
            return False #no need to build road
        
        #find distance to nearest road from the best location using terrain distance
        distance_map, parent = self.terrain_distance_helper(maps, best_location)

        #find the best location to connect to the road 
        distance_map[~roads.astype(bool)] = np.inf  # only consider locations on roads
        best_road_location = np.unravel_index(np.argmin(distance_map), distance_map.shape)
        if distance_map[best_road_location] == np.inf:
            return False #no path to road

        #reconstruct path from best_road_location back to best_location using parent
        path = []
        current = best_road_location
        w = parent.shape[1]

        while not np.array_equal(current, best_location):
            path.append(current)
            flat = parent[current[0], current[1]]
            if flat < 0:  # reached source with no parent
                break
            current = (flat // w, flat % w)
        path.append(best_location)
        path.reverse()

        return path
    def terrain_distance_helper(self, maps, pos = None, land_type = 0):
        height_map = maps["height"]
        city_urban_mask = maps["city_urban_mask"]
        blocked_mask = maps["blocked_mask"].copy()
        road_map = maps["road"]
        river_map = maps["river"]
        
        if pos is not None:
            #create a source mask with True at the given position
            city_urban_mask = np.zeros_like(city_urban_mask, dtype=bool)
            city_urban_mask[pos[0], pos[1]] = True
            

        blocked_mask = blocked_mask.astype(bool) & (~river_map.astype(bool))
        blocked_mask = 1 - blocked_mask.astype(float) #convert to 0 for blocked

        if land_type == 0:
            #remove the rural area from the blocked mask, allowing dijsktra to work
            rural_mask = (maps["city"][:, :, 0] == self.id) & (maps["city"][:, :, 1] == 0)
            blocked_mask[rural_mask] = 0
            self.plot(blocked_mask, title = "blocked mask for terrain distance with land type 1")

        distance_map, parent = terrain_distance(height_map, city_urban_mask, blocked_mask, road_map, river_map)
        return distance_map, parent
    def calculate_distance_with_cost(self, maps, land_type = 0):
        #self.plot(maps["blocked_mask"], title = "BUG blocked mask for distance calculation")
        #self.plot(maps["city_urban_mask"], title = "BUG  city urban mask for distance calculation")
        distance_map, parent = self.terrain_distance_helper(maps, land_type = land_type)
        #self.plot(distance_map, title = "BUG raw distance map")
        
        distance_map[distance_map > self.max_radius] = self.max_radius  # beyond max radius is not reachable
        distance_score = 1 - (distance_map / self.max_radius)
        return distance_score
    def choose_n_best_locations(self, score, n):
        #given 2d score array, choose n best locations without replacement, return the inddices as a (n, 2) array
        flat_indices = np.argpartition(score.flatten(), -n)[-n:]  # Get the indices of the n largest values
        best_indices = np.unravel_index(flat_indices, score.shape)  # Convert flat indices to 2D indices
        return np.column_stack(best_indices)  # Combine into (n, 2) array
    def _get_city_urban_mask(self, maps):
        mask = (maps["city"][:, :, 0] == self.id) & (maps["city"][:, :, 1] == 1)
        #if tensor, convert to numpy
        if (isinstance(mask, torch.Tensor)):
            mask = mask.cpu().numpy()
        return mask.copy()
    def get_cut_box_indices(self, array):
        box = get_bounding_box_indices(array)
        min_y, max_y, min_x, max_x = box
        #grow by self.max_radius but not beyond array bounds
        min_y = max(0, min_y - self.max_radius)
        max_y = min(array.shape[0], max_y + self.max_radius)
        min_x = max(0, min_x - self.max_radius)
        max_x = min(array.shape[1], max_x + self.max_radius)

        #convert to int
        min_y = int(min_y)
        max_y = int(max_y)
        min_x = int(min_x)
        max_x = int(max_x)
        return (min_y, max_y, min_x, max_x)

    
    def _get_blocked_mask(self, maps, land_type):
        sea = maps["sea"]
        river = maps["river"]
        #other cities
        other_cities = (maps["city"][:, :, 0] != self.id) & (maps["city"][:, :, 0] != 0)
        this_city = (maps["city"][:, :, 0] == self.id)
        this_urban = (maps["city"][:, :, 0] == self.id) & (maps["city"][:, :, 1] == 1)

        island = maps["island"] != self.island_id
        
        blocked = sea | river | other_cities | island
        if land_type == 0: #rural areas cannot grow on this city 
            blocked = blocked | this_city 
        else: #urban areas cannot grow on this city or other cities
            blocked = blocked | this_urban

        blocked_mask = 1 - blocked.astype(float)
        return blocked_mask

    

    



        
        


    def register_pos(self, pos, land_type = 0):
        if type(pos) == tuple:
            pos = [pos]

        for p in pos:
            self.maps["city"][p[0], p[1], 0] = self.id
            self.maps["city"][p[0], p[1], 1] = land_type #owner id, for now just 0
            self.maps["city"][p[0], p[1], 2] = 0
    def register_vectorized(self, pos_array, land_type = 0, preference = "fertility"):
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 0] = self.id
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 1] = land_type
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 2] = preference_types[preference]