import numpy as np
import torch
import torch.nn.functional as F

from .gpu_helpers import binary_dilation, compute_on_gpu,  masked_avg,  timer
from scipy.ndimage import distance_transform_cdt, distance_transform_edt
import matplotlib.pyplot as plt

growth_factor = {
    "preference": [1.0, 1.0, 1.0],  # all land types prefer high fertility equally for now
    "proximity": [-1.0, -0.5, -0.2],
    "distance": [-0.5, -0.3, -0.1],
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
        self.register_pos(pos, 1) #register initial position with land type 1 (urban)
    
    def register_pos(self, pos, land_type = 0):
        if type(pos) == tuple:
            pos = [pos]

        for p in pos:
            self.maps["city"][p[0], p[1], 0] = self.id
            self.maps["city"][p[0], p[1], 1] = land_type #owner id, for now just 0

    def register_vectorized(self, pos_array, land_type = 0):
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 0] = self.id
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 1] = land_type

    def get_bounding_box_indices(self, array):
        #find the bounding box of the True values in the array and return the indices of the box
        indices = np.argwhere(array)
        if len(indices) == 0:
            return (0, array.shape[0], 0, array.shape[1])
        min_y, min_x = indices.min(axis=0)
        max_y, max_x = indices.max(axis=0)
        return (min_y, max_y, min_x, max_x)

    def cut_box(self,array, margin = 2):
        #find the bounding box of the True values in the array and cut a box around it with the given margin, return the cut array and the indices of the cut
        indices = np.argwhere(array)
        if len(indices) == 0:
            return array, (0, array.shape[0], 0, array.shape[1])
        min_y, min_x = indices.min(axis=0)
        max_y, max_x = indices.max(axis=0)
        min_y = max(0, min_y - margin)
        max_y = min(array.shape[0], max_y + margin)
        min_x = max(0, min_x - margin)
        max_x = min(array.shape[1], max_x + margin)
        return array[min_y:max_y, min_x:max_x], (min_y, max_y, min_x, max_x)

    @timer
    def grow(self, amount = 1, preference = "fertility", land_type = 0 ):
        blocked_mask = self._get_blocked_mask(land_type=land_type)
        
        preference_score = self.calculate_preference_score(amount, preference, land_type, blocked_mask)
        reachability_score = self.calculate_reachability_score(land_type)
        distance_score = self.calculate_distance_score(land_type)
        
        self.plot(preference_score, f"Preference Score for {self.name}")
        self.plot(reachability_score, f"Reachability Score for {self.name}")
        self.plot(distance_score, f"Distance Score for {self.name}")

        
        self.plot(blocked_mask, f"Blocked Mask for {self.name}")
        score = (preference_score ** self.growth_factor["preference"][land_type]) * \
                (reachability_score ** self.growth_factor["proximity"][land_type]) * \
                (distance_score ** self.growth_factor["distance"][land_type])  * blocked_mask

    

        chosen_index = self.choose_growth_location(score)
        self.plot(score, f"Growth Score for {self.name}", chosen_index)

        self.grow_at_location(score, chosen_index, land_type, amount)
        pass
    

    


    def plot(self, score, title, chosen_index = None):
        #cut score to radius around city center
        cut_indices = (slice(max(0, self.pos[0]-self.max_radius), min(score.shape[0], self.pos[0]+self.max_radius)),
                       slice(max(0, self.pos[1]-self.max_radius), min(score.shape[1], self.pos[1]+self.max_radius)))
        score = score[cut_indices]
        plt.imshow(score, cmap='viridis')
        plt.scatter(self.pos[1] - cut_indices[1].start, self.pos[0] - cut_indices[0].start, color='red', label='City Center', s=5)
        if chosen_index is not None:
            plt.scatter(chosen_index[1] - cut_indices[1].start, chosen_index[0] - cut_indices[0].start, color='blue', s=5, label='Chosen Growth Point')
        
        plt.title(title)
        plt.colorbar()
        plt.show()
    @timer
    def _get_city_urban_mask(self):
        return (self.maps["city"][:, :, 0] == self.id) & (self.maps["city"][:, :, 1] == 1)
    @timer
    def _get_blocked_mask(self, land_type):
        sea = self.maps["sea"]
        river = self.maps["river"]
        #other cities
        other_cities = (self.maps["city"][:, :, 0] != self.id) & (self.maps["city"][:, :, 0] != 0)
        this_city = (self.maps["city"][:, :, 0] == self.id)
        this_urban = self.maps["city"][:, :, 1] == 1

        blocked = sea | river | other_cities
        if land_type == 0: #rural areas cannot grow on this city 
            blocked = blocked | this_city 
        else: #urban areas cannot grow on this city or other cities
            blocked = blocked | this_urban

        blocked_mask = 1 - blocked.numpy().astype(float)
        return blocked
    def _normalize(self, array):
        return (array - array.min()) / (array.max() - array.min() + 1e-5)
    @timer
    def calculate_preference_score(self, amount, preference, land_type, blocked_mask):
        map = self.maps[preference].numpy().copy()
        radius = int(np.ceil(np.sqrt(amount)))
        if radius > 1:
            map = masked_avg(map, blocked_mask, radius)
        
        return map
    @timer
    def calculate_reachability_score(self, land_type):
        roads = self.maps["road"]
        distance_to_road = distance_transform_edt(~roads)

        road_radius = self.max_road_radius
        if land_type == 1: #urban areas can only grow very close to roads
            road_radius *= 0.1

        distance_to_road[distance_to_road < road_radius] = 0
        reachability_score = 1 - (distance_to_road / self.max_radius)

        reachability_score[distance_to_road > self.max_radius] = 0  # beyond max radius is not reachable

        return reachability_score
    
    @timer
    def calculate_distance_score(self, land_type):
        city_urban_mask = self._get_city_urban_mask()
        distance_map = distance_transform_edt(~city_urban_mask)

        max_radius = self.max_radius
        if land_type == 1: #urban areas cannot grow beyond max radius
            max_radius *= 0.5 #urban areas have smaller max radius

        distance_score = 1 - (distance_map / max_radius)
        distance_score[distance_map > max_radius] = 0  # beyond max radius is not suitable for growth
        return distance_score
    @timer
    def choose_growth_location(self, score, amount = 1):
        max_growth_index = np.unravel_index(np.argmax(score), score.shape)
        return max_growth_index
    def grow_at_location(self, score, chosen_index, land_type, amount = 1):
        chosen_mask = np.zeros_like(score, dtype=bool)
        chosen_mask[chosen_index] = True
        for _ in range(amount - 1):
            cut_mask, indices = self.cut_box(chosen_mask, margin=2)
            cut_score = score[indices[0]:indices[1], indices[2]:indices[3]]
            best_neighbor_index = self.find_best_neighbor(cut_score, cut_mask)
            if cut_score[best_neighbor_index] == 0:
                break  # no more valid neighbors to grow into
            
            uncut_index = (best_neighbor_index[0] + indices[0], best_neighbor_index[1] + indices[2])
            chosen_mask[uncut_index] = True
            score[uncut_index] = 0  # prevent growing into the same cell again
        indices_to_grow = np.argwhere(chosen_mask)

        
        print(f"Growing {len(indices_to_grow)} cells at indices: {indices_to_grow.shape}")
        self.register_vectorized(indices_to_grow, land_type=land_type)
            
    def find_best_neighbor(self, score, chosen_mask):
        #dilate chosen mask to find neighbors
        neighbor_mask = binary_dilation(chosen_mask) & ~chosen_mask
        neighbor_scores = score * neighbor_mask
        best_neighbor_index = np.unravel_index(np.argmax(neighbor_scores), score.shape)
        return best_neighbor_index