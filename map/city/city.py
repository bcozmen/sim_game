import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..functions.gpu import binary_dilation
from ..functions.path_finding import dijkstra, astar
from .city_helper import *

from ..functions.helper import timer
growth_factor = {
    "preference": [1.0, 1.0, 1.0],
    "distance": [1.0, 1.0, 1.0],
}

growth_factor_name_to_index = {
    "preference": 0,
    "distance": 1,
}
growth_factor = np.asarray([
    [1.0, 1.0, 1.0],  # preference
    [1.0, 1.0, 1.0],  # proximity
    [1.0, 1.0, 1.0],  # distance
])



preference_types = {
    "fertility": 0,
    "forest": 1,
}

class City:
    def __init__(self, id=1, maps=None, 
                max_radius=150, max_road_radius=10, city_init_factor = 4,growth_factor=growth_factor):
        self.id = id
        self.maps = maps

        self.max_radius = max_radius
        self.max_road_radius = max_road_radius
        self.city_init_factor = city_init_factor
        self.growth_factor = growth_factor

        self.pos = self.init_location()
        if self.pos is not None:
            self.init_roads()

    # ------------------------------------------------------------------ #
    #  Growth                                                              #
    # ------------------------------------------------------------------ #
    @timer
    def grow(self, amount, preference="fertility", land_type=0, plot=False):
        plan = self.plan_growth(amount, preference=preference, land_type=land_type)
        if plan is None:
            return False
        self.apply_growth(plan)

    def plan_growth(self, amount, preference="fertility", land_type=0):
        """Read-only phase: compute desired cells without writing to shared maps.
        Returns a plan dict, or None if there is nothing to grow into."""
        cut_maps = self.get_cut_maps()
        growth_mask = self.get_growth_mask(cut_maps, land_type=land_type)
        growth_score = self.calculate_score(cut_maps, growth_mask, land_type=land_type)
        chosen_cells, global_cells = self.choose_growth_cells(
            cut_maps, growth_score, growth_mask, amount, land_type=land_type
        )
        if chosen_cells is False:
            return None
        return {
            "global_cells": global_cells,
            "chosen_cells": chosen_cells,
            "cut_maps": cut_maps,
            "land_type": land_type,
            "preference": preference,
        }

    def apply_growth(self, plan, already_claimed=None):
        """Write phase: register cells, skipping any already claimed this tick.
        `already_claimed` is an optional boolean mask (H, W) of cells taken by
        other cities in the same parallel grow round."""
        global_cells = plan["global_cells"]

        if already_claimed is not None:
            # Filter out cells that were grabbed by a city that resolved before us
            keep = ~already_claimed[global_cells[:, 0], global_cells[:, 1]]
            global_cells = global_cells[keep]
            if len(global_cells) == 0:
                return

        self.register(global_cells, land_type=plan["land_type"], preference=plan["preference"])

        if plan["land_type"] == 1:
            # chosen_cells[0] is the seed cell in local coords; reuse cut_maps
            chosen_cells = plan["chosen_cells"]
            if already_claimed is not None:
                # re-filter chosen_cells in sync with global_cells
                cut_indices = plan["cut_maps"]["cut_indices"]
                kept_global = global_cells  # already filtered above
                kept_local = kept_global - np.array([cut_indices[0], cut_indices[2]])
                if len(kept_local) > 0:
                    self.construct_roads(plan["cut_maps"], kept_local[0])

    def calculate_score(self, cut_maps, growth_mask, land_type=0):
        distance_score = self.calculate_distance(cut_maps)
        preference_score = cut_maps["fertility"]
        growth_score = (
            preference_score ** self.growth_factor[0][land_type]
            * distance_score ** self.growth_factor[1][land_type]
            * growth_mask
        )
        return growth_score
    
    @timer
    def calculate_distance(self, cut_maps):
        cost_map = self.get_cost_map(cut_maps, alpha=5, multiplicative=False)
        _, dist, _ = dijkstra(cost_map, cut_maps["urban_mask"], goals=None, max_cost = min(150,self.max_radius))
        return normalize_inverted(dist)
    
    def get_growth_mask(self, cut_maps, land_type=0):
        water_mask = cut_maps["sea"] | cut_maps["river"]
        other_city_mask = (cut_maps["city"][:, :, 0] > 0) & (cut_maps["city"][:, :, 0] != self.id)
        growth_mask = ~(water_mask | other_city_mask | cut_maps["urban_mask"])
        if land_type == 0:
            growth_mask &= ~cut_maps["rural_mask"]
        elif land_type == 1:
            growth_mask &= ~cut_maps["road"]
        return growth_mask
    
    @timer
    def choose_growth_cells(self, maps, growth_score, growth_mask, amount, land_type=0):
        # Remove places we cannot grow into because there isn't enought space
        growth_mask = remove_small_islands(growth_mask, min_size=amount)
        # Chose cells to grow into based on the growth score and the growth mask
        chosen_cells, growth_mask = self.grow_into(maps, growth_score, amount, growth_mask, land_type=land_type)
        if len(chosen_cells) == 0:
            return False, False

        chosen_cells = np.array(chosen_cells)
        global_cells = convert_local_to_global(chosen_cells, maps["cut_indices"])
        return chosen_cells, global_cells

    @timer
    def grow_into(self, maps, growth_score, amount, growth_mask, land_type=0):
        probs = soft_max(growth_score)
        probs[~growth_mask] = 0
        index = choose_from_pdf(probs)
        #index = np.unravel_index(np.argmax(probs), probs.shape)

        coef = 1 if land_type == 1 else 5
        cost_map = 1 + coef * (maps["directional_slope"] ** 2)
        cost_map[~growth_mask] = np.inf

        if land_type == 1:
            roads = maps["road"]
            cost_map[roads] = np.inf
        elif land_type == 0:
            roads = maps["road"]
            cost_map[roads] *= 2.0
        
        _, dist, _ = dijkstra(cost_map, index, goals=None, max_amount=amount*4)
        dist[~growth_mask] = np.inf

        chosen_indices = choose_indices(dist, amount, minimize=True)
        growth_mask[chosen_indices[:, 0], chosen_indices[:, 1]] = False
        return chosen_indices, growth_mask

    # ------------------------------------------------------------------ #
    #  Road construction                                                   #
    # ------------------------------------------------------------------ #
    @timer
    def construct_roads(self, maps, cell):
        slope = maps["directional_slope"]
        cost_map = 1 + slope ** 2 + np.random.rand(*slope.shape) * 0.1
        _, _, path = dijkstra(cost_map, cell, maps["road"])

        if path is None or len(path) < self.max_road_radius:
            return

        path = np.array(path)
        self.register_road(convert_local_to_global(path, maps["cut_indices"]))
        self.register_junction(convert_local_to_global(np.array([cell]), maps["cut_indices"]).reshape(-1, 2))
        self.add_random_road(maps, cell)


    @timer
    def add_random_road(self, maps, cell):
        if np.random.rand() > 0.2:
            return
        junctions = np.argwhere(maps["junction"])
        if len(junctions) == 0:
            return

        distances = np.linalg.norm(junctions - cell, axis=1)
        distances = 1 - (distances - distances.min()) / (distances.max() - distances.min() + 1e-8)
        target = junctions[np.random.choice(len(junctions), p=soft_max(distances))]

        road, sea, river = maps["road"], maps["sea"], maps["river"]
        cost_map = 2 + 10 * (maps["directional_slope"] ** 2)
        cost_map[road] *= 0.65
        cost_map[sea] = np.inf
        cost_map[river & ~road] *= 2

        _, _, path = astar(cost_map, cell, tuple(target))
        if path is None:
            return
        self.register_road(convert_local_to_global(np.array(path), maps["cut_indices"]))

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #

    def init_roads(self):
        goals = self.maps["city"][:, :, 0].cpu().numpy().copy()
        goals[self.pos[0], self.pos[1]] = 0  # exclude self
        starts = [self.pos]

        for _ in range(int(goals.sum())):
            _, _, path = dijkstra(self.get_cost_map(self.get_maps()), starts, goals > 0)
            if path is None:
                break
            path = np.array(path)
            self.register_road(path)
            self.register_junction(path, cut=True)
            goals[path[-1, 0], path[-1, 1]] = 0

    def init_location(self):
        city_map = self.maps["city"][:, :, 0].cpu().numpy()
        city_mask = build_city_exclusion_mask(city_map, self.max_radius, factor=self.city_init_factor)

        river_mask = binary_dilation(self.maps["river"].cpu().numpy(), iterations=2)
        sea_mask   = binary_dilation(self.maps["sea"].cpu().numpy(),   iterations=2)
        location_mask = ~(city_mask | river_mask | sea_mask)

        height_mask = self.maps["height"] < 0.85
        location_score = (self.maps["fertility"] * location_mask * height_mask).cpu().numpy()

        if np.sum(location_score) == 0:
            self.pos = None
            return None

        pos = np.unravel_index(np.argmax(location_score), location_score.shape)
        self.register(np.array([pos]), land_type=1, preference="fertility")
        return pos



    # ------------------------------------------------------------------ #
    #  Registration helpers                                                #
    # ------------------------------------------------------------------ #

    def register(self, pos_array, land_type=0, preference="fertility"):
        pos_array = np.asarray(pos_array)
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 0] = self.id
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 1] = land_type
        self.maps["city"][pos_array[:, 0], pos_array[:, 1], 2] = preference_types[preference]

    def register_road(self, pos_array):
        pos_array = np.asarray(pos_array)
        self.maps["road"][pos_array[:, 0], pos_array[:, 1]] = True

    def register_junction(self, pos, cut=False):
        if cut:
            pos = pos[::self.max_road_radius]
        junction_map = self.maps["junction"]
        for p in pos:
            y, x = p
            if not junction_map[circle_mask(junction_map.shape, (y, x), self.max_road_radius)].any():
                junction_map[y, x] = True

    # ------------------------------------------------------------------ #
    #  Map helpers                                                         #
    # ------------------------------------------------------------------ #

    def get_cost_map(self, maps, alpha=5, multiplicative=True, land_type=None):
        cost_map = maps["directional_slope"].copy()
        if multiplicative:
            cost_map = np.exp(alpha * cost_map)
        else:
            cost_map = 1 + alpha * cost_map ** 2

        road_factor = 0.5 if (multiplicative or land_type == 1) else 0.7
        cost_map[maps["sea"]] = np.inf
        cost_map[maps["road"]] *= road_factor
        cost_map[maps["river"] & ~maps["road"]] *= 5
        return cost_map

    def get_city_mask(self, land_type):
        """Return a boolean numpy mask for this city's cells of the given land_type."""
        city_cells = self.maps["city"][:, :, 0] == self.id
        type_cells = (self.maps["city"][:, :, 1] > 0) if land_type == 1 else (self.maps["city"][:, :, 1] == 0)
        return (city_cells & type_cells).cpu().numpy()

    def get_cut_maps(self):
        urban_mask = self.get_city_mask(1)
        rural_mask = self.get_city_mask(0)
        #indices = get_box_indices(urban_mask, self.max_radius)
        sea_mask = self.maps["sea"].cpu().numpy()
        other_cities_mask = (self.maps["city"][:, :, 0] > 0) & (self.maps["city"][:, :, 0] != self.id)
        other_cities_mask = other_cities_mask.cpu().numpy()
        growth_mask = ~(sea_mask | other_cities_mask)
        indices = get_box_indices_smart(urban_mask, growth_mask, self.max_radius)
        
        cut_maps = {
            key: (cut_box(self.maps[key].cpu().numpy(), indices)
                  if self.maps[key].cpu().numpy().ndim >= 2
                  else self.maps[key].cpu().numpy())
            for key in self.maps
        }
        cut_maps["urban_mask"] = cut_box(urban_mask, indices)
        cut_maps["rural_mask"] = cut_box(rural_mask, indices)
        cut_maps["cut_indices"] = indices
        return cut_maps

    def get_maps(self):
        return {key: self.maps[key].cpu().numpy() for key in self.maps}



