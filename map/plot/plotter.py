import matplotlib.pyplot as plt
import numpy as np

from ..functions.gpu import binary_dilation
from .helper import *

colors = {
    "sea": (0.0, 0.5, 1.0, 0.9),  # Semi-transparent blue for sea
    "fertility": (1.0, 1.0, 0.0, 0.7),  # Yellow for rural farmland
    "forest": (0.0, 0.5, 0.0, 0.7),  # Dark Green for rural forest
    "urban": (1.0, 0.0, 0.0, 0.7),  # Red for urban areas
    "road": (0.0, 0.0, 0.0, 1.0),  # Grey color for roads
    "edge": (0.5, 0.5, 0.5, 1.0),  # Grey color for city edges
}

type_to_cmap = {
    "height": "terrain",
    "fertility": "YlGn",
    "forest": "Greens",
    "humidity": "PuBuGn",
    "husbandry": "Oranges",
    "habitability": "coolwarm"
}

plotter_params = {
    "colors": colors,
    "type_to_cmap": type_to_cmap
}



class Plotter:
    def __init__(self, maps, cities, plotter_params = None):
        self.maps = maps
        self.cities = cities
        if plotter_params is None:
            plotter_params = {
                "colors": colors,
                "type_to_cmap": type_to_cmap
            }
        self.colors = plotter_params["colors"]
        self.type_to_cmap = plotter_params["type_to_cmap"]
        self.sea_mask = get_sea_and_river_mask(maps)

    def print_info(self):
        density = 0.001
        city_map = self.maps["city"].cpu().numpy()
        for city in self.cities:
            urban_mask = (city_map[:, :, 1] == 1) & (city_map[:, :, 0] == city.id)
            area = urban_mask.sum()
            population = int(area * 100 * density)
            #10 x 10 m per cell = 100 m2 per cell = 0.0001 km2 per cell
            area_in_km2 = area * 0.0001
            print(f"City {city.id}: Population = {population}, Area = {area_in_km2 : .2f} km2")
        


    def plot_all(self):
        fig, axs = plt.subplots(3, 2, figsize=(20, 15))
        keys = ["height", "fertility", "forest", "humidity", "husbandry" ,"habitability"]
        for ax, key in zip(axs.flatten(), keys):
            self.plot_map(ax, key,title=key)
        plt.tight_layout()
        plt.show()

    def plot(self, map_type = "height", title = None):
        fig, ax = plt.subplots(figsize=(12, 12))
        self.plot_map(ax, map_type, title)
        plt.show()
        self.print_info()
    def plot_map(self, ax, map_type = "height", title = None):
        this_map = self.maps[map_type].cpu().numpy()

        cmap = self.type_to_cmap.get(map_type, "viridis")
        vmin, vmax = 0, 1
        if map_type == "height":
            vmin, vmax = -0.2, 1

        #add colorbar per axis
        ax.imshow(this_map, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(ax.imshow(this_map, cmap=cmap, vmin=vmin, vmax=vmax), ax=ax)
        ax.set_title(title if title else f"{map_type.capitalize()} Map")

        overlay = get_overlay(this_map.shape, color=(0.0, 0.0, 0.0))
        overlay = self.plot_sea_and_river_overlay(overlay)
        overlay = self.plot_cities(overlay)
        if self.cities is not None:
            for city in self.cities:
                ax.scatter(city.pos[1], city.pos[0], c='white',s=25, edgecolors = 'black')

        overlay = self.plot_roads(overlay)
        ax.imshow(overlay)
        if map_type == "height":
            self.plot_forest(ax)

    def plot_sea_and_river_overlay(self, overlay):
        sea_mask = self.sea_mask
        overlay = mask_overlay(overlay, sea_mask, color=self.colors["sea"], alpha=0.9)  # Semi-transparent blue for sea and river
        return overlay

    def plot_cities(self, overlay):
        city = self.maps["city"].cpu().numpy()
        overlay = plot_rural_farmland_overlay(city, overlay, color=self.colors["fertility"])
        overlay = plot_rural_forest_overlay(city, overlay, color=self.colors["forest"])
        overlay = plot_urban_overlay(city, overlay, color=self.colors["urban"])
        #overlay = plot_city_edges_overlay(city, overlay, color=self.colors["edge"])
        return overlay
    def plot_forest(self, ax):
        forest = self.maps["forest"].cpu().numpy().copy()

        city_mask = self.maps["city"].cpu().numpy()[:, :, 0] != 0
        sea_mask = self.maps["sea"].cpu().numpy()
        river_mask = self.maps["river"].cpu().numpy()
        road_mask = self.maps["road"].cpu().numpy()
        mask = city_mask | sea_mask | river_mask | road_mask
        forest[mask] = 0  # Remove forest where cities, rivers, seas, or roads are present
        ax.imshow(forest, cmap="Greens", alpha=1.0 * forest)  # Green color for forest with alpha based on density
    def plot_roads(self, overlay):
        road_mask = self.maps["road"].cpu().numpy()
        H, W = road_mask.shape
        overlay = mask_overlay(overlay, road_mask, color=self.colors["road"][:3], alpha=self.colors["road"][3])  # Grey color for roads
        return overlay