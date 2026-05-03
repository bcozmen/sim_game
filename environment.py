import numpy as np
from world_map import WorldMap

class Environment:
    def __init__(self, seed=43, num_lands=8):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.day = 0
        self.world = WorldMap(size=2**8+1, num_rivers=8, sea_level=0.18)
        #self.world.plot_2d("terrain")
        #self.world.plot_2d("fertility")
        #self.world.plot_2d("forest")

        self.lands = []