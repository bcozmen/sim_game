import numpy as np
import time

index_to_preference = {
    0 : "fertility",
    1 : "forest"
}

preference_p = np.asarray([0.75, 0.25])
land_type_p = np.asarray([0.25, 0.75])



def grow_random_all(world_map):
    cities = world_map.cities
    config = np.zeros((len(cities), 3), dtype=np.float32)
    for i, city in enumerate(cities):
        config[i, 1] = np.random.choice([0, 1], p=preference_p)     # preference index
        config[i, 2] = np.random.choice([0, 1], p=land_type_p)     # land type
        config[i, 0] = random_between(config[i, 2], index_to_preference[config[i, 1]])  # amount
    world_map.grow_all(config)

def grow(world_map, index, land_type, preference):
    rnd = random_between(land_type, preference)
    world_map.cities[index].grow(amount = rnd, land_type = land_type, preference = preference)


def grow_random(world_map):
    num_citiess = len(world_map.cities)
    cities_p = np.array([(np.random.rand()*1.5) + 1 for city in world_map.cities], dtype=np.float32)
    cities_p /= cities_p.sum()
    city_index = np.random.choice(num_citiess, p=cities_p)
    land_type = np.random.choice([0, 1], p=land_type_p)
    preference = np.random.choice([0, 1], p=preference_p)
    grow(world_map, city_index, land_type, index_to_preference[preference])
def random_between(land_type, preference):
    if land_type == 0:
        if preference == "fertility":
            a, b = 300, 1000
        else:
            a, b = 200, 800
    else:
        a, b = 30, 150
    return int(a + (b - a) * np.random.rand())




st = None
def tstart():
    global st
    st = time.time()
def tend():
    global st
    if st is not None:
        print(f"Time taken: {time.time() - st:.2f} seconds")
        st = None