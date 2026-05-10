from abc import ABC, abstractmethod



class BuildingManager(ABC):
    def __init__(self, maps, building_id):
        self.maps = maps
        self.building_id = building_id #int
        

    def add(self,pos, city_id, owner_id):
        self.maps["city"][pos] = (city_id, owner_id, self.building_id)

    def get_mask(self):
        return self.maps["city"][..., 2] == self.building_id

    @abstractmethod
    def requires(self):
        pass
    
    @abstractmethod
    def produces(self):
        pass