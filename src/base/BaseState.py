from src.Map.Map import Map

#get the base information of the map including no fly place, landing place and obstacles
class BaseState:
    def __init__(self, map_init: Map):
        self.no_fly_zone = map_init.nfz
        self.obstacles = map_init.obstacles
        self.landing_zone = map_init.landing_zone

    @property
    def shape(self):
        return self.landing_zone.shape[:2] #get the 0 to 2 which is the width and height of the ladning zone
