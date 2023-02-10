from src.ModelStats import ModelStats
import src.Map.Map as Map
#get the map and some information

class BaseGridParams:
    def __init__(self):
        self.movement_range = (100, 200)
        self.map_path = 'res/newmap.png'


class BaseGrid:
    def __init__(self, params: BaseGridParams, stats: ModelStats):
        self.map_image = Map.load_map(params.map_path)
        self.shape = self.map_image.landing_zone.shape #get the start and land zone shape
        self.starting_vector = self.map_image.get_starting_vector() #get the points in the landing zone
        stats.set_env_map_callback(self.get_map_image)

    def get_map_image(self):
        return self.map_image

    def get_grid_size(self):
        return self.shape

    def get_no_fly(self):
        return self.map_image.nfz

    def get_landing_zone(self):
        return self.map_image.landing_zone
