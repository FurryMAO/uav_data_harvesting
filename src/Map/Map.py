import numpy as np
from skimage import io
from scipy.ndimage import binary_dilation

class Map:
    def __init__(self, map_data):
        self.start_landing_zone = map_data[:, :, 2].astype(bool) # blue get the start and landing zone
        self.nfz_ = map_data[:, :, 0].astype(bool) # red get the no-fly zone
        self.obstacles = map_data[:, :, 1].astype(bool) #Green get the building blocking wireless links areas
        selem = np.ones((3, 3), dtype=bool)
        self.nfz= binary_dilation(self.nfz_, selem) #将不能飞的地方向外膨胀一个单位，防止无人机撞击大楼
        self.obstacles_= binary_dilation(self.obstacles , selem)


    def get_starting_vector(self):
        similar = np.where(self.start_landing_zone)
        return list(zip(similar[1], similar[0]))

    def get_free_space_vector(self):
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_landing_zone))
        free_idcs = np.where(free_space)
        return list(zip(free_idcs[1], free_idcs[0]))

    def get_size(self):
        return self.start_landing_zone.shape[:2]


def load_image(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)


def save_image(path, image):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_map(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=False)
    return Map(data)


def load_target(path):
    if type(path) is not str:
        raise TypeError('path needs to be a string')
    data = io.imread(path, as_gray=True)
    return np.array(data, dtype=bool)
