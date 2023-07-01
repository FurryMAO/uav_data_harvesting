import numpy as np
import os
import tqdm
from src.Map.Map import load_map


def bresenham(x0, y0, x1, y1, obstacles, shadow_map):
    if obstacles[y0, x0]:
        return
    x_dist = abs(x0 - x1)
    y_dist = -abs(y0 - y1)
    x_step = 1 if x1 > x0 else -1
    y_step = 1 if y1 > y0 else -1

    error = x_dist + y_dist

    # shadowed = False
    shadow_map[y0, x0] = False

    while x0 != x1 or y0 != y1:
        if 2 * error - y_dist > x_dist - 2 * error:
            # horizontal step
            error += y_dist
            x0 += x_step
        else:
            # vertical step
            error += x_dist
            y0 += y_step

        if obstacles[y0, x0]:
            # shadowed = True
            return

        # if shadowed:
        shadow_map[y0, x0] = False
'''这段代码实现了 Bresenham 算法用于计算从 (x0, y0) 到 (x1, y1) 两点之间的直线路径，并更新障碍物和阴影地图。

具体步骤如下：

首先判断起始点 (x0, y0) 是否为障碍物，如果是障碍物则直接返回。
计算 x 轴和 y 轴的距离差值：x_dist = abs(x0 - x1)，y_dist = -abs(y0 - y1)。
根据目标点相对于起始点的位置关系，确定 x 轴和 y 轴的递增方向：x_step = 1 if x1 > x0 else -1，y_step = 1 if y1 > y0 else -1。
初始化误差项 error = x_dist + y_dist。
将起始点标记为非阴影区域：shadow_map[y0, x0] = False。
进入循环，直到当前点 (x0, y0) 到达目标点 (x1, y1)。
在每次循环中，根据误差项的值判断是进行水平步进还是垂直步进。
如果 2 * error - y_dist > x_dist - 2 * error，则进行水平步进。更新误差项和 x 坐标。
否则，进行垂直步进。更新误差项和 y 坐标。
在每次循环中，检查当前点是否为障碍物。如果是障碍物，则直接返回。
循环结束后，表示从 (x0, y0) 到 (x1, y1) 的直线路径已经计算完毕，并且没有遇到障碍物。'''





def calculate_shadowing(map_path, save_as):
    total_map = load_map(map_path)
    obstacles = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size * size

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    with tqdm.tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            shadow_map = np.ones((size, size), dtype=bool)

            for x in range(size):
                bresenham(i, j, x, 0, obstacles, shadow_map)
                bresenham(i, j, x, size - 1, obstacles, shadow_map)
                bresenham(i, j, 0, x, obstacles, shadow_map)
                bresenham(i, j, size - 1, x, obstacles, shadow_map)

            total_shadow_map[j, i] = shadow_map
            pbar.update(1)

    np.save(save_as, total_shadow_map)
    return total_shadow_map


def load_or_create_shadowing(map_path):
    shadow_file_name = os.path.splitext(map_path)[0] + "_shadowing.npy"
    if os.path.exists(shadow_file_name):
        return np.load(shadow_file_name)
    else:
        return calculate_shadowing(map_path, shadow_file_name)
