import os

import numpy as np

from src.IoTDevice import JammerDeviceParams, DeviceList, JammerList

ColorMap = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


class JammerManagerParams:
    def __init__(self):
        self.device_count_range = (2, 5)
        self.power_range = (5.0, 20.0)
        self.fixed_jammers = False


class JammerManager:
    """
    The DeviceManager is used to generate DeviceLists according to DeviceManagerParams
    """

    def __init__(self, params: JammerManagerParams):
        self.params = params #the params is the grid_parameter---->device_manager

    def generate_jammer_list(self, positions_vector): #the psiition_vector is the free_space but if we set the fisxed_device, then don't need it
        if self.params.fixed_jammers:
            return JammerList(self.params.devices)

        else:
            # Roll number of devices
            #device_count = np.random.randint(self.params.device_count_range[0], self.params.device_count_range[1] + 1)
            jammer_count =np.random.randint(self.params.jammer_count_range[0], self.params.jammer_count_range[1] + 1)
            # Roll Positions
            position_idcs = np.random.choice(range(len(positions_vector)), jammer_count, replace=False)
            positions = [positions_vector[idx] for idx in position_idcs]

            # Roll Data
            power = np.random.uniform(self.params.power_range[0], self.params.power_range[1], jammer_count)

            return self.generate_jammer_list_from_args(jammer_count, positions, power)

    def generate_jammer_list_from_args(self, jammer_count, positions, power):

        params = [JammerDeviceParams(position=positions[k],
                                  power=power[k])
                  for k in range(jammer_count)]

        return JammerList(params)
