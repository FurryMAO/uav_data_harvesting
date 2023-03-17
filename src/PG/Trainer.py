from src.PG.Agent import PGAgent
from src.PG.TrackStorage import TrackStorage
import tensorflow as tf
import numpy as np
import tqdm


class PGTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""
        self.reward=[]


class PGTrainer:
    def __init__(self, params: PGTrainerParams, agent: PGAgent):
        self.params = params
        self.history_storage = TrackStorage(size=params.rm_size)
        self.agent = agent
        self.use_scalar_input = self.agent.params.use_scalar_input #false


    def add_experience(self, state, action, reward, next_state):
        if self.use_scalar_input:
            self.history_storage.store((state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      state.get_scalars(give_position=True),
                                      action,
                                      reward,
                                      next_state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      next_state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      next_state.get_scalars(give_position=True),
                                      next_state.terminal))
        else:
            self.history_storage.store((state.get_boolean_map(),
                                      state.get_float_map(),
                                      state.get_scalars(),
                                      action,
                                      reward,
                                      next_state.get_boolean_map(),
                                      next_state.get_float_map(),
                                      next_state.get_scalars(),
                                      next_state.terminal))


    def train_agent(self,rewards):
        self.agent.train(rewards)







