from src.AC.Agent import ACAgent
from src.AC.ReplayMemory import ReplayMemory
import tqdm


class ACTrainerParams:
    def __init__(self):
        self.batch_size = 128
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""


class ACTrainer:
    def __init__(self, params: ACTrainerParams, agent: ACAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size) # replaymemory ==> 50000
        self.agent = agent
        self.use_scalar_input = self.agent.params.use_scalar_input #false

        if self.params.load_model != "": #True
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state):
        if self.use_scalar_input:
            self.replay_memory.store((state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      state.get_scalars(give_position=True),
                                      action,
                                      reward,
                                      next_state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      next_state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      next_state.get_scalars(give_position=True),
                                      next_state.terminal))
        else:
            self.replay_memory.store((state.get_boolean_map(),
                                      state.get_float_map(),
                                      state.get_scalars(),
                                      action,
                                      reward,
                                      next_state.get_boolean_map(),
                                      next_state.get_float_map(),
                                      next_state.get_scalars(),
                                      next_state.terminal))

    def train_agent(self):
        mini_batch = self.replay_memory.sample()
        self.agent.train(mini_batch)
        self.replay_memory.reset()


    def should_fill_replay_memory(self):
        return False


