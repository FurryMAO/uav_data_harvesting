from src.PPO.Agent import PPOAgent
from src.PPO.ReplayMemory import ReplayMemory
import numpy as np
import tqdm


class PPOTrainerParams:
    def __init__(self):
        self.batch_size = 2**9
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""


class PPOTrainer:
    def __init__(self, params: PPOTrainerParams, agent: PPOAgent):
        self.params = params
        self.replay_memory = ReplayMemory(size=params.rm_size) # replaymemory ==> 50000
        self.agent = agent
        self.use_scalar_input = self.agent.params.use_scalar_input #false
        self.observation_buffer=[]
        self.action_buffer=[]
        self.advantage_buffer=[]
        self.return_buffer=[]
        self.logprobability_buffer=[]


        if self.params.load_model != "": #True
            print("Loading model", self.params.load_model, "for agent")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None

    def add_experience(self, state, action, reward, next_state, old_action_prob):
        self.replay_memory.store((state.get_boolean_map(),
                                  state.get_float_map(),
                                  state.get_scalars(),
                                  action,
                                  reward,
                                  next_state.get_boolean_map(),
                                  next_state.get_float_map(),
                                  next_state.get_scalars(),
                                  next_state.terminal,
                                  old_action_prob))

    def train_agent(self):
        if self.params.batch_size > self.replay_memory.get_size():
            return
        mini_batch = self.replay_memory.sample(self.params.batch_size)

        self.agent.train(mini_batch)

    def should_fill_replay_memory(self):
        target_size = self.replay_memory.get_max_size() * self.params.rm_pre_fill_ratio
        if self.replay_memory.get_size() >= target_size or self.replay_memory.full:
            if self.prefill_bar:
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None:
            print("Filling replay memory")
            self.prefill_bar = tqdm.tqdm(total=target_size)
        self.prefill_bar.update(self.replay_memory.get_size() - self.prefill_bar.n)

        return True

