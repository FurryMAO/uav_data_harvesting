from src.PG.Agent import PGAgent
import tensorflow as tf
import numpy as np
import tqdm


class PGTrainerParams:
    def __init__(self):
        self.num_steps = 1e6
        self.rm_pre_fill_ratio = 0.5
        self.rm_pre_fill_random = True
        self.eval_period = 5
        self.rm_size = 50000
        self.load_model = ""


class PGTrainer:
    def __init__(self, params: PGTrainerParams, agent: PGAgent):
        self.params = params
        self.agent = agent
        self.use_scalar_input = self.agent.params.use_scalar_input #false
        self.stop=False

        self.b_list=[]
        self.f_list=[]
        self.s_list=[]
        self.a_list=[]
        self.r_list=[]
        self.nb_list=[]
        self.nf_list=[]
        self.ns_list=[]
        self.nt_list=[]



    def add_experience(self, state, action, reward, next_state):
        if isinstance(action, int):
            action=np.array([action])
        if self.use_scalar_input:
            self.store((state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      state.get_scalars(give_position=True),
                                      action,
                                      reward,
                                      next_state.get_device_scalars(self.agent.params.max_devices, self.agent.params.relative_scalars),
                                      next_state.get_uav_scalars(self.agent.params.max_uavs, self.agent.params.relative_scalars),
                                      next_state.get_scalars(give_position=True),
                                      next_state.terminal))
        else:
            self.store((state.get_boolean_map(),
                                      state.get_float_map(),
                                      state.get_scalars(),
                                      action,
                                      reward,
                                      next_state.get_boolean_map(),
                                      next_state.get_float_map(),
                                      next_state.get_scalars(),
                                      next_state.terminal))
            if next_state.terminal==True:
                self.stop=True


    def store(self, experiences):
        #----------------------------------------#

        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars =experiences[2]
        # current_state=[boolean_map, float_map, scalars]
        action =experiences[3]
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars =experiences[7]
        terminated = experiences[8]


        self.b_list.append(boolean_map)
        self.f_list.append(float_map)
        self.s_list.append(scalars)
        self.a_list.append(action)
        self.r_list.append(reward)
        self.nb_list.append(next_boolean_map)
        self.nf_list.append(next_float_map)
        self.ns_list.append(next_scalars)
        self.nt_list.append(terminated)



    def train_agent(self):
        if not self.b_list:
            pass
        else:
            self.agent.train(self.b_list,self.f_list,self.s_list, self.a_list, self.r_list, self.nb_list, self.nf_list, self.ns_list, self.nt_list)
            self.b_list.clear()
            self.f_list.clear()
            self.s_list.clear()
            self.a_list.clear()
            self.r_list.clear()
            self.nb_list.clear()
            self.nf_list.clear()
            self.ns_list.clear()
            self.nt_list.clear()


    def should_fill_replay_memory(self):
        return False




