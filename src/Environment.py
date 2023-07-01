import copy
import numpy as np
import random
from src.DDQN.Agent import DDQNAgentParams, DDQNAgent
from src.DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from src.PG.Agent import PGAgent, PGAgentParams
from src.PG.Trainer import PGTrainerParams, PGTrainer
from src.PPO.Agent import PPOAgent, PPOAgentParams
from src.PPO.Trainer import PPOTrainerParams, PPOTrainer
from src.AC.Agent import ACAgent, ACAgentParams
from src.AC.Trainer import ACTrainerParams, ACTrainer
from src.Display import DHDisplay
from src.Grid import GridParams, Grid
from src.Physics import PhysicsParams, Physics
from src.Rewards import RewardParams, Rewards
from src.State import State
from src.base.Environment import BaseEnvironment, BaseEnvironmentParams
from src.base.GridActions import GridActions


class EnvironmentParams(BaseEnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = GridParams()
        self.reward_params = RewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = PhysicsParams()


class Environment(BaseEnvironment):
    def __init__(self, params: EnvironmentParams):
        self.display = DHDisplay()
        super().__init__(params, self.display)

        self.grid = Grid(params.grid_params, stats=self.stats)
        self.rewards = Rewards(params.reward_params, stats=self.stats)
        self.physics = Physics(params=params.physics_params, stats=self.stats)
        #-------------自定义变量-----------------
        self.algorithm_select= params.algorithm_params.__dict__
        self.flag=0
        #配置agent 的信息
        if self.algorithm_select['Policy_Gradient']==True:
            self.agent = PGAgent(params.agent_params, self.grid.get_example_state(),
                                   self.physics.get_example_action(),stats=self.stats) #PGAGRNT( params,
            self.trainer = PGTrainer(params.trainer_params, agent=self.agent)

        elif self.algorithm_select['DDQN']==True:
            self.agent = DDQNAgent(params.agent_params, self.grid.get_example_state(),
                                   self.physics.get_example_action(), stats=self.stats)  # DDQNAGRNT( params,
            self.trainer = DDQNTrainer(params.trainer_params, agent=self.agent)
            self.flag = 1

        elif self.algorithm_select['AC']==True:
            self.agent = ACAgent(params.agent_params, self.grid.get_example_state(),
                                 self.physics.get_example_action(), stats=self.stats)  # PGAGRNT( params,
            self.trainer = ACTrainer(params.trainer_params, agent=self.agent)

        elif self.algorithm_select['PPO'] == True:
            self.agent = PPOAgent(params.agent_params, self.grid.get_example_state(),
                                 self.physics.get_example_action(), stats=self.stats)  # PGAGRNT( params,
            self.trainer = PPOTrainer(params.trainer_params, agent=self.agent)
            self.flag = 2


        self.display.set_channel(self.physics.channel)
        self.first_action = True
        self.last_actions = []
        self.last_rewards = []
        self.last_states = []

    def test_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        first_action = True
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                action = self.agent.get_exploitation_action_target(state)

                #self.loss=self.agent.aloss
                if not first_action:
                    reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                           GridActions(self.last_actions[state.active_agent]), state)
                    self.stats.add_experience(
                        (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                         copy.deepcopy(state)))

                self.last_states[state.active_agent] = copy.deepcopy(state)
                self.last_actions[state.active_agent] = action
                state = self.physics.step(GridActions(action))

                if state.terminal:
                    reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                           GridActions(self.last_actions[state.active_agent]), state)
                    self.stats.add_experience(
                        (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                         copy.deepcopy(state)))
            first_action = False

        self.stats.on_episode_end(self.episode_count)
        self.stats.log_testing_data(step=self.step_count)

    def test_scenario(self, scenario):
        state = copy.deepcopy(self.init_episode(scenario))
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                action = self.agent.get_exploitation_action_target(state)
                state = self.physics.step(GridActions(action))



    def step(self, state: State, is_random=False): #update the action, reward, next state 向下存储一部 保存信息
        for state.active_agent in range(state.num_agents):
            if state.terminal:
                continue
            if is_random:
                action = self.agent.get_random_action()
            ## episollion greddy
            # else:
            #     if np.random.uniform() < self.eps:
            #         action = self.agent.get_random_action()
            #     else:
            #         # choose the best action according to current Q table
            #         action = self.agent.act(state)
            else:
                action = self.agent.act(state)

            if not self.first_action:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                     copy.deepcopy(state)))

            self.last_states[state.active_agent] = copy.deepcopy(state)
            self.last_actions[state.active_agent] = action
            state = self.physics.step(GridActions(action))
            if state.terminal:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                     copy.deepcopy(state)))

        self.step_count += 1
        self.first_action = False
        return state


    def step_ppo(self, state: State, is_random=False):  # update the action, reward, next state 向下存储一部 保存信息
        for state.active_agent in range(state.num_agents):
            if state.terminal:
                continue
            if is_random:
                action = self.agent.get_random_action()
            ## episollion greddy
            # else:
            #     if np.random.uniform() < self.eps:
            #         action = self.agent.get_random_action()
            #     else:
            #         # choose the best action according to current Q table
            #         action = self.agent.act(state)
            else:
                action = self.agent.act(state)
                # if random.random() < 0.1:
                #     action=self.agent.get_random_action()
                # else:
                #     action = self.agent.act(state)

            prob, value = self.agent.get_old_possiblility_value(state, action)
            if not self.first_action:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state, prob, value)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                     copy.deepcopy(state)))

            self.last_states[state.active_agent] = copy.deepcopy(state)
            self.last_actions[state.active_agent] = action
            state = self.physics.step(GridActions(action))
            if state.terminal:
                reward = self.rewards.calculate_reward(self.last_states[state.active_agent],
                                                       GridActions(self.last_actions[state.active_agent]), state)
                self.trainer.add_experience(self.last_states[state.active_agent], self.last_actions[state.active_agent],
                                            reward, state, prob, value)
                self.stats.add_experience(
                    (self.last_states[state.active_agent], self.last_actions[state.active_agent], reward,
                     copy.deepcopy(state)))

        self.step_count += 1
        self.first_action = False
        return state




    def init_episode(self, init_state=None):
        state = super().init_episode(init_state)
        self.last_states = [None] * state.num_agents
        self.last_actions = [None] * state.num_agents
        self.first_action = True
        return state
