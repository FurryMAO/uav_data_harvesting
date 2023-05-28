import copy
import tqdm
import distutils.util
from src.ModelStats import ModelStatsParams, ModelStats
from src.base.BaseDisplay import BaseDisplay


class BaseEnvironmentParams:
    def __init__(self):
        self.model_stats_params = ModelStatsParams()


class BaseEnvironment:
    def __init__(self, params: BaseEnvironmentParams, display: BaseDisplay):
        self.stats = ModelStats(params.model_stats_params, display=display)
        self.trainer = None
        self.grid = None
        self.rewards = None
        self.physics = None
        self.display = display
        self.episode_count = 0
        self.step_count = 0
        self.count_in_episode=0

    def fill_replay_memory(self):
        while self.trainer.should_fill_replay_memory(): #True
            state = copy.deepcopy(self.init_episode())
            while not state.terminal:
                if self.flag == 2:
                    next_state = self.step(state, random=self.trainer.params.rm_pre_fill_random) #产生的动作是随机的
                if self.flag==1:
                    next_state = self.step_on(state, random=self.trainer.params.rm_pre_fill_random)
                state = copy.deepcopy(next_state)


    def train_episode(self):
        state = copy.deepcopy(self.init_episode())
        self.stats.on_episode_begin(self.episode_count)
        if self.flag==2: # on policy 走一步更新一次
            while not state.is_terminal():
                #self.count_in_episode=self.count_in_episode+1
                state = self.step(state)
                self.trainer.train_agent()

            # print('This is the:',self.episode_count,'episode')
            # print('There are:',self.count_in_episode,'steps')
            # self.count_in_episode=0
            self.episode_count += 1
            self.stats.on_episode_end(self.episode_count)
            self.stats.log_training_data(step=self.step_count)

        if self.flag==1: # on policy 走一把更新一次
            while not state.is_terminal():
                #self.count_in_episode=self.count_in_episode+1
                state = self.step(state)
            self.trainer.train_agent()
            # print('This is the:',self.episode_count,'episode')
            # print('There are:',self.count_in_episode,'steps')
            # self.count_in_episode=0
            self.episode_count += 1
            self.stats.on_episode_end(self.episode_count)
            self.stats.log_training_data(step=self.step_count)

    def run(self):
        #self.fill_replay_memory()
        print('Running ', self.stats.params.log_file_name)
        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()

            if self.episode_count % self.trainer.params.eval_period == 0:
                self.test_episode()
            #self.stats.add_log_data_callback('actor_loss', self.loss)

            self.stats.save_if_best()

        self.stats.training_ended()

    def step(self, state, random=False):
        pass

    def step_on(self, state, random=False):
        pass

    def init_episode(self, init_state=None):
        if init_state:
            state = copy.deepcopy(self.grid.init_scenario(init_state))
        else:
            state = copy.deepcopy(self.grid.init_episode())

        self.rewards.reset()
        self.physics.reset(state)
        return state

    def test_episode(self):
        pass

    def test_scenario(self, scenario):
        pass

    def eval(self, episodes, show=False):
        for _ in tqdm.tqdm(range(episodes)):
            self.test_episode()
            self.step_count += 1  # Increase step count so that logging works properly

            if show:
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

                resp = input('Save run? [y/N]\n')
                try:
                    if distutils.util.strtobool(resp):
                        save_as = input('Save as: [run_' + str(self.step_count) + ']\n')
                        if save_as == '':
                            save_as = 'run_' + str(self.step_count)
                        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                                     save_path=save_as + '.png')
                        self.stats.save_episode(save_as)
                        print("Saved as run_" + str(self.step_count))
                except ValueError:
                    pass
                print("next then")

    def eval_scenario(self, init_state):
        self.test_scenario(init_state)

        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=True)

        resp = input('Save run? [y/N]\n')
        try:
            if distutils.util.strtobool(resp):
                save_as = input('Save as: [scenario]\n')
                if save_as == '':
                    save_as = 'scenario'
                self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=False,
                                             save_path=save_as + '.png')
                self.stats.save_episode(save_as)
                print("Saved as", save_as)
        except ValueError:
            pass