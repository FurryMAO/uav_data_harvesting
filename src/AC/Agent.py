
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
from src.AC.AC_Network import Critic, Actor
import numpy as np

def print_node(x):
    print(x)
    return x


class ACAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.global_map_scaling = 3
        self.local_map_size = 17

        # Scalar inputs instead of map
        self.use_scalar_input = False
        self.relative_scalars = False
        self.blind_agent = False
        self.max_uavs = 3
        self.max_devices = 10

        # Printing
        self.print_summary = False



class ACAgent(object):

    def __init__(self, params: ACAgentParams, example_state, example_action, stats=None):

        self.params = params
        self.gamma = tf.constant(self.params.gamma, dtype=float)
        self.boolean_map_shape = example_state.get_boolean_map_shape() #get the environment map shape
        self.float_map_shape = example_state.get_float_map_shape() # get the device data map
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input) #get the movement budget size
        self.num_actions = len(type(example_action)) # 1
        self.epsilon=0.1
        self.stats=stats
        self.aloss=0
        self.closs=0

        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,
                  float_map_input,
                  scalars_input]
        self.input=self.get_network_input(states)

        self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                      outputs=self.global_map)
        self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.local_map)
        self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.total_map)

        self.critic_network = Critic()
        # self.critic_network.build((None, self.state_dim))
        self.actor_network = Actor()
        # self.actor_network.build((None, self.state_dim))
        self.critic_network(self.input)
        self.actor_network(self.input)

        self.a_opt = tf.optimizers.Adam(learning_rate=params.learning_rate, clipvalue=1.0)
        self.c_opt=tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.critic_network.summary()
            self.actor_network.summary()

        if stats:
            stats.set_model(self.actor_network)
            stats.set_model(self.critic_network)

    def get_network_input(self, inputs):
        boolean_map_input=inputs[0]
        float_map_input=inputs[1]
        scalars_input=inputs[2]
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)  # 张量数据类型转换
        padded_map = tf.concat([map_cast, float_map_input], axis=-1)  # tensors combine in one dimension
        states_proc=scalars_input
        map_proc=padded_map
        flatten_map = self.create_map_proc(map_proc, name='pre') #return the flatten local and environment map
        layer = tf.concat([flatten_map, states_proc], axis=1)
        return layer


    def create_map_proc(self, conv_in, name): #parameter is the total map
        # Forking for global and local map
        # Global Map
        global_map = tf.stop_gradient(
            AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))
        self.global_map = global_map # is the total map been poolled
        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                strides=(1, 1),
                                name=name + 'global_conv_' + str(k + 1))(global_map) #pass two conv layers

        flatten_global = Flatten(name=name + 'global_flatten')(global_map) #flat the whole map

        # Local Map
        crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        self.local_map = local_map

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        flatten_local = Flatten(name=name + 'local_flatten')(local_map)

        return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])

    def get_exploitation_action_target(self, state):
        return self.get_action(state)

    def act(self, state):
        return self.get_action(state)

    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    # def get_exploitation_action(self, state): # get the maximum value action
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
    #     return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_action(self, state): # get the action base on the possibility
        #action is a array with dimension(1,)
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        network_input = self.get_network_input([boolean_map_in, float_map_in, scalars])
        prob_value = self.actor_network(network_input).numpy()[0]
        #print(prob_value)
        try:
            action = np.random.choice(range(self.num_actions), size=1, p=prob_value)
        except ValueError:
            print('Invalid probabilities. Choosing a random action.')

        return action

    # def get_action_probossiblity(self, prob_value): # get the action base on the possibility
    #     softmax_scaling = tf.divide(prob_value, tf.constant(self.params.soft_max_scaling, dtype=float))
    #     softmax_action = tf.math.softmax(softmax_scaling)
    #     p = tf.reshape(softmax_action, [-1]).numpy()
    #     return p

    def get_exploitation_action_target(self, state):
        return self.get_action(state)


    # def actor_loss(self, prob, action, td):
    #     entropy_weight = 0.001
    #     dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    #     log_prob = dist.log_prob(action)
    #     entropy = dist.entropy()
    #     loss = -(log_prob * td + entropy_weight * entropy)
    #     return loss

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * td
        return loss

    def train(self, experiences):
        boolean_map = experiences[0]
        #print("a=",boolean_map.shape)
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        # 8*********************************************************8
        boolean_map=self.get_unit_data(boolean_map)
        float_map=self.get_unit_data(float_map)
        scalars = self.get_unit_data(scalars)
        action = self.get_unit_data(action)
        reward = self.get_unit_data(reward)
        next_boolean_map = self.get_unit_data(next_boolean_map)
        next_float_map = self.get_unit_data(next_float_map)
        next_scalars = self.get_unit_data(next_scalars)
        terminated = self.get_unit_data(terminated)

        # print('second chect')
        # print(boolean_map.shape)
        # print(float_map.shape)
        # print(scalars.shape)

        input_current=self.get_network_input([boolean_map, float_map, scalars])
        input_next=self.get_network_input([next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p= self.actor_network(input_current, training=True)
            v = self.critic_network(input_current, training=True)
            v_ = self.critic_network(input_next, training=True)
            td = reward + self.gamma * v_ * (1 - np.asarray(terminated, dtype=np.int32)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td ** 2
        grads1 = tape1.gradient(a_loss, self.actor_network.trainable_variables)
        #tf.print("Gradients:", grads1)
        grads2 = tape2.gradient(c_loss, self.critic_network.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor_network.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic_network.trainable_variables))
        self.aloss=a_loss
        self.closs=c_loss


        # logdir = "logs"
        # summary_writer = tf.summary.create_file_writer(logdir)
        # with summary_writer.as_default():
        #     tf.summary.scalar("Actor Loss", a_loss, step=self.i)
        #     tf.summary.scalar("Critic Loss", c_loss, step=self.i)
        #     self.i=self.i+1



    def save_model(self, path_to_model):
        self.actor_network.save(path_to_model)
        self.critic_network.save(path_to_model)

    def get_unit_data(self, input_data):
        return np.expand_dims(input_data, axis=0)

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()


################################################################# second version  #################3
# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
# import numpy as np
#
# def print_node(x):
#     print(x)
#     return x
#
#
# class ACAgentParams:
#     def __init__(self):
#         # Convolutional part config
#         self.conv_layers = 2
#         self.conv_kernel_size = 5
#         self.conv_kernels = 16
#
#         # Fully Connected config
#         self.hidden_layer_size = 256
#         self.hidden_layer_num = 3
#
#         # Training Params
#         self.learning_rate = 3e-5
#         self.alpha = 0.005
#         self.gamma = 0.95
#
#         # Exploration strategy
#         self.soft_max_scaling = 0.1
#
#         # Global-Local Map
#         self.global_map_scaling = 3
#         self.local_map_size = 17
#
#         # Scalar inputs instead of map
#         self.use_scalar_input = False
#         self.relative_scalars = False
#         self.blind_agent = False
#         self.max_uavs = 3
#         self.max_devices = 10
#
#         # Printing
#         self.print_summary = False
#
#
# class ACAgent(object):
#
#     def __init__(self, params: ACAgentParams, example_state, example_action, stats=None):
#
#         self.params = params
#         self.gamma = tf.constant(self.params.gamma, dtype=float)
#         self.boolean_map_shape = example_state.get_boolean_map_shape() #get the environment map shape
#         self.float_map_shape = example_state.get_float_map_shape() # get the device data map
#         self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input) #get the movement budget size
#         self.num_actions = len(type(example_action)) # 1
#         #-----------------self define variable-----------------
#         self.step_count=0
#         self.epsilon=0.5
#         # Create shared inputs
#         action_input = Input(shape=(), name='action_input', dtype=tf.int64)
#         reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
#         termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
#         TD_target = Input(shape=(), name='score_input', dtype=tf.float32)
#
#         boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
#         float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
#         scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
#         states = [boolean_map_input,
#                   float_map_input,
#                   scalars_input]
#
#         map_cast = tf.cast(boolean_map_input, dtype=tf.float32) #张量数据类型转换
#         #print(map_cast.shape)
#         #print(float_map_input.shape)
#         padded_map = tf.concat([map_cast, float_map_input], axis=3) # tensors combine in one dimension
#
#         self.actor_network = self.build_actor_model(padded_map, scalars_input, states,'actor')
#
#         self.critic_network=self.build_critic_model(padded_map, scalars_input, states,'critic')
#
#         self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
#                                       outputs=self.global_map)
#         self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
#                                      outputs=self.local_map)
#         self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
#                                      outputs=self.total_map)
#
#         actor_pro_pre = self.actor_network.output
#         #actor_pro_pre = tf.where(tf.math.is_nan(actor_pro_pre), tf.zeros_like(actor_pro_pre), actor_pro_pre)
#
#
#         q_value=self.critic_network.output
#         self.get_score_model=Model(inputs=states, outputs=q_value)
#
#
#         # Softmax explore model
#         softmax_scaling = tf.divide(actor_pro_pre, tf.constant(self.params.soft_max_scaling, dtype=float))
#         softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
#         self.soft_explore_model = Model(inputs=states, outputs=softmax_action)
#
#
#         # Exploit act model
#         max_action = tf.argmax(softmax_action, axis=1, name='max_action', output_type=tf.int64)
#         self.exploit_model = Model(inputs=states, outputs=max_action)
#         self.exploit_model_target = self.exploit_model
#
#
#         # max_action = tf.argmax(softmax_action, axis=1, name='max_action', output_type=tf.int64)
#         # max_action_target = tf.argmax(softmax_action_target, axis=1, name='max_action', output_type=tf.int64)
#         # one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)
#         # q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, softmax_action_target, name='mul_hot_target'), axis=1,
#         #                        name='q_star')
#
#
#         #define actor network loss model
#         #a_loss = -tf.reduce_mean(TD_target * tf.math.log(softmax_action))
#         TD_target2=tf.expand_dims(TD_target, axis=1)
#         a_loss = -tf.reduce_mean(TD_target2 * tf.math.log(softmax_action + 1e-10))
#         self.a_loss_model= Model(
#             inputs=states + [action_input, reward_input, termination_input, TD_target],
#             outputs=a_loss)
#
#
#         self.actor_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)
#         self.critic_optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)
#
#         if self.params.print_summary:
#             self.a_loss_model.summary()
#
#
#         if stats:
#             stats.set_model(self.actor_network)
#             stats.set_model(self.critic_network)
#
#     def build_actor_model(self, map_proc, states_proc, inputs, name=''):
#         flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
#         layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
#         for k in range(self.params.hidden_layer_num):
#             layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
#                 layer)
#         output_a = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
#         model_a = Model(inputs=inputs, outputs=output_a)
#         return model_a
#
#     def build_critic_model(self, map_proc, states_proc, inputs, name=''):
#         flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
#         layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
#         for k in range(self.params.hidden_layer_num):
#             layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
#                 layer)
#         output_c = Dense(1, activation='linear', name=name + 'output_layer')(layer)
#         model_c = Model(inputs=inputs, outputs=output_c)
#         return model_c
#
#     def create_map_proc(self, conv_in, name): #parameter is the total map
#
#         # Forking for global and local map
#         # Global Map
#         global_map = tf.stop_gradient(
#             AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))
#
#         self.global_map = global_map # is the total map been poolled
#         self.total_map = conv_in
#
#         for k in range(self.params.conv_layers):
#             global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
#                                 strides=(1, 1),
#                                 name=name + 'global_conv_' + str(k + 1))(global_map) #pass two conv layers
#
#         flatten_global = Flatten(name=name + 'global_flatten')(global_map) #flat the whole map
#
#         # Local Map
#         crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
#         local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
#         self.local_map = local_map
#
#         for k in range(self.params.conv_layers):
#             local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
#                                strides=(1, 1),
#                                name=name + 'local_conv_' + str(k + 1))(local_map)
#
#         flatten_local = Flatten(name=name + 'local_flatten')(local_map)
#
#         return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])
#
#     def act(self, state): #get the achtion based on possibility
#         return self.get_soft_max_exploration(state)
#
#
#     def get_random_action(self):
#         return np.random.randint(0, self.num_actions)
#
#     def get_exploitation_action(self, state): # get the maximum value action
#
#         if self.params.blind_agent: #false
#             scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
#             return self.exploit_model(scalars).numpy()[0]
#
#         if self.params.use_scalar_input: #false
#             devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
#             uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
#             scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
#             return self.exploit_model([devices_in, uavs_in, scalars]).numpy()[0]
#
#         boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
#         float_map_in = state.get_float_map()[tf.newaxis, ...]
#         scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
#
#         return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
#
#     def get_soft_max_exploration(self, state): # given the state and get the action base on the possibility
#
#         if self.params.blind_agent:
#             scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
#             p = self.soft_explore_model(scalars).numpy()[0]
#         elif self.params.use_scalar_input:
#             devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
#             uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
#             scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
#             p = self.soft_explore_model([devices_in, uavs_in, scalars]).numpy()[0]
#         else:
#             boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
#             float_map_in = state.get_float_map()[tf.newaxis, ...]
#             scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
#             p = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
#         self.step_count = self.step_count + 1
#
#         if np.random.rand() < self.epsilon:
#             action=self.get_random_action()
#         else: action=
#
#         return action
#
#     def get_exploitation_action_target(self, state): # *********given state information and get the maximum value action for target netweork
#
#         if self.params.blind_agent:
#             scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
#             return self.exploit_model_target(scalars).numpy()[0]
#
#         if self.params.use_scalar_input:
#             devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
#             uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
#             scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
#             return self.exploit_model_target([devices_in, uavs_in, scalars]).numpy()[0]
#
#         boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
#         float_map_in = state.get_float_map()[tf.newaxis, ...]
#         scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
#
#         return self.exploit_model_target([boolean_map_in, float_map_in, scalars]).numpy()[0]
#
#     # def hard_update(self):
#     #     self.target_actor_network.set_weights(self.actor_network.get_weights())
#     #     self.target_critic_network.set_weights(self.critic_network.get_weights())
#     #
#     # def soft_update_a(self, alpha):
#     #     weights = self.actor_network.get_weights()
#     #     target_weights = self.target_actor_network.get_weights()
#     #     self.target_actor_network.set_weights(
#     #         [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights, target_weights)])
#     #
#     #
#     # def soft_update_c(self, alpha):
#     #     weights_ = self.critic_network.get_weights()
#     #     target_weights_ = self.target_critic_network.get_weights()
#     #     self.target_critic_network.set_weights(
#     #         [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights_, target_weights_)])
#
#     def train(self, experiences):
#         boolean_map = experiences[0]
#         #print("a=",boolean_map.shape)
#         float_map = experiences[1]
#         scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
#         action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
#         reward = experiences[4]
#         next_boolean_map = experiences[5]
#         next_float_map = experiences[6]
#         next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
#         terminated = experiences[8]
#
#         if terminated==True:
#             print("already finsh one game", self.step_count)
#             self.step_count=0
#
#         # before that the dimsension is [128,1]
#         boolean_map = self.get_unit_data(boolean_map)
#         # print("a=",boolean_map.shape)
#         float_map = self.get_unit_data(float_map)
#         scalars = self.get_unit_data(scalars)
#         action = self.get_unit_data(action)
#         reward = self.get_unit_data(reward)
#         next_boolean_map = self.get_unit_data(next_boolean_map)
#         next_float_map = self.get_unit_data(next_float_map)
#         next_scalars = self.get_unit_data(next_scalars)
#         terminated = self.get_unit_data(terminated)
#
#
#
#         with tf.GradientTape(persistent=True) as tape:
#             #print('------------')
#             # print(rewards_expanded.shape)
#             #print(boolean_map.shape)
#             #print(float_map.shape)
#             #print(scalars.shape)
#             current_score=self.get_score_model([boolean_map, float_map, scalars])  # [128,1]
#             next_score = self.get_score_model([next_boolean_map, next_float_map, next_scalars]) #[128,1]
#             # Train Value network
#             # define critic network loss model
#             gamma_terminated = tf.expand_dims(tf.multiply(tf.cast(tf.math.logical_not(terminated), tf.float32), self.gamma),axis=1)
#
#             rewards=tf.convert_to_tensor(reward, dtype=tf.float32)
#             rewards_expanded = tf.expand_dims(rewards, axis=1)
#             #print('------------')
#             # print(rewards_expanded.shape)
#             #print(gamma_terminated.shape)
#             #print(rewards_expanded.shape)
#
#             target_input = tf.add(rewards_expanded, tf.multiply(next_score, gamma_terminated))
#             #print(target_input.shape)
#             TD_error=target_input-current_score
#             #print(TD_error.shape)
#             #print(TD_error.shape)
#             #print(target_input.shape)
#             #print(q_value.shape)
#             a_loss = self.a_loss_model(
#                 [boolean_map, float_map, scalars, action, reward,
#                  terminated, TD_error])
#             c_loss = tf.square(TD_error)
#
#         a_grads = tape.gradient(a_loss, self.actor_network.trainable_variables)
#         self.actor_optimizer.apply_gradients(zip(a_grads, self.actor_network.trainable_variables))
#         #self.soft_update_a(self.params.alpha)
#
#         c_grads = tape.gradient(c_loss, self.critic_network.trainable_variables)
#         self.critic_optimizer.apply_gradients(zip(c_grads, self.critic_network.trainable_variables))
#
#
#     def get_unit_data(self,input_data):
#         return np.expand_dims(input_data, axis=0)
#
#     def save_weights(self, path_to_weights):
#         self.actor_network.save_weights(path_to_weights)
#         self.critic_network.save_weights(path_to_weights)
#
#     def save_model(self, path_to_model):
#         self.actor_network.save(path_to_model)
#         self.critic_network.save(path_to_model)
#
#     def load_weights(self, path_to_weights):
#         self.critic_network.load_weights(path_to_weights)
#         self.actor_network.load_weights(path_to_weights)
#         #self.hard_update()
#
#     def get_global_map(self, state):
#         boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
#         float_map_in = state.get_float_map()[tf.newaxis, ...]
#         return self.global_map_model([boolean_map_in, float_map_in]).numpy()
#
#     def get_local_map(self, state):
#         boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
#         float_map_in = state.get_float_map()[tf.newaxis, ...]
#         return self.local_map_model([boolean_map_in, float_map_in]).numpy()
#
#     def get_total_map(self, state):
#         boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
#         float_map_in = state.get_float_map()[tf.newaxis, ...]
#         return self.total_map_model([boolean_map_in, float_map_in]).numpy()
