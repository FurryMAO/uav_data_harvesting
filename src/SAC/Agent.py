import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
from src.SAC.SAC_Network import Critic, Actor, Value
import numpy as np

def print_node(x):
    print(x)
    return x


class SACAgentParams:
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






class SACAgent(object):

    def __init__(self, params: SACAgentParams, example_state, example_action, stats=None):

        self.params = params
        self.gamma = tf.constant(self.params.gamma, dtype=float)
        self.boolean_map_shape = example_state.get_boolean_map_shape() #get the environment map shape
        self.float_map_shape = example_state.get_float_map_shape() # get the device data map
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input) #get the movement budget size
        self.num_actions = len(type(example_action)) # 6
        self.epsilon=0.1
        self.tau=0.005
        self.alpha=0.0003
        self.beta=0.0003
        self.boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        self.float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        self.scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        self.states = [self.boolean_map_input,
                  self.float_map_input,
                  self.scalars_input]
        self.action_input=Input(shape=(), name='action_input', dtype=tf.int64)
        self.input=self.get_network_input(self.states)

        # self.global_map_model = Model(inputs=[self.boolean_map_input, self.float_map_input],
        #                               outputs=self.global_map)
        # self.local_map_model = Model(inputs=[self.boolean_map_input, self.float_map_input],
        #                              outputs=self.local_map)
        # self.total_map_model = Model(inputs=[self.boolean_map_input, self.float_map_input],
        #                              outputs=self.total_map)
        self.critic_network_1 = Critic(name='critic_1')
        self.critic_network_2 = Critic(name='critic_2')
        self.value_network=Value(name='value')
        self.target_value_network = Value(name='target_value')
        self.actor_network=Actor(name='actor')

        self.actor_network.compile(optimizer=tf.optimizers.Adam(learning_rate=self.alpha))
        self.critic_network_1.compile(optimizer=tf.optimizers.Adam(learning_rate=self.beta))
        self.critic_network_2.compile(optimizer=tf.optimizers.Adam(learning_rate=self.beta))
        self.value_network.compile(optimizer=tf.optimizers.Adam(learning_rate=self.beta))
        self.target_value_network.compile(optimizer=tf.optimizers.Adam(learning_rate=self.beta))

        # self.actor_network = Actor()
        # #self.actor_network.build((None, self.state_dim))
        # self.critic_network_1(self.input)
        # self.critic_network_2(self.input)
        # self.actor_network(self.input)
        # self.value_network(self.input)
        # self.target_value_network(self.input)
        # # self.a_opt = tf.optimizers.Adam(learning_rate=params.learning_rate, clipvalue=1.0)
        # # self.c_opt=tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)
        #
        # if self.params.print_summary:
        #     self.actor_network.summary()
        #     self.value_network.summary()
        #     self.target_value_network.summary()
        #     self.critic_network_1.summary()
        #     self.critic_network_2.summary()

        if stats:
            stats.set_model(self.target_value_network)
            stats.set_model(self.actor_network)

    def get_network_input(self, inputs):
        boolean_map_input=inputs[0]
        float_map_input=inputs[1]
        scalars_input=inputs[2]
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)  # 张量数据类型转换
        padded_map = tf.concat([map_cast, float_map_input], axis=3)  # tensors combine in one dimension
        states_proc=scalars_input
        map_proc=padded_map
        flatten_map = self.create_map_proc(map_proc, name='pre') #return the flatten local and environment map
        pre_input = tf.concat([flatten_map, states_proc], axis=1)
        return pre_input


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

    def get_action(self, state):  # get the action base on the possibility
        # action is a array with dimension(1,)
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
            # action = np.random.randint(self.num_actions)
        # print(action.shape)
        return action

    def get_prob_get_action(self, into): # get the action base on the possibility
        prob_value = self.actor_network(into)
        batch_size = tf.shape(prob_value)[0]  # 获取prob_value的第一维大小
        actions = tf.random.categorical(prob_value, 1)  # 从prob_value中采样动作
        #actions = tf.squeeze(actions, axis=-1)  # 去掉形状为1的维度
        return prob_value, actions

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

    # def actor_loss(self, prob, action, td):
    #     dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
    #     log_prob = dist.log_prob(action)
    #     loss = -log_prob * td
    #     return loss

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

        input_current=self.get_network_input([boolean_map, float_map, scalars])
        input_next=self.get_network_input([next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value_network(input_current), 1)
            value_ = tf.squeeze(self.target_value_network(input_next), 1)

            log_probs,current_policy_actions= self.get_prob_get_action(input_current)

            #log_probs = tf.squeeze(log_probs, 1)
            inputs = [input_current, current_policy_actions]
            q1_new_policy = self.critic_network_1(inputs)
            q2_new_policy = self.critic_network_2(inputs)
            critic_value = tf.squeeze(
                tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            log_probs = tf.expand_dims(log_probs, axis=-1)
            value_target = critic_value - log_probs
            value_loss = 0.5 * tf.keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss,
                                               self.value_network.trainable_variables)
        self.value_network.optimizer.apply_gradients(zip(
            value_network_gradient, self.value_network.trainable_variables))

        #calculate and update actor network
        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.get_prob_get_action(input_current)
            log_probs = tf.squeeze(log_probs, 1)
            inputs = [input_current, current_policy_actions]
            q1_new_policy = self.critic_network_1(input_current, new_policy_actions)
            q2_new_policy = self.critic_network_2(input_current, new_policy_actions)
            critic_value = tf.squeeze(
                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor_network.trainable_variables)
        self.actor_network.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor_network.trainable_variables))

        # update the cricic network
        with tf.GradientTape(persistent=True) as tape:
            q_hat = reward + self.gamma * value_* (1 - np.asarray(terminated, dtype=np.int32))
            q1_old_policy = tf.squeeze(self.critic_network_1(input_current, action), 1)
            q2_old_policy = tf.squeeze(self.critic_network_2(input_current, action), 1)
            critic_1_loss = 0.5 * tf.keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * tf.keras.losses.MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                                  self.critic_network_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
                                                  self.critic_network_2.trainable_variables)

        self.critic_network_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_network_1.trainable_variables))
        self.critic_network_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_network_2.trainable_variables))

        self.update_network_parameters()


    def update_network_parameters(self,):
        tau = self.tau
        weights = []
        targets = self.target_value_network.weights
        for i, weight in enumerate(self.value_network.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.target_value_network.set_weights(weights)




    def save_model(self, path_to_model):
        self.actor_network.save(path_to_model)
        self.value_network.save(path_to_model)
        self.target_value_network.save(path_to_model)
        self.critic_network_1.save(path_to_model)
        self.critic_network_2.save(path_to_model)

    # def get_global_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.global_map_model([boolean_map_in, float_map_in]).numpy()
    #
    # def get_local_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.local_map_model([boolean_map_in, float_map_in]).numpy()
    #
    # def get_total_map(self, state):
    #     boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
    #     float_map_in = state.get_float_map()[tf.newaxis, ...]
    #     return self.total_map_model([boolean_map_in, float_map_in]).numpy()