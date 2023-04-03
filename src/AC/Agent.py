import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
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

        # Create shared inputs
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        td_input = Input(shape=(), name='advantages _input', dtype=tf.float32)


        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,
                  float_map_input,
                  scalars_input]

        map_cast = tf.cast(boolean_map_input, dtype=tf.float32) #张量数据类型转换
        padded_map = tf.concat([map_cast, float_map_input], axis=3) # tensors combine in one dimension

        self.critic_network = self.build_critic_model(padded_map, scalars_input, states,'critic_model')

        self.actor_network = self.build_actor_model(padded_map, scalars_input, states, 'actor_model')

        self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                      outputs=self.global_map)
        self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.local_map)
        self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.total_map)

        values= self.critic_network.output
        action_probs_logist = self.actor_network.output
        self.critic_model = Model(inputs=states, outputs=values)

        # Softmax explore model
        # softmax_scaling = action_probs_logist - tf.reduce_max(action_probs_logist, axis=-1, keepdims=True)
        # softmax_scaling = tf.divide(softmax_scaling, tf.constant(self.params.soft_max_scaling, dtype=float))
        # softmax_action_possibility = tf.math.softmax(softmax_scaling, name='softmax_action_posibility')
        action_probs =tf.keras.activations.softmax(action_probs_logist)
        self.soft_explore_model = Model(inputs=states, outputs=action_probs )

        self.a_opt = tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)
        self.c_opt=tf.optimizers.Adam(learning_rate=params.learning_rate, amsgrad=True)

        '''
        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action = tf.argmax(q_values, axis=1, name='max_action', output_type=tf.int64)
        max_action_target = tf.argmax(q_target_values, axis=1, name='max_action', output_type=tf.int64)
        one_hot_max_action = tf.one_hot(max_action, depth=self.num_actions, dtype=float)
        q_star = tf.reduce_sum(tf.multiply(one_hot_max_action, q_target_values, name='mul_hot_target'), axis=1,
                               name='q_star')
        self.q_star_model = Model(inputs=states, outputs=q_star)

        # Define Bellman loss
        one_hot_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=1.0, off_value=0.0, dtype=float)
        one_cold_rm_action = tf.one_hot(action_input, depth=self.num_actions, on_value=0.0, off_value=1.0, dtype=float)
        q_old = tf.stop_gradient(tf.multiply(q_values, one_cold_rm_action))
        gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(termination_input), tf.float32), self.gamma)
        q_update = tf.expand_dims(tf.add(reward_input, tf.multiply(q_star_input, gamma_terminated)), 1)
        q_update_hot = tf.multiply(q_update, one_hot_rm_action)
        q_new = tf.add(q_update_hot, q_old)
        q_loss = tf.losses.MeanSquaredError()(q_new, q_values)
        self.q_loss_model = Model(
            inputs=states + [action_input, reward_input, termination_input, q_star_input],
            outputs=q_loss)'''

        # Exploit act model
        # self.exploit_model = Model(inputs=states, outputs=max_action)
        # self.exploit_model_target = Model(inputs=states, outputs=max_action_target)



        if self.params.print_summary:
            self.critic_network.summary()

        if stats:
            stats.set_model(self.actor_network)
            stats.set_model(self.critic_network)

    def build_actor_model(self, map_proc, states_proc, inputs, name=''):
        flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        policy = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=policy)

        return model

    def build_critic_model(self, map_proc, states_proc, inputs, name=''):
        flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        value = Dense(1, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=value)

        return model


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

    def get_exploitation_action(self, state): # get the maximum value action
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_action(self, state): # get the action base on the possibility
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        prob = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        #return np.random.choice(range(self.num_actions), size=1, p=p)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return action

    def get_exploitation_action_target(self, state):
        return self.get_action(state)


    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob*td
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
        # Train Value network
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.soft_explore_model([boolean_map, float_map, scalars])
            v = self.critic_model(
                [boolean_map, float_map, scalars])
            v_ = self.critic_model(
                [next_boolean_map, next_float_map, next_scalars])
            td = reward + self.gamma * v_ * (1 - np.asarray(terminated, dtype=np.int32)) - v
            a_loss = self.actor_loss(p, action, td)
            c_loss = td ** 2
        grads1 = tape1.gradient(a_loss, self.actor_network.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic_network.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor_network.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic_network.trainable_variables))
        return a_loss, c_loss

    # def save_weights(self, path_to_weights):
    #     self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.actor_network.save(path_to_model)
        self.critic_network.save(path_to_model)

    # def load_weights(self, path_to_weights):
    #     self.q_network.load_weights(path_to_weights)
    #     self.hard_update()

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