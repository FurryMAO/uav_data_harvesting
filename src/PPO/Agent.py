import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
import numpy as np
from collections import namedtuple

def print_node(x):
    print(x)
    return x


class PPOAgentParams:
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


class PPOAgent(object):

    def __init__(self, params: PPOAgentParams, example_state, example_action, stats=None):

        self.params = params
        self.gamma = tf.constant(self.params.gamma, dtype=float)
        self.boolean_map_shape = example_state.get_boolean_map_shape() #get the environment map shape
        self.float_map_shape = example_state.get_float_map_shape() # get the device data map
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input) #get the movement budget size
        self.num_actions = len(type(example_action)) # 1
        self.buffer = []  # 数据缓冲池
        actor_learning_rate=self.params.actor_learning_rate
        critic_learning_rate=self.params.critic_learning_rate
        self.actor_optimizer = tf.optimizers.Adam(actor_learning_rate)  # Actor优化器
        self.critic_optimizer = tf.optimizers.Adam(critic_learning_rate)  # Critic优化器
        self.Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])



        # Create shared inputs
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)

        ####--------------#######@自定义变量集合

        self.epsilon=0.2
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,
                  float_map_input,
                  scalars_input]

        map_cast = tf.cast(boolean_map_input, dtype=tf.float32) #张量数据类型转换
        #print(map_cast.shape)
        #print(float_map_input.shape)
        padded_map = tf.concat([map_cast, float_map_input], axis=3) # tensors combine in one dimension

        self.value_network = self.build_critic_model(padded_map, scalars_input, states, 'critic_network')
        self.logits_network = self.build_actor_model(padded_map, scalars_input, states, 'act_network')

        self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                      outputs=self.global_map)
        self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.local_map)
        self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.total_map)
        value = self.value_network.output
        logits = self.logits_network.output

        self.critic_model=Model(inputs=states, outputs=value)
        self.actor_model=Model(inputs=states, outputs=logits)


       ######################@self define model

        max_action = tf.argmax(logits, axis=1, name='max_action', output_type=tf.int64)
        self.exploit_model_target = Model(inputs=states, outputs=max_action)
        # Softmax explore model

        scaled_logits = logits
        softmax_scaling = tf.divide(scaled_logits, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.keras.activations.softmax(softmax_scaling)
        softmax_action= tf.clip_by_value(softmax_action, 1e-8, 1 - 1e-8)

        self.soft_explore_model = Model(inputs=states, outputs=softmax_action)


        if self.params.print_summary:
            self.value_network.summary()
            self.logits_network.summary()

        if stats:
            stats.set_model(self.value_network)
            stats.set_model(self.logits_network)

    def build_actor_model(self, map_proc, states_proc, inputs, name=''):
        flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        model = Model(inputs=inputs, outputs=output)

        return model

    def build_critic_model(self, map_proc, states_proc, inputs, name=''):
        flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(1, activation='linear', name=name + 'output_layer')(layer)
        model = Model(inputs=inputs, outputs=output)
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

    def act(self, state): #get the achtion based on possibility
        p, a=self.get_soft_max_exploration(state)
        return p, a


    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_soft_max_exploration(self, state): # given the state and get the action base on the possibility
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        prob = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        a= np.random.choice(range(self.num_actions), size=1, p=prob)
        action_onehot = tf.one_hot(a, self.num_actions)
        action_prob = tf.reduce_sum(prob * action_onehot)
        return action_prob, a

    def get_exploitation_action_target(self, state): # given state information and get the maximum value action for target netweork
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        return self.exploit_model_target([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def train(self, experiences):
        boolean_map = experiences[0]
        # print("a=",boolean_map.shape)
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]
        old_action_log_prob = tf.convert_to_tensor(experiences[9], dtype=tf.float32)
        #old_action_log_prob= tf.expand_dims(old_action_log_prob, axis=1)
        # 通过MC方法循环计算R(st)
        discounted = []
        r = 0
        for reward, done in zip(reward[::-1], terminated[::-1]):
            r = reward + self.gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        Rs= discounted[::-1]
        Rs = tf.convert_to_tensor(Rs, dtype=tf.float32)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            v = self.critic_model([boolean_map, float_map, scalars])
            v_target = tf.expand_dims(Rs, axis=1)
            delta = v_target - v  # 计算优势值
            advantage = tf.stop_gradient(delta)  # 断开梯度连接
            pi = self.soft_explore_model([boolean_map, float_map, scalars])
            action = tf.cast(action, dtype=tf.int32)
            action = tf.expand_dims(action, axis=1)
            indices = tf.expand_dims(tf.range(action.shape[0]), axis=1)
            indices = tf.concat([indices, action], axis=1)
            pi_a = tf.gather_nd(pi, indices)  # 动作的概率值pi(at|st), [b]
            pi_a = tf.expand_dims(pi_a, axis=1)

            # 重要性采样
            ratio = (pi_a /old_action_log_prob)
            surr1 = ratio * advantage
            surr2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            # PPO误差函数
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            # 对于偏置v来说，希望与MC估计的R(st)越接近越好
            value_loss = tf.losses.MSE(v_target, v)
        # 优化策略网络
        grads1 = tape1.gradient(policy_loss, self.logits_network.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads1, self.logits_network.trainable_variables))
        # 优化偏置值网络
        grads2 = tape2.gradient(value_loss, self.value_network.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads2, self.value_network.trainable_variables))



    def save_weights(self, path_to_weights):
        self.value_network.save_weights(path_to_weights)
        self.logits_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.value_network.save(path_to_model)
        self.logits_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.value_network.load_weights(path_to_weights)
        self.logits_network.load_weights(path_to_weights)


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


