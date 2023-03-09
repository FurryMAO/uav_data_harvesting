import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
import numpy as np


class PGAgentParams:
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


class PGAgent(object):

    def __init__(self, params: PGAgentParams, example_state, example_action, stats=None):
        self.params = params
        gamma = tf.constant(self.params.gamma, dtype=float)
        self.boolean_map_shape = example_state.get_boolean_map_shape() #get the environment map shape
        self.float_map_shape = example_state.get_float_map_shape() # get the device data map
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input) #get the movement budget size
        self.num_actions = len(type(example_action)) # 1

        # Create shared inputs
        action_input = Input(shape=(), name='action_input', dtype=tf.int64)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        states = [boolean_map_input,float_map_input,scalars_input]
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32) #张量数据类型转换
        padded_map = tf.concat([map_cast, float_map_input], axis=3) # tensors combine in one dimension


        #build the policy gradient network
        self.pg_network = self.build_model(padded_map, scalars_input, states,'PG model')
        logits=self.pg_network.outputs
        action_probs = tf.keras.activations.softmax(logits)
        self.soft_explore_model = Model(inputs=states, outputs=action_probs)


        self.global_map_model = Model(inputs=[boolean_map_input, float_map_input], outputs=self.global_map)
        self.local_map_model = Model(inputs=[boolean_map_input, float_map_input], outputs=self.local_map)
        self.total_map_model = Model(inputs=[boolean_map_input, float_map_input], outputs=self.total_map)

        # Softmax explore model
        softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        # Exploit act model
        self.exploit_model_target = Model(inputs=states, outputs=max_action_target)

        if self.params.print_summary:
            self.q_loss_model.summary()

        if stats:
            stats.set_model(self.target_network)



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


    def build_model(self, map_proc, states_proc, inputs, name=''):
        flatten_map = self.create_map_proc(map_proc, name) #return the flatten local and environment map
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        model = Model(inputs=inputs, outputs=output)
        return model


    def get_random_action(self):
        return np.random.randint(0, self.num_actions)

    def get_action(self, state): # get the action base on the possiblity
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        p = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        return np.random.choice(range(self.num_actions), size=1, p=p)

    def get_exploitation_action_target(self, state):
        return self.get_action(state)

    def act(self, state):
        return self.get_action(state)

    def _discount_and_norm_rewards(self):
        """
        通过回溯计算G值
        """
        # 先创建一个数组，大小和ep_rs一样。ep_rs记录的是每个状态的收获r。
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 从ep_rs的最后往前，逐个计算G
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 归一化G值。
        # 我们希望G值有正有负，这样比较容易学习。
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


    def train(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]

        if self.params.blind_agent:
            q_star = self.q_star_model(
                [next_scalars])
        else:
            q_star = self.q_star_model(
                [next_boolean_map, next_float_map, next_scalars])

        # Train Value network
        with tf.GradientTape() as tape:
            # 把s放入神经网络，就算_logits
            _logits = self.model(np.vstack(self.ep_obs))

            # 敲黑板
            ## _logits和真正的动作的差距
            # 差距也可以这样算,和sparse_softmax_cross_entropy_with_logits等价的:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_logits, labels=np.array(self.ep_as))

            # 在原来的差距乘以G值，也就是以G值作为更新
            loss = tf.reduce_mean(neg_log_prob * discounted_ep_rs_norm)

        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

        #     if self.params.blind_agent:
        #         q_loss = self.q_loss_model(
        #             [scalars, action, reward,
        #              terminated, q_star])
        #     else:
        #         q_loss = self.q_loss_model(
        #             [boolean_map, float_map, scalars, action, reward,
        #              terminated, q_star])
        # q_grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        # self.q_optimizer.apply_gradients(zip(q_grads, self.q_network.trainable_variables))
        #
        # self.soft_update(self.params.alpha)


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
