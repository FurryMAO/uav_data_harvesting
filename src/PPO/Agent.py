import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D
import tensorflow_probability as tfp
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
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=params.actor_learning_rate)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=params.critic_learning_rate)
        ####--------------#######@自定义变量集合
        self.gamma = 0.95
        self.clip_pram=0.3
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

        self.A_network = self.build_actor_model(padded_map, scalars_input, states, 'actor')
        self.C_network = self.build_critic_model(padded_map, scalars_input, states, 'critic')

        self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                      outputs=self.global_map)
        self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.local_map)
        self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                     outputs=self.total_map)

        possibility = self.A_network.output
        values = self.C_network.output

        # Define Q* in min(Q - (r + gamma_terminated * Q*))^2
        max_action = tf.argmax(possibility, axis=1, name='max_action', output_type=tf.int64)

        # Exploit act model
        self.exploit_model = Model(inputs=states, outputs=max_action)
        self.exploit_model_target = Model(inputs=states, outputs=max_action)

        # Softmax explore model

        self.soft_explore_model = Model(inputs=states, outputs=possibility)
        self.criting_model = Model(inputs=states, outputs=values)

        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=params.actor_learning_rate)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=params.critic_learning_rate)

        # self.a_opt = tf.optimizers.Adam(learning_rate=params.actor_learning_rate, amsgrad=True)
        # self.c_opt = tf.optimizers.Adam(learning_rate=params.critic_learning_rate, amsgrad=True)

        if self.params.print_summary:
            self.A_network.summary()
            self.C_network.summary()

        if stats:
            stats.set_model(self.A_network)
            stats.set_model(self.C_network)

    def build_actor_model(self, map_proc, states_proc, inputs, name=''):
        flatten_map = self.create_map_proc(map_proc, name)  # return the flatten local and environment map
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        softmax_scaling = tf.divide(output, tf.constant(self.params.soft_max_scaling, dtype=float))
        softmax_action = tf.keras.activations.softmax(softmax_scaling)
        softmax_action = tf.clip_by_value(softmax_action, 1e-8, 1 - 1e-8)

        model = Model(inputs=inputs, outputs=softmax_action)

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

    def act(self, state):  # get the achtion based on possibility
        action = self.get_soft_max_exploration(state)
        return action

    def get_random_action(self):
        random_action = np.random.randint(0, self.num_actions)
        random_action_tensor = tf.convert_to_tensor(random_action, dtype=tf.int32)
        random_action_tensor = tf.expand_dims(random_action_tensor, axis=0)
        return random_action_tensor


    def get_exploitation_action(self, state): # get the maximum value action
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]

    def get_soft_max_exploration(self, state): # given the state and get the action base on the possibility
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        p = self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        return np.random.choice(range(self.num_actions), size=1, p=p)


    def get_old_possiblility_value(self,state,action):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]
        p=self.soft_explore_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        v=self.criting_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
        return p,v


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
        value=tf.convert_to_tensor(experiences[10], dtype=tf.float32)
        last_state=[next_boolean_map,next_float_map,next_scalars]
        v_=self.C_network(last_state)
        last_element = v_[-1]
        value = tf.concat([value, tf.expand_dims(last_element, axis=0)], axis=0)

        returns, adv = self.preprocess(reward, terminated, value, self.gamma)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            input_current = [boolean_map,
                             float_map,
                             scalars]
            p_new = self.A_network(input_current, training=True)
            v = self.C_network(input_current, training=True)
            td = tf.math.subtract(returns, v)
            c_loss = 0.5 * tf.losses.MSE(returns, v)
            a_loss = self.actor_loss(p_new, action, adv, old_action_log_prob , c_loss)
        grads1 = tape1.gradient(a_loss, self.A_network.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.C_network.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.A_network.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.C_network.trainable_variables))

    def preprocess(self, rewards, done, values,  gamma):
        g = 0
        lmbda = 0.95
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * done[i] - values[i]
            g = delta + gamma * lmbda * done[i] * g
            returns.append(g + values[i])

        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - values[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        returns = np.array(returns, dtype=np.float32)
        return returns, adv




    def actor_loss(self, probs, actions, adv, old_probs, closs):
        probability = probs
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability, tf.math.log(probability))))
        # print(probability)
        # print(entropy)
        sur1 = []
        sur2 = []
        for pb, t, op, a in zip(probability, adv, old_probs, actions):
            a = tf.squeeze(a, axis=0)
            t = tf.constant(t)
            # op =  tf.constant(op)
            # print(f"t{t}")
            # ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
            ratio = tf.math.divide(pb[a], op[a])
            # print(f"ratio{ratio}")
            s1 = tf.math.multiply(ratio, t)
            # print(f"s1{s1}")
            s2 = tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram), t)
            # print(f"s2{s2}")
            sur1.append(s1)
            sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)

        # closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        # print(loss)
        return loss



    def save_weights(self, path_to_weights):
        self.A_network.save_weights(path_to_weights)
        self.C_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.A_network.save(path_to_model)
        self.C_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.A_network.load_weights(path_to_weights)
        self.C_network.load_weights(path_to_weights)

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

