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
        self.stats=stats
        self.boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
        self.float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
        self.scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
        self.states = [self.boolean_map_input,
                  self.float_map_input,
                  self.scalars_input]
        self.input=self.get_network_input(self.states)

        # self.global_map_model = Model(inputs=[self.boolean_map_input, self.float_map_input],
        #                               outputs=self.global_map)
        # self.local_map_model = Model(inputs=[self.boolean_map_input, self.float_map_input],
        #                              outputs=self.local_map)
        # self.total_map_model = Model(inputs=[self.boolean_map_input, self.float_map_input],
        #                              outputs=self.total_map)
        self.critic_network = Critic()
        #self.critic_network.build((None, self.state_dim))
        self.actor_network = Actor()
        #self.actor_network.build((None, self.state_dim))
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
        padded_map = tf.concat([map_cast, float_map_input], axis=3)  # tensors combine in one dimension
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
            action = np.random.randint(self.num_actions)
        return action

    # def get_action_probossiblity(self, prob_value): # get the action base on the possibility
    #     softmax_scaling = tf.divide(prob_value, tf.constant(self.params.soft_max_scaling, dtype=float))
    #     softmax_action = tf.math.softmax(softmax_scaling)
    #     p = tf.reshape(softmax_action, [-1]).numpy()
    #     return p

    def get_exploitation_action_target(self, state):
        return self.get_action(state)


    def actor_loss(self, prob, action, td):
        entropy_weight = 0.001
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        loss = -(log_prob * td + entropy_weight * entropy)
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
        return a_loss, c_loss


    def save_model(self, path_to_model):
        self.actor_network.save(path_to_model)
        self.critic_network.save(path_to_model)

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