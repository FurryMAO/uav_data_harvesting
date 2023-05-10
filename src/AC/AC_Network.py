import tensorflow as tf



class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__(name='my_critic')
        self.d1 = tf.keras.layers.Dense(256, activation='relu')
        self.d2 = tf.keras.layers.Dense(256, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        return v


class Actor(tf.keras.Model):
    def __init__(self):
        super().__init__(name='my_actor')
        self.soft_max_scaling=0.1
        self.d1 = tf.keras.layers.Dense(256, activation='relu')
        self.d2 = tf.keras.layers.Dense(256, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.a = tf.keras.layers.Dense(6, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        logits = self.a(x)
        action_probs = tf.keras.activations.softmax(logits)
        # softmax_prob = tf.math.softmax(softmax_scaling)
        action_probs = tf.clip_by_value(action_probs, 1e-8, 1 - 1e-8)
         # 限制概率的范围
        return action_probs