import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, name='my_critic'):
        super(Critic, self).__init__(name=name)
        self.state_layer = tf.keras.layers.Dense(128, activation='relu')
        self.action_layer = tf.keras.layers.Dense(128, activation='relu')
        self.concat_layer = tf.keras.layers.Concatenate()
        self.q1_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        state, action = inputs
        state = self.state_layer(state)
        action = self.action_layer(action)
        concat = self.concat_layer([state, action])
        q1 = self.q1_layer(concat)
        return q1


class Actor(tf.keras.Model):
    def __init__(self, name='my_actor'):
        super().__init__(name=name)
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.logits = tf.keras.layers.Dense(6, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        logits = self.logits(x)
        a = tf.nn.softmax(logits)
        return a


class Value(tf.keras.Model):
    def __init__(self, name='my_value'):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.a = tf.keras.layers.Dense(1, activation=None)
        self.model_name = name

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v= self.a(x)
        return v