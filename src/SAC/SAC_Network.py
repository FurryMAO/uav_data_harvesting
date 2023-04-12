import tensorflow as tf



class Critic(tf.keras.Model):
    def __init__(self,action,name='my_critic'):
        super().__init__()
        self.action_input = action
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(128, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)
        self.model_name = name

    def call(self, input_data):
        pre_combine = tf.cast(self.action_input, tf.float32)
        x = self.d1(tf.concat([input_data, pre_combine], axis=1))
        x = self.d2(x)
        q = self.q(x)
        return q


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