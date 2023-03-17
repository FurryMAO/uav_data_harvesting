import tensorflow as tf
import numpy as np


def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype
    else:
        return type(exp)


class TrackStorage:
    """
    Replay memory class for RL
    """

    def __init__(self, size):
        self.done = False

    def initialize(self):
        self.stateslist = []
        self.actionslist = []
        self.rewardslist = []


    def store(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        current_state=[boolean_map, float_map, scalars]
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = experiences[4]
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]
        next_state=[next_boolean_map, next_float_map, next_scalars]
        self.stateslist.append(current_state)
        self.actionslist.append(action)
        self.rewardslist.append(reward)
        if terminated is True:
            self.done=True
        else: self.done= False

        if self.done is True:
            print("Already find one track")
            self.stateslist.append(next_state)

    def get_track(self):
        return self.actionslist, self.rewardslist, self.stateslist




