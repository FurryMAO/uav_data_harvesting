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

class ReplayMemory:
    """
    Replay memory class for RL
    """

    def __init__(self, size):
        self.k = 0
        self.size = size
        self.memory = None

    def initialize(self, experience):
        self.memory = [np.zeros(shape=[self.size] + shape(exp), dtype=type_of(exp)) for exp in experience]

    def store(self, experience):
        if self.memory is None:
            self.initialize(experience)
        if len(experience) != len(self.memory):
            raise Exception('Experience not the same size as memory', len(experience), '!=', len(self.memory))

        for e, mem in zip(experience, self.memory):
            mem[self.k] = e

        self.k += 1

    def sample(self):
        if self.memory is None:
            return None
        else:
            return [mem[:self.k] for mem in self.memory]

    def reset(self):
        self.k = 0
        self.memory=None