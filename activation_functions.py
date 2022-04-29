import numpy as np


class Relu:
    def __init__(self):
        self.relu = lambda x: x if x > 0 else 0
        self.der_relu = lambda x: 1 if x > 0 else 0


class Tanh:
    def __init__(self):
        self.tanh = np.tanh
        self.der_tanh = lambda x: (1 / np.cosh(x)) ** 2
