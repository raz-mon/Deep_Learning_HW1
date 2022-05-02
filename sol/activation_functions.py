import numpy as np


class ActiveFunc:
    def __init__(self, activ, deriv):
        self.activ = activ
        self.deriv = deriv

    def act(self):
        return self.activ

    def der(self):
        return self.deriv


class ReLU(ActiveFunc):
    def __init__(self):
        activ = lambda x: x if x > 0 else 0
        deriv = lambda x: 1 if x > 0 else 0
        super().__init__(activ, deriv)


class Tanh(ActiveFunc):
    def __init__(self):
        activ = np.tanh
        deriv = lambda x: (1 / np.cosh(x)) ** 2
        super().__init__(activ, deriv)
