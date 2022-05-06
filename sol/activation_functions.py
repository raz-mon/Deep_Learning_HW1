import numpy as np


class ActiveFunc:
    def __init__(self, activ, deriv):
        self.activ = activ
        self.deriv = deriv

    def act(self, arg):
        return self.activ(arg)

    def der(self, arg):
        return self.deriv(arg)


class ReLU(ActiveFunc):
    def __init__(self):
        activ = lambda x: x if x > 0 else 0
        activ_vec = np.vectorize(activ)
        deriv = lambda x: 1 if x > 0 else 0
        deriv_vec = np.vectorize(deriv)
        super().__init__(activ_vec, deriv_vec)


class Tanh(ActiveFunc):
    def __init__(self):
        activ = np.tanh
        deriv = lambda x: (1 / np.cosh(x)) ** 2
        super().__init__(activ, deriv)


class Identity(ActiveFunc):
    def __init__(self):
        activ = lambda x: x
        deriv = lambda x: 0
        super().__init__(np.vectorize(activ), np.vectorize(deriv))

