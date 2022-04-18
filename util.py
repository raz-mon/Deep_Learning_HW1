import random

import numpy as np

"""
Here we'll put our utility functions (SGD, derivatives etc.).
"""


class SGD:
    """Optimize a function using Stochastic Gradient Descent"""

    def __init__(self, func, params, f_grad, data, indicator_arr, mb_size=50, learning_rate=0.001, toll=10e-19,
                 max_iter=20):
        """
        Initialize SGD optimizer.
        :param func: Objective function of which we wish to find the Minimum using SGD.
        :param params: Parameters of function 'func'.
        :param f_grad: Gradient of the function we wish to minimize.
        :param data: Data.
        :param indicator_arr: Indicator array.
        :param mb_size: Mini-batch size
        :param learning_rate: Learning rate.
        :param toll: Tolerance for distance between consecutive gradients.
        :param max_iter: Maximum iterations.
        """
        self.func = func
        self.params = params  # params = weights!
        self.f_grad = f_grad
        self.data = data
        self.indicator_arr = indicator_arr
        self.mb_size = mb_size
        self.lr = learning_rate
        self.toll = toll
        self.max_iter = max_iter
        self.last_grad = None
        self.loss = []

    def _step(self):
        """
        Perform the Gradient-Descent step (epoch), on data 'data', with objective function with gradient 'f_grad'.
        :return: The distance between the current gradient and the previous one.
        """
        rand_data = self._rand_pick()[0]  # ignore the indicator array for now.
        grad = (1 / len(rand_data)) * sum([self.f_grad(d_i, self.params) for d_i in rand_data])
        self.params = self.params - (self.lr * grad)
        dist = np.linalg.norm(grad - self.last_grad)
        self.last_grad = grad
        return dist

    def go(self):
        """
        Perform the SGD algorithm, until one of the stoppage conditions are met.
        :param max_iter: Maximum iterations.
        :param toll: From this distance and smaller, we can stop computing, we are close enough to the minumum.
        :return:
        """
        for i in range(self.max_iter):
            """
            if self._step() < self.toll:
                break
            """
            self._step()
            self.loss += self._calc_loss()
        return self.params

    def _calc_loss(self):
        return sum([self.func(x_i, self.params) for x_i in self.data]) / len(self.data)

    def _rand_pick(self):
        """
        Pick mb_size samples from the data, return an array with them and their respective indicator array.
        :return:
        """
        rand_data_arr = []
        rand_indicator_arr = []

        for ind in random.sample(range(0, len(self.data)), self.mb_size):
            rand_data_arr += [self.data[ind]]
            rand_indicator_arr += [self.indicator_arr[ind]]

        return [rand_data_arr, rand_indicator_arr]


# Todo: biases for this part.
def soft_max_regression(X: np.array, W: np.array, C: np.array):
    X_tW = X.transpose() @ W
    divisor = np.sum(np.exp(X_tW), axis=0)
    F = np.sum(C * np.log(np.exp(X_tW) / divisor), axis=0)
    grad_w_F = 1 / (len(X)) * (X @ (np.exp(X_tW) / divisor - C))
    return F, grad_w_F
