import random
import functools
import matplotlib.pyplot as plt
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
        :return:
        """
        for i in range(self.max_iter):
            """
            if self._step() < self.toll:
                break
            """
            self._step()
            self.loss += [self._calc_loss()]
        return self.params

    def _calc_loss(self):
        return sum([self.func(x_i, *self.params) for x_i in self.data]) / len(self.data)

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
    """""
    Computing the loss function 'Soft-Max regression'.
        :param X. The data input as a matrix of size nXm
        :param W. The weights, size of nXl (where l is the amount of labels)
        :param C. Indicators matrix. size of mXl.
        :return the loss function, and the gradients with respect to X,W.
    """""
    X_tW = X.transpose() @ W
    arg = X_tW - etta(X_tW)
    prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
    m = len(X.T)
    F = - (1 / m) * np.sum(C * np.log(prob))
    grad_W = (1 / m) * (X @ (prob - C))
    grad_X = (1 / m) * (W @ (prob - C).T)
    grad_b = (1 / m) * np.sum((prob - C), axis=1).reshape(-1, 1)
    return F, grad_W, grad_X, grad_b


def sm_loss(X, W, C):
    """

    :param X:
    :type X:
    :param W:
    :type W:
    :param C:
    :type C:
    :return:
    :rtype:
    """
    m = len(X.T)
    X_tW = X.T @ W
    arg = X_tW - etta(X_tW)
    prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
    F = - (1 / m) * np.sum(C * np.log(prob))
    return F


def sm_grad_w(X, W, C):
    """

    :param X:
    :type X:
    :param W:
    :type W:
    :param C:
    :type C:
    :return:
    :rtype:
    """
    X_tW = X.T @ W
    arg = X_tW - etta(X_tW)
    prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
    m = len(X.T)
    return (1 / m) * (X @ (prob - C))


def etta(A: np.array):
    """""
    This method calculate the etta vector that required to reduce from A in order to prevent numerical overflow.
    :return etta vector. this vector is the column with the maximal norm from A.
    """""
    etta = A.T[0]
    for a in A.T:
        if np.linalg.norm(a) > np.linalg.norm(etta):
            etta = a
    return etta.reshape(-1, 1)


def gradient_test_X(X: np.array, W: np.array, C: np.array):
    """""
    Gradient test with respect for X.
    :param X matrix.
    :param W matrix.
    :param C matrix.
    :return matplotlib graph which shows the gradiant test.
    """""
    V = np.random.rand(X.shape[0], X.shape[1])
    d = (V / np.linalg.norm(V))
    err_1 = []
    err_2 = []
    ks = []
    for k in range(1, 10):
        epsilon = 0.5 ** k
        f_x, _, grad_X, _ = soft_max_regression(X, W, C)
        f_x_d, _, grad_X_d, _ = soft_max_regression(X + epsilon * d, W, C)
        err_1.append(abs(f_x_d - f_x))
        err_2.append(abs(f_x_d - f_x - epsilon * (d.reshape(1, -1) @ grad_X.reshape(-1, 1))[0][0]))
        ks.append(k)
    print_grad_test(ks, err_1, err_2, "$\delta X$")


def gradient_test_W(X: np.array, W: np.array, C: np.array):
    """""
    Gradient test with respect for W.
    :param X matrix.
    :param W matrix.
    :param C matrix.
    :return matplotlib graph which shows the gradiant test.
    """""
    V = np.random.rand(W.shape[0], W.shape[1])
    d = (V / np.linalg.norm(V))
    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    f_x, grad_W, _, _ = soft_max_regression(W @ X, W, C)

    grad_W = grad_W.reshape(-1, 1)
    for k in range(1, 25):
        epsilon = 0.5 ** k
        W_new = W + epsilon * d
        X_new = W_new @ X
        f_x_d, _, _, _ = soft_max_regression(X_new, W_new, C)
        err_1.append(abs(f_x_d - f_x))
        err_2.append(abs(f_x_d - f_x - (epsilon * d_vector.T @ grad_W)[0][0]))
        ks.append(k)
    print_grad_test(ks, err_1, err_2, "$\delta W$")


def print_grad_test(k, err_1, err_2, title):
    """""
    Plot a graph with O(e) and O(e^2) for the gradiant test. 
    """""
    plt.rc("font", size=16, family="Times New Roman")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.semilogy(k, err_1, label="O($\\varepsilon$)")
    ax.semilogy(k, err_2, label="O($\\varepsilon^2$)")
    ax.set_xlabel("k", fontdict={"size": 21})
    ax.set_ylabel("error", fontdict={"size": 21})
    plt.grid(True)
    plt.title("Gradiant Test: " + title)
    plt.legend()
    plt.show()


def generate_batches(X, C, mb_size):
    """

    :param X:
    :type X:
    :param C:
    :type C:
    :param mb_size:
    :type mb_size:
    :return:
    :rtype:
    """
    data = []
    mb = []
    mbs = []
    # Generate 'data' - An array containing 2-component arrays of [data, indicator].
    for i in range(len(X)):
        data.append([X[i], C[i]])  # C[i] is the i'th row, corresponding to the i'th data-sample (it's indicator).
    indices = list(range(len(data)))
    random.shuffle(indices)
    while len(indices) > mb_size:
        for i in range(mb_size):
            mb.append(data[indices.pop()])
        """
        Mb: Array of size nXmb_size.
        Indicator: Matrix of size lXmb_size
        """
        Mb, Indicator = functools.reduce(lambda acc, curr: [acc[0] + [curr[0]], acc[1] + [curr[1]]], mb, [[], []])
        mb = []
        mbs += [(np.array(Mb).T, np.array(Indicator))]

    return mbs
