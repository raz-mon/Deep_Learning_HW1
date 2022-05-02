import random
import functools
import matplotlib.pyplot as plt
import numpy as np
from sol.activation_functions import ActiveFunc
from sol.activation_functions import ReLU

"""
Here we'll put our utility functions (SGD, derivatives etc.).
"""


# Todo: biases for this part.
def soft_max_regression(X: np.array, W: np.array, C: np.array, b: np.array = None):
    """""
    Computing the loss function 'Soft-Max regression'.
        :param X. The data input as a matrix of size nXm
        :param W. The weights, size of nXl (where l is the amount of labels)
        :param C. Indicators matrix. size of mXl.
        :return the loss function, and the gradients with respect to X,W.
    """""
    X_tW = X.T @ W
    arg = X_tW - etta(X_tW)
    prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
    m = len(X.T)
    F = - (1 / m) * np.sum(C * np.log(prob))
    grad_W = (1 / m) * (X @ (prob - C))
    grad_X = (1 / m) * (W @ (prob - C).T)
    grad_b = []  # ?
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
    etta = np.array([])
    for a in A:
        etta = np.append(etta, max(a))
    return etta.reshape(-1, 1)


def gradient_test(X: np.array, W: np.array, C: np.array, b: np.array, policy):
    """""
    Gradient test with respect for W.
    :param X matrix.
    :param W matrix.
    :param C matrix.
    :return matplotlib graph which shows the gradiant test.
    """""
    dict_num = {"W": 1, "X": 2, "b": 3}
    dict_param = {"W": W, "X": X, "b": b}
    dict_name = {"W": "$\delta W$", "X": "$\delta X$", "b": "$\delta b$"}
    V = np.random.rand(dict_param[policy].shape[0], dict_param[policy].shape[1])
    d = (V / np.linalg.norm(V))
    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    params = soft_max_regression(X, W, C)
    grad = params[dict_num[policy]].reshape(-1, 1)
    for k in range(1, 20):
        epsilon = 0.5 ** k
        new = dict_param[policy] + epsilon * d
        dict_args = {"W": (X, new, C, b), "X": (new, W, C, b), "b": (X, W, C, new)}
        f_x_d, _, _, _ = soft_max_regression(*dict_args[policy])
        err_1.append(abs(f_x_d - params[0]))
        err_2.append(abs(f_x_d - params[0] - (epsilon * d_vector.T @ grad)[0][0]))
        ks.append(k)
    print_grad_test(ks, err_1, err_2, dict_name[policy])


def gradient_test(X: np.array, W: np.array, C: np.array, b: np.array, policy):
    """""
    Gradient test with respect for W.
    :param X matrix.
    :param W matrix.
    :param C matrix.
    :return matplotlib graph which shows the gradiant test.
    """""
    dict_num = {"W": 1, "X": 2, "b": 3}
    dict_param = {"W": W, "X": X, "b": b}
    dict_name = {"W": "$\delta W$", "X": "$\delta X$", "b": "$\delta b$"}
    V = np.random.rand(dict_param[policy].shape[0], dict_param[policy].shape[1])
    d = (V / np.linalg.norm(V))
    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    params = soft_max_regression(X, W, C)
    grad = params[dict_num[policy]].reshape(-1, 1)
    for k in range(1, 20):
        epsilon = 0.5 ** k
        new = dict_param[policy] + epsilon * d
        dict_args = {"W": (X, new, C, b), "X": (new, W, C, b), "b": (X, W, C, new)}
        f_x_d, _, _, _ = soft_max_regression(*dict_args[policy])
        err_1.append(abs(f_x_d - params[0]))
        err_2.append(abs(f_x_d - params[0] - (epsilon * d_vector.T @ grad)[0][0]))
        ks.append(k)
    print_test(ks, err_1, err_2, "Gradiant Test: " + dict_name[policy])


def jacobian_test(X: np.array, W: np.array, b: np.array, active_func: ActiveFunc, policy):
    """""
    Gradient test with respect for W.
    :param X matrix.
    :param W matrix.
    :param C matrix.
    :return matplotlib graph which shows the gradiant test.
    """""
    dict_num = {"W": 1, "X": 2, "b": 3}
    dict_param = {"W": W, "X": X, "b": b}
    dict_name = {"W": "$\delta W$", "X": "$\delta X$", "b": "$\delta b$"}
    V = np.random.rand(W.shape[0], X.shape[1])
    d = (V / np.linalg.norm(V))
    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    f_x = active_func.activ(W @ X + b)
    for k in range(1, 20):
        epsilon = 0.5 ** k
        jac_v = JacMV(X, W, b, epsilon * d, active_func, policy)
        # dict_args = {"W": (W + epsilon * d) @ X + b, "X": W @ (X + epsilon * d) + b, "b": W @ X + (b + epsilon * d)}
        dict_args = {"W": (W @ X + b) + epsilon * d}
        f_x_d = active_func.activ(dict_args[policy])
        err_1.append(np.linalg.norm(f_x_d - f_x))
        err_2.append(np.linalg.norm(f_x_d - f_x - jac_v))
        ks.append(k)
    print_test(ks, err_1, err_2, "Jacobian Test: " + dict_name[policy])


def JacMV(X: np.array, W: np.array, b: np.array, V: np.array, active_func: ActiveFunc, policy):
    temp = W @ X + b
    arg = active_func.deriv(temp) * V
    dict_jac = {"W": arg @ X.T, "X": W.T @ arg, "b": np.sum(arg, axis=1).reshape(-1, 1)}
    return dict_jac[policy]


def print_test(k, err_1, err_2, title):
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
    plt.title(title)
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
