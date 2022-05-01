import random
import functools
import matplotlib.pyplot as plt
import numpy as np


def soft_max_regression(X: np.array, W: np.array, C: np.array, b: np.array = None):
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
    grad_b = [] # ?
    return F, grad_W, grad_X, grad_b


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
