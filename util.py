"""
Here we'll put our utility functions (SGD, derivatives etc.).
"""

import numpy as np


class SGD:
    """Optimize a function using Stochastich Gradient Descent"""


# Todo: biases
def soft_max_regression(X: np.array, W: np.array, C: np.array):
    X_tW = X.transpose() @ W
    divisor = np.sum(np.exp(X_tW), axis=0)
    F = np.sum(C * np.log(np.exp(X_tW) / divisor), axis=0)
    grad_w_F = 1 / (len(X)) * (X @ (np.exp(X_tW) / divisor - C))
    return F, grad_w_F


def gradient_test(v: np.array, epsilon: float, X: np.array, W: np.array, C: np.array):
    d = v / np.linalg.norm(v)
    f_x, df_x = soft_max_regression(X, W, C)
    f_x_ed, df_x_ed = soft_max_regression(X + (epsilon * d), W, C)
    O_e = np.linalg.norm(f_x_ed - f_x)
    O_e_square = np.linalg.norm(f_x_ed - f_x - epsilon + np.transpose(d) * df_x_ed)
    return O_e, O_e_square

