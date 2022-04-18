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
