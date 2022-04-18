"""
Here we'll put our utility functions (SGD, derivatives etc.).
"""
import numpy as np


def soft_max_regression(X: np.array, W: np.array, C: np.array, j):
    if 0 < j < len(W):
        w_j = W[j]
        X_t = X.transpose()
        X_tW = X_t @ W
        divisor = np.sum(np.exp(X_tW), axis=0)
        return np.sum(C * np.log(np.exp(X_tW) / divisor), axis=0)
    else:
        print("j is out of bound")
