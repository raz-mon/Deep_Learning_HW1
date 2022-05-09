import pandas as pd
from pymatreader import read_mat
from sol.Layer import SoftmaxLayer
from sol.NeuralNetwork import NeuralNetwork
from sol.activation_functions import ReLU, Tanh
import sol.util as util
import numpy as np


def main():
    mat = read_mat('../Data/SwissRollData.mat')
    X = pd.DataFrame(mat['Yt']).to_numpy()
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    print(f'X shape: {X.shape}, C shape: {C.shape}')

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.
    m = len(X.T)  # Amount of data-samples.

    print(f'n: {n}, l: {l}, m: {m}')
    print('X: ', X.shape)
    print('C: ', C.shape)

    n1 = n
    n2 = l
    W1 = np.random.randn(n2, n1)
    W2 = np.random.randn(n1, n2)
    b = np.random.randn(n2, 1)

    util.resnet_jacobian_test(X, W1, W2, b, Tanh(), "W1")
    # util.jacobian_test(X.copy(), W1.copy(), b.copy(), Tanh(), "b")
    # util.gradient_test(X.copy(),W1.copy(),C.copy(),b.copy(),"b")
    """
    max_epochs = 80
    nn = NeuralNetwork(X=X, C=C, layers_size=[n, n, n, n, n], n_classes=2, mb_size=500, max_epochs=max_epochs, lr=0.001,
                       activation=ReLU())
    util.nn_gradient_test(nn, X, C, "W")
    """


if __name__ == "__main__":
    main()
