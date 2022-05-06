import pandas as pd
from pymatreader import read_mat
from sol.Layer import SoftmaxLayer
from sol.NeuralNetwork import NeuralNetwork
from sol.activation_functions import ReLU
import sol.util as util


def main():
    mat = read_mat('../Data/SwissRollData.mat')
    X = pd.DataFrame(mat['Yt']).to_numpy()
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.
    m = len(X.T)  # Amount of data-samples.

    max_epochs = 80
    nn = NeuralNetwork(X=X, C=C, layers_size=[n, n, n, n, n], n_classes=2, mb_size=500, max_epochs=max_epochs, lr=0.001,
                       activation=ReLU())
    l: SoftmaxLayer = nn.layers[-1]
    util.gradient_test(l.X, l.W, l.C, l.b, "W")


if __name__ == "__main__":
    main()
