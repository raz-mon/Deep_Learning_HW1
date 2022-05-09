from NeuralNetwork import NeuralNetwork
from activation_functions import ReLU
import numpy as np
import pandas as pd
from pymatreader import read_mat
import matplotlib.pyplot as plt


def print_loss(xs, ys, label, title):
    plt.rc("font", size=16, family="Times New Roman")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(xs, ys, label=label)
    ax.set_xlabel("epoch", fontdict={"size": 21})
    ax.set_ylabel("loss", fontdict={"size": 21})
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()


mat = read_mat('../Data/SwissRollData.mat')
X = pd.DataFrame(mat['Yt']).to_numpy()[:, :2000]
C = pd.DataFrame(mat['Ct']).to_numpy().T[:, :2000]

print(f'X shape: {X.shape}, C shape: {C.shape}')

n = len(X)  # Amount of rows in X, this is the input dimension.
l = len(C.T)  # C = m X l holds the indicators as columns.
m = len(X.T)  # Amount of data-samples.

print(f'n: {n}, l: {l}, m: {m}')
print('X: ', X.shape)
print('C: ', C.shape)

max_epochs = 80
nn = NeuralNetwork(X=X, C=C, layers_size=[n, 2 * n, 3 * n, 2 * n, n], n_classes=2, mb_size=300, max_epochs=max_epochs,
                   lr=0.005, activation=ReLU())
loss = nn.train_net()
print(f'loss: {loss}')

xs = np.arange(0, max_epochs, 1)
ys = loss
label = "learning rate: 0.005"
title = "Loss as function of epoch"
print_loss(xs, ys, label, title)
