from NeuralNetwork import NeuralNetwork
from activation_functions import ReLU, Tanh
import numpy as np
import pandas as pd
from pymatreader import read_mat
import matplotlib.pyplot as plt


def print_graph(xs, ys, label, title, x_label, y_label):
    plt.rc("font", size=16, family="Times New Roman")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(xs, ys, label=label)
    ax.set_xlabel(x_label, fontdict={"size": 21})
    ax.set_ylabel(y_label, fontdict={"size": 21})
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()


def initiate_graph(row, col):
    plt.rc("font", size=12, family="Times New Roman")
    figure, axis = plt.subplots(row, col)
    return axis


def plot_multi_graph(axis, row, col, xs, ys, labels, title, x_label, y_label, flag):
    if flag:
        fmt = ["", "b", "g", "r", "c", "m", "y", "k"]
        for i, x in enumerate(xs):
            axis[row].plot(x, ys[i], fmt[i], label=labels[i])
        axis[row].set_xlabel(x_label, fontdict={"size": 12})
        axis[row].set_ylabel(y_label, fontdict={"size": 12})
        axis[row].set_title(title)
        axis[row].grid(True)
        axis[row].legend()
    else:
        fmt = ["", "b", "g", "r", "c", "m", "y", "k"]
        for i, x in enumerate(xs):
            axis[row, col].plot(x, ys[i], fmt[i], label=labels[i])
        axis[row, col].set_xlabel(x_label, fontdict={"size": 12})
        axis[row, col].set_ylabel(y_label, fontdict={"size": 12})
        axis[row, col].set_title(title)
        axis[row, col].grid(True)
        axis[row, col].legend()

mat = read_mat('../Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yt']).to_numpy())
C = pd.DataFrame(mat['Ct']).to_numpy()

# X = X.copy()[:, :1000]
# C = C.copy()[:, :1000]

C = C.T

print(f'X shape: {X.shape}, C shape: {C.shape}')

n = len(X)  # Amount of rows in X, this is the input dimension.
l = len(C.T)  # C = m X l holds the indicators as columns.
m = len(X.T)  # Amount of data-samples.

print(f'n: {n}, l: {l}, m: {m}')
print('X: ', X.shape)
print('C: ', C.shape)

"""
(X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.001,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.002,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.003,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.004,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.005,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
             (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.001,
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.002,
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.003,
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.004,
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C, [n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n, n, n, 2 * n, 3 * n, 4 * n, 3 * n, 2 * n, n], 2, 32, 80, 0.005,
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C,
             [n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 144 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, n], 2,
             77, 80, 0.001, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C,
             [n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 144 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, n], 2,
             77, 80, 0.005, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C,
             [n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 144 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, n], 2,
             77, 80, 0.01, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),
            (X, C,
             [n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, 144 * n, 12 * n, 12 * n, 12 * n, 12 * n, 12 * n, n], 2,
             77, 80, 0.015, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ReLU()),

"""

networks = [(X, C, [n, 3 * n, n], 2, 50, 80, 0.002, [0, 0, 0], Tanh()),
            (X, C, [n, 3 * n, 5 * n, 3 * n, n], 2, 50, 80, 0.002, [0, 0, 0, 0, 0], Tanh()),
            (
            X, C, [n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, n], 2, 50, 80, 0.002, [0, 0, 0, 0, 0, 0, 0, 0], Tanh()),
            (X, C, [n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 3 * n, 3 * n, 3 * n, 3 * n, n], 2, 50, 80, 0.002,
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Tanh()),
            (X, C,
             [n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 5 * n,5 * n,5 * n,5 * n,5 * n, 5 * n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, n],
             2, 50, 80, 0.002, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0, 0, 0, 0], Tanh())
            ]


"""
xs = []
ys_loss = []
ys_probs = []
labels = []
for i in range(1):
    nn = NeuralNetwork(*networks.pop(0))
    loss, probs = nn.train_net()
    print(f'mb size: {nn.mb_size}')
    print(f'loss: {loss}')
    print(f'accuracy: {probs}')
    xs.append([i for i in range(80)])
    ys_loss.append(loss)
    ys_probs.append(probs)
    labels.append(f'learing rate: {nn.lr}')
plot_multi_graph(axis, 0, 0, xs, ys_loss, labels, "Regular Net - Mini-batch size - 32\n different learning rates", "loss", "epoch")
plot_multi_graph(axis, 1, 0, xs, ys_probs, labels, "", "accuracy", "epoch")

xs = []
ys_loss = []
ys_probs = []
labels = []
for i in range(5):
    nn = NeuralNetwork(*networks.pop(0))
    loss, probs = nn.train_net()
    print(f'mb size: {nn.mb_size}')
    print(f'loss: {loss}')
    print(f'accuracy: {probs}')
    xs.append([i for i in range(80)])
    ys_loss.append(loss)
    ys_probs.append(probs)
    labels.append(f'learing rate: {nn.lr}')
plot_multi_graph(axis, 0, 1, xs, ys_loss, labels, "ResNet - Mini-batch size - 32", "loss", "epoch")
plot_multi_graph(axis, 1, 1, xs, ys_probs, labels, "", "accuracy", "epoch")

xs = []
ys_loss = []
ys_probs = []
labels = []
for i in range(4):
    nn = NeuralNetwork(*networks.pop(0))
    loss, probs = nn.train_net()
    print(f'mb size: {nn.mb_size}')
    print(f'loss: {loss}')
    print(f'accuracy: {probs}')
    xs.append([i for i in range(80)])
    ys_loss.append(loss)
    ys_probs.append(probs)
    labels.append(f'learing rate: {nn.lr}')
plot_multi_graph(axis, 0, 2, xs, ys_loss, labels, "Regular Net 2 - Mini-batch size - 77", "loss", "epoch")
plot_multi_graph(axis, 1, 2, xs, ys_probs, labels, "", "accuracy", "epoch")

xs = []
ys_loss = []
ys_probs = []
labels = []
for i in range(5):
    nn = NeuralNetwork(*networks.pop(0))
    loss, probs = nn.train_net()
    print(f'mb size: {nn.mb_size}')
    print(f'loss: {loss}')
    print(f'accuracy: {probs}')
    xs.append([i for i in range(80)])
    ys_loss.append(loss)
    ys_probs.append(probs)
    labels.append(f'learing rate: {nn.lr}')
plot_multi_graph(axis, 0, 0, xs, ys_loss, labels, "Big Net - Mini-batch size - 500", "loss", "epoch")
plot_multi_graph(axis, 1, 0, xs, ys_probs, labels, "", "accuracy", "epoch")
"""
axis = initiate_graph(1, 2)
xs = []
ys_loss = []
ys_probs = []
labels = []
for i in range(5):
    nn = NeuralNetwork(*networks.pop(0))
    loss, probs = nn.train_net()
    print(f'mb size: {nn.mb_size}')
    print(f'loss: {loss}')
    print(f'accuracy: {probs}')
    xs.append([i for i in range(80)])
    ys_loss.append(loss)
    ys_probs.append(probs)
    labels.append(f'Network Length: {len(nn.layers)}')
plot_multi_graph(axis, 0, 0, xs, ys_loss, labels, "Loss - Regular Net - Mini-batch size - 50\nLearning rate - 0.002", "loss", "epoch", 1)
plot_multi_graph(axis, 1, 0, xs, ys_probs, labels, "Accuracy - Regular Net - Mini-batch size - 50\nLearning rate - 0.002", "accuracy", "epoch", 1)

plt.show()