from NeuralNetwork import NeuralNetwork
from activation_functions import ReLU, Tanh
import numpy as np
import pandas as pd
from pymatreader import read_mat
import matplotlib.pyplot as plt

from sol import util


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
    fmt = ["r", "b", "g", "r", "c", "m", "y", "k"]
    if flag:
        for i, x in enumerate(xs):
            axis[row].plot(x, ys[i], fmt[i], label=labels[i])
        axis[row].set_xlabel(x_label, fontdict={"size": 12})
        axis[row].set_ylabel(y_label, fontdict={"size": 12})
        axis[row].set_title(title)
        axis[row].grid(True)
        axis[row].legend()
    else:
        for i, x in enumerate(xs):
            axis[row, col].plot(x, ys[i], fmt[i], label=labels[i])
        axis[row, col].set_xlabel(x_label, fontdict={"size": 12})
        axis[row, col].set_ylabel(y_label, fontdict={"size": 12})
        axis[row, col].set_title(title)
        axis[row, col].grid(True)
        axis[row, col].legend()


def plot_data(xs, ys, title, x_label, y_label):
    indicators = np.argmax(ys, axis=1)
    labels_num = len(ys[0])
    groups = [[[], []] for _ in range(labels_num)]
    for i, d in enumerate(xs.T):
        x, y = d
        groups[indicators[i]][0].append(x)
        groups[indicators[i]][1].append(y)
    for i, g in enumerate(groups):
        plt.scatter(*g)
    plt.xlabel(x_label, fontdict={"size": 12})
    plt.ylabel(y_label, fontdict={"size": 12})
    plt.title(title)
    plt.grid(True)
    plt.show()


data = '../Data/SwissRollData.mat'
data2 = '../Data/PeaksData.mat'
data3 = '../Data/GMMData.mat'
mat = read_mat(data)
X_t = (pd.DataFrame(mat['Yt']).to_numpy())
C_t = pd.DataFrame(mat['Ct']).to_numpy()

X_v = (pd.DataFrame(mat['Yv']).to_numpy())
C_v = pd.DataFrame(mat['Cv']).to_numpy()

# X_t = X_t.copy()[:, :5000]
# C_t = C_t.copy()[:, :5000]

C_t = C_t.T
C_v = C_v.T

print(f'X shape: {X_t.shape}, C shape: {C_t.shape}')

n = len(X_t)  # Amount of rows in X, this is the input dimension.
l = len(C_t.T)  # C = m X l holds the indicators as columns.
m = len(X_t.T)  # Amount of data-samples.

print(f'n: {n}, l: {l}, m: {m}')
print('X: ', X_t.shape)
print('C: ', C_t.shape)

networks = [
    (X_t, C_t, X_v, C_v, [n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 3 * n, 3 * n, 3 * n, 3 * n, l], 2, 32, 80, 0.003,
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Tanh()),
    (X_v, C_v, [n, 12 * n, 144 * n, 144 * n, 144 * n, 12 * n, n], 2, 500, 80, 0.005, [0, 0, 0, 0, 0, 0, 0], Tanh())]


nn = NeuralNetwork(*networks[0])
loss_t, probs_t, loss_v, probs_v = nn.train_net()
xs = [[i for i in range(80)], [i for i in range(80)]]
axis = initiate_graph(1, 2)
plot_multi_graph(axis, 0, 0, xs, [loss_t, loss_v], ["train data", "validation data"], "SwissRoll Data - Loss - train data vs validation data", "epoch", "loss", 1)
plot_multi_graph(axis, 1, 0, xs, [probs_t, probs_v], ["train data", "validation data"], "SwissRoll Data - Accuracy - train data vs validation data", "epoch", "accuracy", 1)
plt.show()

nn.forward_validate()
_, p = nn.calc_loss_probs_validate()

# plot_data(X_v, C_v, "Peaks - Validation - True Value", "X", "Y")
# plot_data(X_v, p, "Peaks - Validation - Network Output", "X", "Y")
# plt.show()