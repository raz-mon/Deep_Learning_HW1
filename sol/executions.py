import util
from NeuralNetwork import NeuralNetwork
import numpy as np
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt

from sol.activation_functions import Tanh


def main():
    data = '../Data/SwissRollData.mat'
    data2 = '../Data/PeaksData.mat'
    data3 = '../Data/GMMData.mat'
    test_sgd_for_softmax(data, 1)


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
            axis[row].plot(x, ys[i], fmt[i], label="learning rate: " + str(labels[i]))
        axis[row].set_xlabel(x_label, fontdict={"size": 12})
        axis[row].set_ylabel(y_label, fontdict={"size": 12})
        axis[row].set_title(title)
        axis[row].grid(True)
        axis[row].legend()
    else:
        for i, x in enumerate(xs):
            axis[row, col].plot(x, ys[i], fmt[i], label=str(labels[i]))
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


def SGD_for_Softmax(X, W, b, C, mb_size, max_epochs, lr):
    """
    :param loss_func: loss function to be evaluated.
    :param loss_func_grad: gradient of loss function.
    :param X: X matrix.
    :param W: W matrix.
    :param b: biases.
    :param C: Indicators matrix.
    :param mb_size: batches size.
    :param max_epochs: Number of epochs.
    :param lr: learning rate.
    :return: the value of W after the GD, and the loss for each epoch.
    """
    loss = []
    prob = []
    for k in range(max_epochs):
        bchs = util.generate_batches(X.T, C, mb_size)
        # Partition the data to random mini-batches of size mb_size.
        for curr_Mb, curr_Ind in bchs:
            # curr_Mb is a matrix of size n X mb_size.
            # curr_Ind is a matrix of size mb_size X l.
            _, grad, _, _ = util.soft_max_regression(curr_Mb, W, curr_Ind, b)
            W -= lr * grad
        l, p = util.calc_probs(X, W, C, b)
        loss += [l]
        prob += [util.get_accuracy(p, C)]
    return W, loss, prob


def test_sgd_for_softmax(data, flag):
    mat = read_mat(data)
    if flag:
        X = (pd.DataFrame(mat['Yv']).to_numpy())
        C = (pd.DataFrame(mat['Cv']).to_numpy()).T
    else:
        X = (pd.DataFrame(mat['Yt']).to_numpy())
        C = (pd.DataFrame(mat['Ct']).to_numpy()).T
    n = len(X)
    l = len(C[0])
    W = np.random.randn(l, n)
    b = np.random.randn(l, 1)

    axis = initiate_graph(1, 2)

    max_epochs = 20
    losses = []
    probs = []
    lrs = [i / 100 for i in range(1, 8)]
    mnb = [i * 100 for i in range(1, 8)]
    epochs = [list(range(max_epochs)) for _ in range(len(lrs))]
    counter = 0
    for lr in lrs:
        print("Epoch Number: % d" % counter)
        counter += 1
        _, loss, prob = SGD_for_Softmax(X, W.copy(), b, C, 30, max_epochs, lr)
        losses.append(loss)
        probs.append(prob)

    plot_multi_graph(axis, 0, 0, epochs, losses, lrs, "SGD Test: Softmax - Validation Data\nDifferent learning rates - Loss vs Epoch",
                     "epoch", "loss", 1)
    plot_multi_graph(axis, 1, 0, epochs, probs, lrs, "SGD Test: Softmax - Validation Data\nDifferent learning rates - Accuracy vs Epoch",
                     "epoch", "accuracy", 1)
    plt.show()


def test_train_network(data):
    # Train the network for different layers lengths.
    mat = read_mat(data)
    X = (pd.DataFrame(mat['Yt']).to_numpy())
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    print(f'X shape: {X.shape}, C shape: {C.shape}')

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.
    m = len(X.T)  # Amount of data-samples.

    networks = [(X, C, X, C, [n, 3 * n, n], 2, 50, 80, 0.002, [0, 0, 0], Tanh()),
                (X, C, X, C, [n, 3 * n, 5 * n, 3 * n, n], 2, 50, 80, 0.002, [0, 0, 0, 0, 0], Tanh()),
                (X, C, X, C, [n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, n], 2, 50, 80, 0.002,
                 [0, 0, 0, 0, 0, 0, 0, 0], Tanh()),
                (X, C, X, C, [n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 3 * n, 3 * n, 3 * n, 3 * n, n], 2, 50, 80,
                 0.002, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Tanh()),
                (X, C, X, C,
                 [n, 3 * n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 5 * n, 5 * n, 5 * n, 5 * n, 5 * n, 5 * n, 3 * n,
                  3 * n, 3 * n, 3 * n, 3 * n, n],
                 2, 50, 80, 0.002, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Tanh())]

    axis = initiate_graph(1, 2)
    xs = []
    ys_loss = []
    ys_probs = []
    labels = []
    for i in range(5):
        nn = NeuralNetwork(*networks.pop(0))
        loss, probs, _, _ = nn.train_net()
        print(f'mb size: {nn.mb_size}')
        print(f'loss: {loss}')
        print(f'accuracy: {probs}')
        xs.append([i for i in range(80)])
        ys_loss.append(loss)
        ys_probs.append(probs)
        labels.append(f'Network Length: {len(nn.layers)}')
    plot_multi_graph(axis, 0, 0, xs, ys_loss, labels,
                     "Loss - Regular Net - Mini-batch size - 50\nLearning rate - 0.002", "loss", "epoch", 1)
    plot_multi_graph(axis, 1, 0, xs, ys_probs, labels,
                     "Accuracy - Regular Net - Mini-batch size - 50\nLearning rate - 0.002", "accuracy", "epoch", 1)

    plt.show()


def validate_accuracy(data):
    # Train the network for different layers lengths.
    mat = read_mat(data)
    X_t = (pd.DataFrame(mat['Yt']).to_numpy())
    C_t = pd.DataFrame(mat['Ct']).to_numpy()

    X_v = (pd.DataFrame(mat['Yv']).to_numpy())
    C_v = pd.DataFrame(mat['Cv']).to_numpy()

    n = len(X_t)  # Amount of rows in X, this is the input dimension.
    l = len(C_t.T)  # C = m X l holds the indicators as columns.

    nn = NeuralNetwork(*(
        X_t, C_t, X_v, C_v, [n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 3 * n, 3 * n, 3 * n, 3 * n, l], 2, 15, 80,
        0.002, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Tanh()))
    loss_t, probs_t, loss_v, probs_v = nn.train_net()
    xs = [[i for i in range(80)], [i for i in range(80)]]
    axis = initiate_graph(1, 2)
    plot_multi_graph(axis, 0, 0, xs, [loss_t, loss_v], ["train data", "validation data"],
                     "Swiss Roll Data - Loss - train data vs validation data\n200 train data points", "epoch", "loss",
                     1)
    plot_multi_graph(axis, 1, 0, xs, [probs_t, probs_v], ["train data", "validation data"],
                     "Swiss Roll Data - Accuracy - train data vs validation data\n200 train data points", "epoch",
                     "accuracy", 1)
    plt.show()


def validate_accuracy_small_data(data):
    # Train the network for different layers lengths.
    mat = read_mat(data)
    X_t = (pd.DataFrame(mat['Yt']).to_numpy())
    C_t = pd.DataFrame(mat['Ct']).to_numpy()

    X_v = (pd.DataFrame(mat['Yv']).to_numpy())
    C_v = pd.DataFrame(mat['Cv']).to_numpy()

    print(f'X shape: {X.shape}, C shape: {C.shape}')

    n = len(X_t)  # Amount of rows in X, this is the input dimension.
    l = len(C_t.T)  # C = m X l holds the indicators as columns.

    bchs = util.generate_batches(X_t.T, C_t, 200)
    new_X, new_Y = bchs[0]

    nn_small = NeuralNetwork(*(
        new_X, new_Y, X_v, C_v, [n, 3 * n, 3 * n, 3 * n, 3 * n, 5 * n, 5 * n, 3 * n, 3 * n, 3 * n, 3 * n, l], 2, 15, 80,
        0.002,
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], Tanh()))
    loss_t, probs_t, loss_v, probs_v = nn_small.train_net()
    xs = [[i for i in range(80)], [i for i in range(80)]]
    axis = initiate_graph(1, 2)
    plot_multi_graph(axis, 0, 0, xs, [loss_t, loss_v], ["train data", "validation data"],
                     "Swiss Roll Data - Loss - train data vs validation data\n200 train data points", "epoch", "loss",
                     1)
    plot_multi_graph(axis, 1, 0, xs, [probs_t, probs_v], ["train data", "validation data"],
                     "Swiss Roll Data - Accuracy - train data vs validation data\n200 train data points", "epoch",
                     "accuracy", 1)
    plt.show()


def test_gradient_test(data):
    mat = read_mat(data)
    X = pd.DataFrame(mat['Yt']).to_numpy()
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.

    n1 = n
    n2 = l
    W1 = np.random.randn(n2, n1)
    b = np.random.randn(n2, 1)

    util.gradient_test(X.copy(), W1.copy(), C.copy(), b.copy(), "W")


def test_resnet_jacobian_test(data):
    mat = read_mat(data)
    X = pd.DataFrame(mat['Yt']).to_numpy()
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.

    n1 = n
    n2 = l
    W1 = np.random.randn(n2, n1)
    W2 = np.random.randn(n1, n2)
    b = np.random.randn(n2, 1)

    util.resnet_jacobian_test(X, W1, W2, b, Tanh(), "W2")


def test_jacobian_test(data):
    mat = read_mat(data)
    X = pd.DataFrame(mat['Yt']).to_numpy()
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.

    n1 = n
    n2 = l
    W1 = np.random.randn(n2, n1)
    b = np.random.randn(n2, 1)

    util.jacobian_test(X.copy(), W1.copy(), b.copy(), Tanh(), "b")


def test_nn_gradient_test(data):
    mat = read_mat(data)
    X = pd.DataFrame(mat['Yt']).to_numpy()
    C = pd.DataFrame(mat['Ct']).to_numpy().T

    n = len(X)  # Amount of rows in X, this is the input dimension.
    l = len(C.T)  # C = m X l holds the indicators as columns.

    max_epochs = 80
    mb_size = 32
    lr = 0.001
    nn = NeuralNetwork(X=X, C=C, X_v=X, C_v=C, layers_size=[n, 2 * n, 3 * n, 2 * n, n], n_classes=2,
                       mb_size=mb_size, max_epochs=max_epochs,
                       lr=lr, activation=Tanh())

    util.nn_gradient_test(nn, X, C)


if __name__ == "__main__":
    main()
