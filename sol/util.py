import random
import functools
import matplotlib.pyplot as plt
import numpy as np
from sol.activation_functions import ActiveFunc, Tanh


def etta(A: np.array):
    """""
    This method calculate the etta vector that required to reduce from A in order to prevent numerical overflow.
    :return etta vector. this vector is the column with the maximal norm from A.
    """""
    etta = np.array([])
    for a in A:
        etta = np.append(etta, max(a))
    return etta.reshape(-1, 1)


def generate_batches(X, C, mb_size):
    """
    :param X: data matrix.
    :param C: indicators matrix.
    :param mb_size: batches size.
    :return: array with all the mini batches and their correspondent indicators.
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


def SGD_for_Softmax(loss_func, loss_func_grad, X, W, b, C, mb_size, max_epochs, lr):
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
    print('X: ', X.shape)
    print('W: ', W.shape)
    print('C: ', C.shape)
    for k in range(max_epochs):
        bchs = generate_batches(X.T, C, mb_size)
        # Partition the data to random mini-batches of size mb_size.
        for curr_Mb, curr_Ind in bchs:
            # curr_Mb is a matrix of size n X mb_size.
            # curr_Ind is a matrix of size mb_size X l.
            grad = loss_func_grad(curr_Mb, W, b, curr_Ind)
            W -= lr * grad
        loss.append(loss_func(X, W, C))
    return W, loss


def soft_max_regression(X: np.array, W: np.array, C: np.array, b: np.array):
    """""
    Computing the loss function 'Soft-Max regression'.
    :param X. The data input as a matrix of size nXm
    :param W. The weights, size of lXn (where l is the amount of labels)
    :param C. Indicators matrix. size of mXl.
    :return the loss function, and the gradients with respect to X,W.
    """""
    expr = (W @ X + b).T  # m X l
    arg = expr - etta(expr)  # m X l
    prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
    m = len(X.T)
    F = - (1 / m) * np.sum(C * np.log(prob))
    grad_W = (1 / m) * (X @ (prob - C)).T
    grad_X = (1 / m) * (W.T @ (prob - C).T)
    grad_b = (1 / m) * np.sum((prob - C).T, axis=1).reshape(-1, 1)
    return F, grad_W, grad_X, grad_b


def calc_probs(X, W, C, b):
    expr = (W @ X + b).T
    arg = expr - etta(expr)
    prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
    m = len(X.T)
    F = - (1 / m) * np.sum(C * np.log(prob))
    return F, prob


def get_accuracy(Prob: np.array, Indicator: np.array):
    """
    Get the probabilities matrix and indicator matrix and calculate the accuracy
    :param Prob:
    :type Prob:
    :param C:
    :type C:
    :return:
    :rtype:
    """
    size = len(Prob)
    probs = np.argmax(Prob, axis=1)
    indicators = np.argmax(Indicator, axis=1)
    counter = sum(probs == indicators)
    return counter / size


def nn_gradient_test(nn, X, C):
    """
    :param nn: The Neural Network.
    :param X: Data matrix.
    :param C: Indicators matrix.
    :return: matplotlib graph which shows the gradiant test.
    """
    Ws = []
    dWs = []
    bs = []
    dbs = []

    nn.forward(X)
    nn.backward(C)
    for layer in nn.layers:
        Ws.append(layer.W.copy())
        dWs.append(layer.grad_W.copy())
        bs.append(layer.b.copy())
        dbs.append(layer.grad_b.copy())
    f_x, _ = nn.calc_loss_probs()

    ds_W = [np.random.randn(*w.shape) for w in Ws]
    ds_b = [np.random.randn(*b.shape) for b in bs]

    err_1 = []
    err_2 = []
    ks = []

    sum_grad = 0
    for i, layer in enumerate(nn.layers):
        sum_grad += (ds_W[i].reshape(1, -1) @ layer.grad_W.reshape(-1, 1))[0][0]
    for i, layer in enumerate(nn.layers):
        sum_grad += (ds_b[i].reshape(1, -1) @ layer.grad_b.reshape(-1, 1))[0][0]

    for k in range(1, 10):
        epsilon = 0.5 ** k
        for i, layer in enumerate(nn.layers):
            layer.set_W(Ws[i] + epsilon * ds_W[i])
        for i, layer in enumerate(nn.layers):
            layer.set_b(bs[i] + epsilon * ds_b[i])
        nn.forward(X)
        f_x_d, _ = nn.calc_loss_probs()

        err_1.append(abs(f_x_d - f_x))
        err_2.append(abs(f_x_d - f_x - epsilon * sum_grad))
        ks.append(k)
    """
        for i, layer in enumerate(nn.layers[:-1]):
        print("layer number jacobian test: " + str(i))
        jacobian_test(layer.X, layer.W, layer.b, Tanh(), "W")
        jacobian_test(layer.X, layer.W, layer.b, Tanh(), "b")
    gradient_test(nn.layers[-1].X, nn.layers[-1].W, nn.layers[-1].C, nn.layers[-1].b, "W")
    """

    print("last layer gradient test")
    print_test(ks, err_1, err_2, "Network Gradiant Test")


def gradient_test(X: np.array, W: np.array, C: np.array, b: np.array, policy):
    """""
    Gradient test with respect for W.
    :param X matrix.
    :param W matrix.
    :param C matrix.
    :param b matrix.
    :param policy. Indicates to which parameter we are doing the test.
    :return matplotlib graph which shows the gradiant test.
    """""
    dict_num = {"W": 1, "X": 2, "b": 3}
    dict_param = {"W": W, "X": X, "b": b}
    dict_name = {"W": "$\delta W$", "X": "$\delta X$", "b": "$\delta b$"}
    V = np.random.rand(dict_param[policy].shape[0], dict_param[policy].shape[1])
    d = (V / np.linalg.norm(V))
    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    params = soft_max_regression(X, W, C, b)
    grad = params[dict_num[policy]].reshape(-1, 1)
    for k in range(1, 20):
        epsilon = 0.5 ** k
        new = dict_param[policy] + epsilon * d
        dict_args = {"W": (X, new, C, b), "X": (new, W, C, b), "b": (X, W, C, new)}
        f_x_d, _, _, _ = soft_max_regression(*dict_args[policy])
        err_1.append(abs(f_x_d - params[0]))
        err_2.append(abs(f_x_d - params[0] - (epsilon * d_vector.T @ grad)[0][0]))
        ks.append(k)
    print_test(ks, err_1, err_2, "Gradiant Test: " + dict_name[policy])


def jacobian_test(X: np.array, W: np.array, b: np.array, active_func: ActiveFunc, policy):
    """""
    Gradient test with respect for W.
    :param X matrix.
    :param W matrix.
    :param b matrix.
    :param active_func. The activation function (can be Relu or Tanh).
    :param policy. Indicates to which parameter we are doing the test.
    :return matplotlib graph which shows the jacobian test.
    """""
    dict_param = {"W": W, "X": X, "b": b}
    dict_name = {"W": "$\delta W$", "X": "$\delta X$", "b": "$\delta b$"}
    d = np.random.randn(*dict_param[policy].shape)

    V = np.random.randn(W.shape[0], X.shape[1])

    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    g_x = sum([arr[i] for i, arr in enumerate(active_func.activ(W @ X + b) @ V.T)])
    dict_args = {"W": lambda: (W + epsilon * d) @ X + b, "X": lambda: W @ (X + epsilon * d) + b,
                 "b": lambda: W @ X + (b + epsilon * d)}
    for k in range(1, 20):
        epsilon = 0.5 ** k
        jac_v = JacTMV(X, W, b, V, active_func, policy).reshape(-1, 1)
        g_x_d = sum([arr[i] for i, arr in enumerate(active_func.activ(dict_args[policy]()) @ V.T)])
        err_1.append(abs(g_x_d - g_x))
        err_2.append(abs(g_x_d - g_x - (epsilon * d_vector.T @ jac_v)[0][0]))
        ks.append(k)
    print_test(ks, err_1, err_2, "Jacobian Test: " + dict_name[policy])


def resnet_jacobian_test(X: np.array, W1: np.array, W2: np.array, b: np.array, active_func: ActiveFunc, policy):
    """
    :param X: data matrix.
    :param W1: first weights matrix.
    :param W2: second weights matrix.
    :param b: bias matrix.
    :param active_func: activation function.
    :param policy: Indicates to which parameter we are doing the test.
    :return: matplotlib graph which shows the jacobian test.
    """
    dict_param = {"W1": W1, "W2": W2, "X": X, "b": b}
    dict_name = {"W1": "$\delta W_1$", "W2": "$\delta W_2$", "X": "$\delta X$", "b": "$\delta b$"}
    d = np.random.randn(*dict_param[policy].shape)

    V = np.random.randn(W2.shape[0], X.shape[1])

    d_vector = d.reshape(-1, 1)
    err_1 = []
    err_2 = []
    ks = []
    g_x = sum([arr[i] for i, arr in enumerate((X + W2 @ active_func.activ(W1 @ X + b)) @ V.T)])
    dict_args = {"W1": lambda: X + W2 @ active_func.activ((W1 + epsilon * d) @ X + b),
                 "W2": lambda: X + (W2 + epsilon * d) @ active_func.activ(W1 @ X + b),
                 "X": lambda: (X + epsilon * d) + W2 @ active_func.activ(W1 @ (X + epsilon * d) + b),
                 "b": lambda: X + W2 @ active_func.activ(W1 @ X + (b + epsilon * d))}
    for k in range(1, 20):
        epsilon = 0.5 ** k
        jac_v = resnetJacTMV(X, W1, W2, b, V, active_func, policy).reshape(-1, 1)

        g_x_d = sum([arr[i] for i, arr in enumerate(dict_args[policy]() @ V.T)])
        err_1.append(abs(g_x_d - g_x))
        err_2.append(abs(g_x_d - g_x - (epsilon * d_vector.T @ jac_v)[0][0]))
        ks.append(k)
    print_test(ks, err_1, err_2, "Resnet Jacobian Test: " + dict_name[policy])


def JacTMV(X: np.array, W: np.array, b: np.array, V: np.array, active_func: ActiveFunc, policy):
    temp = W @ X + b
    arg = active_func.deriv(temp) * V
    dict_jac = {"W": arg @ X.T, "X": W.T @ arg, "b": np.sum(arg, axis=1).reshape(-1, 1)}
    return dict_jac[policy]


def resnetJacTMV(X: np.array, W1: np.array, W2: np.array, b: np.array, V: np.array, active_func: ActiveFunc, policy):
    arg = active_func.der(W1 @ X + b) * (W2.T @ V)
    dict_jac = {"W1": arg @ X.T, "W2": V @ (active_func.act(W1 @ X + b)).T, "X": V + W1.T @ arg,
                "b": np.sum(arg, axis=1).reshape(-1, 1)}
    return dict_jac[policy]


def print_test(k, err_1, err_2, title):
    """""
    Plot a graph with O(e) and O(e^2) for the gradiant test. 
    """""
    plt.rc("font", size=16, family="Times New Roman")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.semilogy(k, err_1, label="O($\\varepsilon$)")
    ax.semilogy(k, err_2, label="O($\\varepsilon^2$)")
    ax.set_xlabel("k", fontdict={"size": 21})
    ax.set_ylabel("error", fontdict={"size": 21})
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.show()
