import numpy as np
import numpy.linalg as LA
import random
import matplotlib.pyplot as plt
from util import SGD

xs = np.arange(-1, 1, 0.1)
ys = []


def f(x):
    return 5 * (x ** 2)


real_ys = [f(x) for x in xs]
ys_1 = [f(x) + 0.2 for x in xs]
ys_2 = [f(x) - 0.2 for x in xs]

rand_arr = random.sample(range(0, len(xs)), int(len(xs) / 2))
for i in range(len(xs)):
    if i in rand_arr:
        ys += [ys_1[i]]
    else:
        ys += [ys_2[i]]

"""
plt.figure()
plt.scatter(xs, ys)
plt.plot(xs, ys)
plt.show()
"""


"""
# Generate data matrix.
data = np.array(np.array([x**2, x, 1]).T for x in xs)
expectations = np.array(f(x) for x in xs)
print(data[0])
"""

data = [np.transpose([x ** 2, x, 1]) for x in xs]
expectations = [f(x) for x in xs]


def loss(f_, data_, weights):
    # print(list(data), "\nlen:", len(list(data)))
    if len(list(data_)) == 0:
        print(f'now len is 0!!@!$@!$@!$@#$@@#\nHere it is: \n{data_}')
        raise Exception(f"Div by zero dude, this is data:\n{data_}")

    else:
        # return sum([f_(x_i, weights, y_i) for [x_i, y_i] in data_]) / len(list(data_))
        # return sum([f_(d_i[0], weights, d_i[1]) for d_i in list(data_)])#  / len(list(data_))
        # print('data: ', data)
        # print('xs: ', [x[0] for x in list(data)])
        return sum([f_(x_i, weights, y_i) for (x_i, y_i) in data_])#  / len(list(data_))


def lls_func(x, w_, y):
    return (1 / 2) * (np.transpose(x) @ w_ - y) ** 2


# The gradient.
def LLS_grad(x, w_, y):
    return (np.transpose(x) @ w_ - y) * x


# data: [[np.array([x**2, x, 1]).T, expectation (f(x))], ...]
def SGD(mini_loss_func, f_grad, data, expectations, mb_size, max_epochs, lr):
    loss_hist = []
    weights = [1, 1, 1]  # initial weights.
    for k in range(max_epochs):
        # Devide data to mini-batches. TBD
        num_of_mbs = int(len(data) / mb_size)
        for j in range(num_of_mbs):
            x_j = data[j]  # Array of 3 - x^2, X, 1.
            y_j = expectations[j]  # Expectation value.
            grad = f_grad(x_j, weights, y_j)
            weights = weights - lr * grad
        loss_hist += [loss(mini_loss_func, zip(data, expectations), weights)]
    return (weights, loss_hist)


w, l = SGD(lls_func, LLS_grad, data, expectations, 1, 100, 0.01)

print('weights: \n', w)
print('loss array:\n', l)

"""
x = list(range(0, len(xs_)))
plt.title('GD-LLS example')
plt.xlabel('epoch')
plt.ylabel('x value')
plt.plot(x, xs_)
plt.show()
"""
