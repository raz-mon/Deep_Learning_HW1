import util_old
import numpy as np
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt

"""
f = lambda x: x**2 if x>=0 else 0
arr = np.array([1, 2])
print(np.vectorize(f)(arr))
# f(arr)
f2 = lambda x: np.maximum(0, x)
print(f2(arr))
f3 = lambda x: np.tanh(x)
print(f3(arr))

a = np.array([1,2,3])
print(a[::-1])
"""

"""
a = np.array([1,2,3])
print(np.concatenate([a, np.array([4, 5])]))
"""

"""
A = np.array([[1,2,3], [4,5,6]])
print(f'A: \n{A}')
print(f'np.sum(A, axis=0): \n{np.sum(A, axis=0)}')
print(f'np.sum(A, axis=1): \n{np.sum(A, axis=1).reshape(-1, 1)}')
"""

from sol.activation_functions import Tanh


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
        bchs = util_old.generate_batches(X.T, C, mb_size)
        # Partition the data to random mini-batches of size mb_size.
        for curr_Mb, curr_Ind in bchs:
            # curr_Mb is a matrix of size n X mb_size.
            # curr_Ind is a matrix of size mb_size X l.
            grad = loss_func_grad(curr_Mb, W, b, curr_Ind)
            W -= lr * grad
        loss.append(loss_func(X, W, C))
    return W, loss


mat = read_mat('../Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yt']).to_numpy())
C = (pd.DataFrame(mat['Ct']).to_numpy()).T
# C = pd.DataFrame(mat['Ct']).to_numpy()

# X = np.random.rand(*X.shape).T
# X /= np.linalg.norm(X)

n = len(X)
l = len(C[0])
W = np.random.uniform(-5, 5, (l, n))
b = np.random.uniform(-5, 5, (l, 1))

print('X: ', X.shape)
print('C: ', C.shape)
print('W: ', W.shape)
print('b: ', b.shape)

util_old.gradient_test(X, W, C, b, "W")
"""
mb_size = 500
max_epochs = 100
losses = []
epochs = list(range(max_epochs))
lrs = [i / 100 for i in range(1, 3)]
for lr in lrs:
    _, loss = SGD_for_Softmax(util_old.sm_loss, util_old.sm_grad_w, X, W.copy(), C, mb_size, max_epochs, lr)
    losses.append(loss)

fmt = ["", "b", "g", "r", "c", "m", "y", "k"]
plt.rc("font", size=16, family="Times New Roman")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i, loss in enumerate(losses):
    ax.plot(epochs, loss, fmt[i], label=f"learning rate: {lrs[i]}")

ax.set_xlabel("Epoch number", fontdict={"size": 21})
ax.set_ylabel("Loss", fontdict={"size": 21})
plt.grid(True)
plt.title("SGD Test: Softmax")
plt.legend()
plt.show()
"""
"""










mat = read_mat('../Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yt']).to_numpy())
C = (pd.DataFrame(mat['Ct']).to_numpy()).T
# C = pd.DataFrame(mat['Ct']).to_numpy()

# X = np.random.rand(*X.shape).T
# X /= np.linalg.norm(X)


# util.gradient_test_W(X, W, C)

mb_size = 500
bchs = util_old.generate_batches(X.T, C, mb_size)
new_X, new_C = bchs[0]

n = len(X)
l = len(C[0])
W = np.random.uniform(-5, 5, (n, l))
b = np.random.uniform(-5, 5, (n, 1))
act = Tanh()
print('new_X: ', new_X.shape)
print('W: ', W.shape)
print('b: ', b.shape)

util_old.jacobian_test(new_X, W.T, b, act, "b")
"""

"""

mat = read_mat('../Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yv']).to_numpy())
C = (pd.DataFrame(mat['Cv']).to_numpy()).T
# C = pd.DataFrame(mat['Ct']).to_numpy()

n = len(X)
l = len(C[0])
W = np.random.uniform(-5, 5, (n, l))

mbs_size = [i * 100 for i in range(1, 9)]
max_epochs = 100
losses = []
lr = 0.08
epochs = list(range(max_epochs))
for mb_size in mbs_size:
    _, loss = SGD_for_Softmax(util_old.sm_loss, util_old.sm_grad_w, X, W.copy(), C, mb_size, max_epochs, lr)
    losses.append(loss)

fmt = ["", "b", "g", "r", "c", "m", "y", "k"]
plt.rc("font", size=16, family="Times New Roman")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
for i, loss in enumerate(losses):
    ax.plot(epochs, loss, fmt[i], label=f"mini-batch size: {mbs_size[i]}")

ax.set_xlabel("Epoch number", fontdict={"size": 21})
ax.set_ylabel("Loss", fontdict={"size": 21})
plt.grid(True)
plt.title("SGD Test: Softmax - Validation Data, different mini-batches size")
plt.legend()
plt.show()

"""
