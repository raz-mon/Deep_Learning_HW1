import util_old
import numpy as np
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt




def change_arr(arr):
    arr = np.array([1, 2, 3])


a1 = np.array([1])
change_arr(a1)
print(a1)


"""
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



def SGD_for_Softmax(loss_func, loss_func_grad, X, W, C, mb_size, max_epochs, lr):
    """
    Perform SGD on the on the softmax function.
    :param loss_func:
    :type loss_func:
    :param loss_func_grad:
    :type loss_func_grad:
    :param X:
    :type X:
    :param C:
    :type C:
    :param mb_size:
    :type mb_size:
    :param max_epochs:
    :type max_epochs:
    :param lr:
    :type lr:
    :return:
    :rtype:
    """
    n = len(X)
    l = len(C[0])
    loss = []
    # W = np.random.uniform(-5, 5, (n, l))
    print('X: ', X.shape)
    print('W: ', W.shape)
    print('C: ', C.shape)
    for k in range(max_epochs):
        bchs = util_old.generate_batches(X.T, C, mb_size)
        # Partition the data to random mini-batches of size mb_size.
        for curr_Mb, curr_Ind in bchs:
            # curr_Mb is a matrix of size n X mb_size.
            # curr_Ind is a matrix of size mb_size X l.
            grad = loss_func_grad(curr_Mb, W, curr_Ind)
            W -= lr * grad
        loss.append(loss_func(X, W, C))
    return W, loss


"""
mat = read_mat('../Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yt']).to_numpy())
C = (pd.DataFrame(mat['Ct']).to_numpy()).T
# C = pd.DataFrame(mat['Ct']).to_numpy()

#X = np.random.rand(*X.shape).T
#X /= np.linalg.norm(X)

n = len(X)
l = len(C[0])
W = np.random.uniform(-5, 5, (n, l))

print('X: ', X.shape)
print('C: ', C.shape)

# util.gradient_test_W(X, W, C)

mb_size = 500
max_epochs = 100
losses = []
epochs = list(range(max_epochs))
lrs = [i / 100 for i in range(1, 9)]
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

# util_old.jacobian_test(new_X, W.T, b, act, "W")
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
