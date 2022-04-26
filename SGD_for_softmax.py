import numpy as np
import util
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt

def SGD_for_Softmax(loss_func, loss_func_grad, X, C, mb_size, max_epochs, lr):
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
    W = np.random.uniform(-5, 5, (n, l))
    print('X: ', X.shape)
    print('W: ', W.shape)
    print('C: ', C.shape)
    for k in range(max_epochs):
        bchs = util.generate_batches(X.T, C, mb_size)
        # Partition the data to random mini-batches of size mb_size.
        for curr_Mb, curr_Ind in bchs:
            # curr_Mb is a matrix of size n X mb_size.
            # curr_Ind is a matrix of size mb_size X l.
            grad = loss_func_grad(curr_Mb, W, curr_Ind)
            W -= lr * grad
        loss.append(loss_func(X, W, C))
    return W, loss

# def SGD_for_Softmax(loss_func, loss_func_grad, X, C, mb_size, max_epochs, lr):
#     """
#     Perform SGD on the on the softmax function.
#     :param loss_func:
#     :type loss_func:
#     :param loss_func_grad:
#     :type loss_func_grad:
#     :param X:
#     :type X:
#     :param C:
#     :type C:
#     :param mb_size:
#     :type mb_size:
#     :param max_epochs:
#     :type max_epochs:
#     :param lr:
#     :type lr:
#     :return:
#     :rtype:
#     """
#
#     m = len(X.T)    # Amount of columns.
#     n = len(X.T[0]) # len of first column.
#     l = len(C.T)    # len of first row.
#     print(f'm: {m}, n: {n}, l: {l}')
#
#     loss = []
#     W = np.random.uniform(0, 1, (n, l))
#
#     for k in range(max_epochs):
#         # Partition the data to random mini-batches of size mb_size.
#         bchs = util.generate_batches(X.T, C, mb_size)
#         num_of_mbs = int(m / mb_size)-1
#         curr_mb_xs = []
#         print(f'num_of_mbs: {num_of_mbs}')
#         for i in range(num_of_mbs):
#             print(f'int epoch {i}')
#             # curr_mb_xs = (np.array(bchs[i][0])).T
#             curr_mb_xs = (np.array(bchs[i][0]))
#             curr_indicator = (np.array(bchs[i][1]))
#             print(f'curr_mb_xs: {len(curr_mb_xs)}, mb_size: {mb_size}')
#             # curr_indicator is a matrix of size mb_size X l.
#             grad = loss_func_grad(curr_mb_xs, W, curr_indicator)
#             W = W - lr * grad
#         loss += [loss_func(X, W, C)]
#     return W, loss


mat = read_mat('Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yt']).to_numpy())
C = (pd.DataFrame(mat['Ct']).to_numpy()).T

print('X: ', X.shape)
print('C: ', C.shape)

mb_size = 500
max_epochs = 50
lr = 0.01
W_, loss = SGD_for_Softmax(util.sm_loss, util.sm_grad_w, X, C, mb_size, max_epochs, lr)

print(loss)

vec = list(range(len(loss)))
plt.rc("font", size=16, family="Times New Roman")
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(vec, loss, label="SoftMax SGD")
ax.set_xlabel("index", fontdict={"size": 21})
ax.set_ylabel("loss", fontdict={"size": 21})
plt.grid(True)
plt.title("SGD Test")
plt.legend()
plt.show()


