import numpy as np
import util
from pymatreader import read_mat
import pandas as pd


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

    m = len(X)
    n = len(X[0])
    l = len(C)

    loss = []
    W = np.random.uniform(0, 1, (n, l))

    for k in range(max_epochs):
        # Partition the data to random mini-batches of size mb_size.
        mbs, indicators = util.generate_batchs(X, C, mb_size)
        num_of_mbs = int(m / mb_size)
        for i in range(num_of_mbs):
            curr_mb_xs = mbs[i]
            curr_indicator = indicators[i]
            # curr_indicator is a matrix of size mb_size X l.
            grad = loss_func_grad(curr_mb_xs, W, curr_indicator)
            W = W - lr * grad
        loss += [loss_func(X, W, C)]
    return W, loss

mat = read_mat('Data/SwissRollData.mat')
X = pd.DataFrame(mat['Yt']).to_numpy()
C = (pd.DataFrame(mat['Ct']).to_numpy()).T
mb_size = 15
max_epochs = 10
lr = 0.01
SGD_for_Softmax(util.sm_loss, util.sm_grad_w, X, C, mb_size, max_epochs, lr)













