import numpy as np
import util
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import util
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt


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
        bchs = util.generate_batches(X.T, C, mb_size)
        # Partition the data to random mini-batches of size mb_size.
        for curr_Mb, curr_Ind in bchs:
            # curr_Mb is a matrix of size n X mb_size.
            # curr_Ind is a matrix of size mb_size X l.
            grad = loss_func_grad(curr_Mb, W, curr_Ind)
            W -= lr * grad
        loss.append(loss_func(X, W, C))
    return W, loss


mat = read_mat('Data/SwissRollData.mat')
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
    _, loss = SGD_for_Softmax(util.sm_loss, util.sm_grad_w, X, W.copy(), C, mb_size, max_epochs, lr)
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



x = np.array([[1, 2], [2, 3], [3, 4]])
print(f'x1 stuff')
print('x: ', x)


print('x shape: \n', x.shape)
print('x[0] shape: \n', x[0].shape)
print('x.T: \n', x.T, '\n')
print(f'x sum axis 0: \n{np.sum(x, axis=0)}\n')
print(f'x sum axis 1: \n{np.sum(x, axis=1).reshape(-1,1).shape}\n')


print(f'\n\n\nx2 stuff:')
x2 = np.array([np.array([1, 2]).T, np.array([2, 3]).T, np.array([3, 4]).T])
print('x2: \n', x2)
print('x2 shape: \n', x2.shape)
print('x2.T: \n', x2.T)




