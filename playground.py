import numpy as np
import util
from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt

mat = read_mat('Data/SwissRollData.mat')
X = (pd.DataFrame(mat['Yt']).to_numpy())
C = (pd.DataFrame(mat['Ct']).to_numpy()).T
# C = pd.DataFrame(mat['Ct']).to_numpy()

#X = np.random.rand(*X.shape).T
#X /= np.linalg.norm(X)


# util.gradient_test_W(X, W, C)

mb_size = 500
bchs = util.generate_batches(X.T, C, mb_size)
new_X, new_C = bchs[0]

n = len(X)
l = len(C[0])
W = np.random.uniform(-5, 5, (n, l))

print('X: ', X.shape)
print('C: ', C.shape)

util.gradient_test(new_X, W, new_C, None, "X")