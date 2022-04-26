import numpy as np
import util
from pymatreader import read_mat
import pandas as pd



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







