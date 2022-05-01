from NeuralNetwork import NeuralNetwork
from activation_functions import ReLU
import numpy as np
import pandas as pd
from pymatreader import read_mat

mat = read_mat('Data/SwissRollData.mat')
X = pd.DataFrame(mat['Yv']).to_numpy()
C = pd.DataFrame(mat['Cv']).to_numpy()

n = len(X)          # Amount of rows in X, this is the input dimension.
l = len(C[0])       # C = m X l holds the indicators as columns.
m = len(X.T)       # Amount of data-samples.
# W = np.random.uniform(-1, 1, (l, n))        # W = l X n is the weights of the last layer (the softmax layer).

print(f'n: {n}, l: {l}, m: {m}')
print('X: ', X.shape)
print('C: ', C.shape)

nn = NeuralNetwork(X, C, 5, 2, 500, 50, ReLU)




































