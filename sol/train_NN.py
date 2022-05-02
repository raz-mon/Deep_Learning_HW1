from NeuralNetwork import NeuralNetwork
from activation_functions import ReLU
import numpy as np
import pandas as pd
from pymatreader import read_mat

mat = read_mat('../Data/SwissRollData.mat')
X = pd.DataFrame(mat['Yt']).to_numpy()
C = pd.DataFrame(mat['Ct']).to_numpy().T

print(f'X shape: {X.shape}, C shape: {C.shape}')

n = len(X)          # Amount of rows in X, this is the input dimension.
l = len(C.T)       # C = m X l holds the indicators as columns.
m = len(X.T)       # Amount of data-samples.

print(f'n: {n}, l: {l}, m: {m}')
print('X: ', X.shape)
print('C: ', C.shape)

nn = NeuralNetwork(X=X, C=C, layers_size=[n, n, n, n, n], n_classes=2, mb_size=500, max_epochs=10, lr=0.01, activation=ReLU)
loss = nn.train_net()
print(loss)




































