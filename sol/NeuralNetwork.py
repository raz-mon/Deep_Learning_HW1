import numpy as np
import pandas as pd
from Layer import Layer
from activation_functions import ReLU, Tanh


class NeuralNetwork:

    def __init__(self, num_layers: int, layers_size, mb_size, max_epochs, activation=ReLU):
        """
        Initialize the neural net.
        :param num_layers:
        :type num_layers:
        :param mb_size:
        :type mb_size:
        :param max_epochs:
        :type max_epochs:
        """
        self.num_layers = num_layers
        self.mb_size = mb_size
        self.max_epochs = max_epochs
        self.layers = []
        for l in range(num_layers):
            # Todo: Initialize W, b
            #  according to 'layers_size' array (component in layers_size[i] is the amount of neurons in
            #  layer i). Notice normalization (see notes).
            W_l = np.array([])    # Some random matrix, with right sizes according to 'layers_size'.
            b_l = np.array([])    # Some random matrix, with right sizes according to 'layers_size' (the size of the next layer).
            X_l = np.array([])    # Initiate with something.
            print(f'Initializing layer {l}. Size: {len(X_l.T)} neurons.')
            self.layers += [Layer(X_l, W_l, b_l, ReLU)]

    def forward(self, X):
        """
        Perform forward steps on all the data. Practically - calculate x_i, of layer i
        :return:
        :rtype:
        """
        # Note: Should we copy X here, or take it as it is?
        prev_output = X
        for layer in self.layers:
            prev_output = layer.act(prev_output)
        return prev_output

    def calc_grad(self):
        """
        Perform backward propagation, i.e., calculate the gradient of the loss function by the parameters
        and change them accordingly (also with respect to the learning rate).
        Practically - Calculate the gradient of the loss function w.r.t each layer's parameters (W_l, b_l), and update
        these parameters (for each layer).
        :return:
        :rtype:
        """
        prev_dx = None
        for layer in self.layers[::-1]:
            prev_dx = layer.calc_grad(prev_dx)

        # Todo: Now can use SGD to perform the step (very easily, since each layer holds the gradient of the loss function
        #  by its parameters (theta_l = {W_l, b_l}).




































