import numpy as np
import pandas as pd
from Layer import Layer, SoftmaxLayer
from activation_functions import ReLU, Tanh, Identity
from util import etta, generate_batches
from SGD import SGD


class NeuralNetwork:

    def __init__(self, X, C, layers_size, n_classes, mb_size, max_epochs, lr, activation=ReLU()):
        """
        Initialize the neural net.
        :param mb_size:
        :type mb_size:
        :param max_epochs:
        :type max_epochs:
        :param n_classes: The amount of classes of this classifier.
        :type n_classes: int
        """
        self.X = X
        self.C = C
        self.lr = lr
        self.mb_size = mb_size
        self.max_epochs = max_epochs
        self.layers = []
        num_layers = len(layers_size)

        # Initiate all layers but first and last (they are initiated separately).
        for l in range(1, num_layers - 1):
            W_l = np.random.uniform(-1, 1, size=(
            layers_size[l + 1], layers_size[l]))  # Some random matrix, with right sizes according to 'layers_size'.
            b_l = np.random.uniform(-1, 1, size=(layers_size[l + 1])).reshape(-1,
                                                                              1)  # Some random matrix, with right sizes according to 'layers_size' (the size of the next layer).
            X_l = np.zeros(layers_size[l])  # Initiate with something. Actually don't need to. Will update in forward.
            print(f'Initializing layer {l}. Size: {len(X_l)} neurons.')
            self.layers += [Layer(X_l, W_l, b_l, ReLU())]
        print(f'Initializing first layer (input). Size: {len(X)} neurons')
        input_layer = Layer(X,
                            np.random.uniform(-1, 1, size=(layers_size[1], layers_size[0])),
                            np.random.uniform(-1, 1, size=(layers_size[1])).reshape(-1, 1),
                            ReLU())
        print(f'Initializing last layer - Softmax. Size: {layers_size[-1]} neurons')
        last_layer = SoftmaxLayer(np.zeros(layers_size[-1]),
                                  np.random.uniform(-1, 1, size=(n_classes, layers_size[-1])),
                                  np.random.uniform(-1, 1, size=n_classes).reshape(-1, 1),
                                  C,
                                  Identity())
        self.layers = np.concatenate([[input_layer], self.layers, [last_layer]])

    def forward(self, X):
        """
        Perform forward steps on all the data. Practically - calculate x_i, of layer i
        :return:
        :rtype:
        """
        # Note: Should we copy X here, or take it as it is?

        # Perform the calculation
        prev_output = X
        for layer in self.layers:
            prev_output = layer.forward_pass(prev_output)
        return prev_output

    def backward(self, C):
        """
        Perform backward propagation, i.e., calculate the gradient of the loss function by the parameters
        and change them accordingly (also with respect to the learning rate).
        Practically - Calculate the gradient of the loss function w.r.t each layer's parameters (W_l, b_l), and update
        these parameters (for each layer).
        :return:
        :rtype:
        """
        # Set Softmax layer's C to the mini-batch C (indicator)
        self.layers[-1].C = C

        V = None
        for layer in self.layers[::-1]:
            V = layer.calc_grad(V)

        # Todo: Now can use SGD to perform the step (very easily, since each layer holds the gradient of the loss
        #  function by its parameters (theta_l = {W_l, b_l}).

    def calc_loss_probs(self):
        """
        Calculate the current loss.
        :return: Current loss function value.
        :rtype: float
        """
        return self.layers[-1].calc_loss_probs()

    def train_net(self):
        sgd = SGD(self.lr)
        loss = []
        prob = []
        for epoch in range(self.max_epochs):
            # Partition batch into mini-batches.
            batches = generate_batches(self.X.T, self.C, self.mb_size)
            # For each mini-batch:
            for curr_Mb, curr_Indicator in batches:
                # Calculate forward-pass -->
                # self.forward(curr_Mb, curr_Indicator.T)
                self.forward(curr_Mb)
                # Calculate backward-pass -->
                self.backward(curr_Indicator)
                # Perform SGD step.
                for layer in self.layers:
                    layer.set_W(sgd.step(layer.W, layer.grad_W))
                    layer.set_b(sgd.step(layer.b, layer.grad_b))
            l, p = self.calc_loss_probs()
            loss += l
            prob += p
        return loss
