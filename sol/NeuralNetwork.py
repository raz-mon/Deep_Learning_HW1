import numpy as np
import pandas as pd
from Layer import Layer, SoftmaxLayer, ResNetLayer
from activation_functions import ReLU, Tanh, Identity
from util import etta, generate_batches, get_accuracy
from SGD import SGD


class NeuralNetwork:

    def __init__(self, X, C, X_v, C_v, layers_size, n_classes, mb_size, max_epochs, lr, res_array=None, activation=ReLU()):
        """
        Initialize the neural net.
        :param mb_size:
        :type mb_size:
        :param max_epochs:
        :type max_epochs:
        :param n_classes: The amount of classes of this classifier.
        :type n_classes: int
        """
        if res_array is None:
            res_array = []
        self.X = X
        self.C = C
        self.X_v = X_v
        self.C_v = C_v
        self.lr = lr
        self.mb_size = mb_size
        self.max_epochs = max_epochs
        self.layers = []
        num_layers = len(layers_size)

        # Initiate all layers but first and last (they are initiated separately).
        for l in range(1, num_layers - 1):
            if res_array and res_array[l]:
                self.create_res_layer(layers_size[l + 1], layers_size[l], l, activation)
            else:
                self.create_regular_layer(layers_size[l + 1], layers_size[l], l, activation)
        print(f'Initializing first layer (input). Size: {len(X)} neurons')
        input_layer = Layer(X,
                            np.random.randn(layers_size[1], layers_size[0]),
                            np.random.randn(layers_size[1]).reshape(-1, 1),
                            activation)
        print(f'Initializing last layer - Softmax. Size: {layers_size[-1]} neurons')
        last_layer = SoftmaxLayer(np.zeros(layers_size[-1]),
                                  np.random.randn(n_classes, layers_size[-1]),
                                  np.random.randn(n_classes).reshape(-1, 1),
                                  C,
                                  Identity())
        self.layers = np.concatenate([[input_layer], self.layers, [last_layer]])

    def create_regular_layer(self, out_dim, in_dim, index, activation):
        W_l = np.random.randn(
            out_dim, in_dim)  # Some random matrix, with right sizes according to 'layers_size'.
        b_l = np.random.randn(out_dim).reshape(-1, 1)  # Some random matrix, with right sizes
        # according to 'layers_size' (the size of the next layer).
        X_l = np.zeros(in_dim)  # Initiate with something. Actually don't need to. Will update in forward.
        print(f'Initializing layer {index}. Size: {len(X_l)} neurons.')
        self.layers += [Layer(X_l, W_l, b_l, activation)]

    def create_res_layer(self, out_dim, in_dim, index, activation):
        W1_l = np.random.randn(
            out_dim, in_dim)  # Some random matrix, with right sizes according to 'layers_size'.
        W2_l = np.random.randn(
            in_dim, out_dim)  # Some random matrix, with right sizes according to 'layers_size'.
        b_l = np.random.randn(out_dim).reshape(-1, 1)  # Some random matrix, with right sizes
        # according to 'layers_size' (the size of the next layer).
        X_l = np.zeros(in_dim)  # Initiate with something. Actually don't need to. Will update in forward.
        print(f'Initializing ResNet layer {index}. Size: {len(X_l)} neurons.')
        self.layers += [ResNetLayer(X_l, W1_l, W2_l, b_l, activation)]

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

    def forward_validate(self):
        """
        Perform forward steps on all the data. Practically - calculate x_i, of layer i
        :return:
        :rtype:
        """
        # Note: Should we copy X here, or take it as it is?

        # Perform the calculation
        prev_output = self.X_v
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

    def calc_loss_probs_validate(self):
        self.layers[-1].C = self.C_v
        """
        Calculate the current loss.
        :return: Current loss function value.
        :rtype: float
        """
        return self.layers[-1].calc_loss_probs()

    def train_net(self):
        sgd = SGD(self, self.lr)
        loss = []
        prob = []
        loss_v = []
        prob_v = []
        m = len(self.X[0])
        for epoch in range(self.max_epochs):
            # Partition batch into mini-batches.
            batches = generate_batches(self.X.T, self.C, self.mb_size)
            # For each mini-batch:
            acc = 0
            curr_loss = 0
            for curr_Mb, curr_Indicator in batches:
                # Calculate forward-pass -->
                self.forward(curr_Mb)
                # Calculate backward-pass -->
                self.backward(curr_Indicator)
                l, p = self.calc_loss_probs()
                acc += get_accuracy(p, curr_Indicator)
                curr_loss += l
                # Perform SGD step.
                sgd.one_step()
            loss += [curr_loss * (self.mb_size / m)]
            prob += [acc * (self.mb_size / m)]
            self.forward_validate()
            l_v, p_v = self.calc_loss_probs_validate()
            loss_v += [l_v]
            prob_v += [get_accuracy(p_v, self.C_v)]
            print("Epoch Number: %0.f,    loss: %f,   accuracy: %f" % (epoch, loss[epoch], prob[epoch]))
        return loss, prob, loss_v, prob_v
