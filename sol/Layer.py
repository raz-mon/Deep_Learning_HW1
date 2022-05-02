from activation_functions import ReLU, Tanh
import numpy as np
from util import etta


class Layer:
    """A layer in a regular (feedforward) Neural Network."""
    def __init__(self, X, W, b, activation=ReLU):
        """
        Initiate the layer.
        :return:
        :rtype:
        """
        self.X = X
        self.W = W
        self.b = b
        self.grad_X = []
        self.grad_W = []
        self.grad_b = []
        self.activation = activation

    def forward_pass(self, prev_out):
        """
        Return the output of this layer.
        :param prev_out: Output of the previous layer.
        :type prev_out:
        :param prev_data: The data from the former layer.
        :type prev_data:
        :return: Output of this layer.
        :rtype:
        """
        self.X = prev_out.copy()
        return self.activation.act(self.W @ prev_out + self.b)

    def calc_grad(self, V):
        """
        Calculate the gradient of the loss function w.r.t the layers parameters ({theta_l}={W_l, b_l}).
        :param V: Previous derivative w.r.t x (of the former layer param grad calculated, which is the 'next'
        layer in the net.
        :type V:
        :return:
        :rtype:
        """
        act_deriv = self.activation.der(self.W @ self.X + self.b)
        self.grad_X = self.W.T @ (act_deriv * V)
        self.grad_W = (act_deriv * V) @ self.X.T
        self.grad_b = np.sum(act_deriv * V, axis=1).reshape(-1, 1)
        return self.grad_X


class SoftmaxLayer(Layer):
    """The last layer of a classifying Neural Network, implementing the soft-max function"""
    def __init__(self, X, W, b, C, activation=ReLU):

        """

        :param self:
        :type self:
        :param W:
        :type W:
        :param b:
        :type b:
        :return:
        :rtype:
        """

        super().__init__(X, W, b)
        self.C = C
        self.activation = lambda x: x

    # Inherits 'forward'.

    def calc_grad(self, V):
        expr = self.W @ self.X + self.b
        arg = expr - etta(expr)
        prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
        m = len(self.X.T)
        # F = - (1 / m) * np.sum(C * np.log(prob))
        self.grad_W = (1 / m) * (self.X @ (prob - self.C))
        self.grad_X = (1 / m) * (self.W @ (prob - self.C).T)
        self.grad_b = (1 / m) * np.sum((prob - self.C), axis=1).reshape(-1, 1)
        return self.grad_X @ V

    def calc_loss(self):
        expr = self.W @ self.X + self.b
        arg = expr - etta(expr)
        prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
        m = len(self.X.T)
        F = - (1 / m) * np.sum(self.C * np.log(prob))
        return F


class ResNetLayer:
    """A layer in a Residual Neural Network"""
    def __init__(self, W1, W2, b1, b2, C, X, W, b):
        # TBD..
        return None











