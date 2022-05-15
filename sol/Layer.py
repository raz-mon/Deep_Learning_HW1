from activation_functions import ReLU, Identity
import numpy as np
from util import etta


class Layer:
    """A layer in a regular (feedforward) Neural Network."""

    def __init__(self, X, W, b, activation=ReLU()):
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

    def set_W(self, val):
        self.W = val

    def set_b(self, val):
        self.b = val

    def is_resnet(self):
        return False

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
        act_deriv = self.activation.der(self.W @ self.X + self.b) * V
        self.grad_X = self.W.T @ (act_deriv)
        self.grad_W = (act_deriv) @ self.X.T
        self.grad_b = np.sum(act_deriv, axis=1).reshape(-1, 1)
        return self.grad_X


class SoftmaxLayer(Layer):
    """The last layer of a classifying Neural Network, implementing the soft-max function"""

    def __init__(self, X, W, b, C, activation=Identity()):
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

        super().__init__(X, W, b, activation)
        self.C = C

    # Inherits 'forward'.

    def calc_grad(self, V=None):
        expr = (self.W @ self.X + self.b).T
        arg = expr - etta(expr)
        prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
        m = len(self.X.T)
        # F = - (1 / m) * np.sum(self.C * np.log(prob))
        self.grad_W = (1 / m) * (self.X @ (prob - self.C)).T
        self.grad_X = (1 / m) * (self.W.T @ (prob - self.C).T)
        self.grad_b = (1 / m) * np.sum((prob - self.C).T, axis=1).reshape(-1, 1)
        return self.grad_X

    def calc_loss_probs(self):
        # Assign whole batch to the object fields.
        expr = (self.W @ self.X + self.b).T
        arg = expr - etta(expr)
        prob = np.exp(arg) / np.sum(np.exp(arg), axis=1).reshape(-1, 1)
        m = len(self.X.T)
        F = - (1 / m) * np.sum(self.C * np.log(prob))
        return F, prob


class ResNetLayer:
    """A layer in a Residual Neural Network"""

    def __init__(self, X, W1, W2, b, activation=ReLU()):
        self.X = X
        self.W1 = W1
        self.W2 = W2
        self.b = b
        self.grad_X = None
        self.grad_W1 = None
        self.grad_W2 = None
        self.grad_b = None
        self.activation = activation

    def set_W1(self, val):
        self.W1 = val

    def set_W2(self, val):
        self.W2 = val

    def set_b(self, val):
        self.b = val

    def is_resnet(self):
        return True

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
        return self.X + self.W2 @ self.activation.act(self.W1 @ self.X + self.b)

    def calc_grad(self, V):
        """
        Calculate the gradient of the loss function w.r.t the layers parameters ({theta_l}={W_l, b_l}).
        :param V: Previous derivative w.r.t x (of the former layer param grad calculated, which is the 'next'
        layer in the net.
        :type V:
        :return:
        :rtype:
        """
        act_deriv = self.activation.der(self.W1 @ self.X + self.b) * (self.W2.T @ V)
        self.grad_X = V + self.W1.T @ (act_deriv)
        self.grad_W1 = act_deriv @ self.X.T
        self.grad_W2 = V @ (self.activation.act(self.W1 @ self.X + self.b)).T
        self.grad_b = np.sum(act_deriv, axis=1).reshape(-1, 1)
        return self.grad_X