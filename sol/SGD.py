from sol.Layer import ResNetLayer


class SGD:
    """
    Stochastic Gradient Descent algorithm, for the NN (designed especially).
    """

    def __init__(self, nn, lr: float):
        """
        Initialize the SGD optimizer
        :param lr:
        :type lr:
        """
        self.nn = nn
        self.lr = lr

    def step(self, curr, grad):
        """
        Perform one step, using the gradient, the learning rate, and the current values.
        :param curr: Current values.
        :type curr: np.array
        :param grad: Gradient of the function.
        :type grad: np.array
        :return: New values of the parameters.
        :rtype: np.array
        """
        return curr - self.lr * grad

    def one_step(self):
        for layer in self.nn.layers:
            if layer.is_resnet():
                layer.set_W1(self.step(layer.W1, layer.grad_W1))
                layer.set_W2(self.step(layer.W2, layer.grad_W2))
                layer.set_b(self.step(layer.b, layer.grad_b))
            else:
                layer.set_W(self.step(layer.W, layer.grad_W))
                layer.set_b(self.step(layer.b, layer.grad_b))
