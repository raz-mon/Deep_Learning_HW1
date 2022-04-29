


class SGD:
    """
    Stochastic Gradient Descent algorithm, for the NN (designed especially).
    """

    def __init__(self, lr: float):
        """
        Initialize the SGD optimizer
        :param lr:
        :type lr:
        """
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












