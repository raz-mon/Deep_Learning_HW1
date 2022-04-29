from activation_functions import ReLU, Tanh

class Layer:

    def __init__(self, X, W, b, activation=ReLU):
        """
        Initiate the layer.
        :return:
        :rtype:
        """
        self.X = X
        self.W = W
        self.b = b
        self.in_dim = W.shape[1]
        self.out_dim = W.shape[0]
        self.grad_X = []
        self.grad_W = []
        self.grad_b = []
        self.activation = activation

    def act(self, prev_data):
        """
        Return the output of this layer.
        :param prev_data: The data from the former layer.
        :type prev_data:
        :return: Output of this layer.
        :rtype:
        """
        return self.activation.act(self.W @ prev_data + self.b)

    def calc_grad(self, prev_dx):
        """
        Calculate the gradient of the loss function w.r.t the layers parameters ({theta_l}={W_l, b_l}).
        :param prev_dx: Previous derivative w.r.t x (of the former layer param grad calculated, which is the 'next'
        layer in the net.
        :type prev_dx:
        :return:
        :rtype:
        """
        # Todo: Update self.grad_X\W\b, and return the gradient w.r.t X of this layer (for the next one).
        return None



def softmax_layer(Layer):
    def __init__(self, W, b):
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
        super.__init__(W, b)

    # Todo: Add functionality special to the layer.



























