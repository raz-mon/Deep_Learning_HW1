""" Part 1"""
# Todo: Write code for the loss function and it's gradient with respect to the wieghts and biases.
#  Make sure it's correct with the gradient and Jacobian verification test.

# Todo: Write code for minimizing an objective funciton using SGD.
#   Demonstrate and verify that your optimizer works on a small least squares example
#   (add plots and submit the code itself).

# Todo: Demonstrate the minimization of the softmax function using your SGD variant. Plot
#  a graph of the success percentages of the data classification after each epoch—for both
#  the training data and the validation data. For the plots, you may only use a random
#  subsample of the two data sets (instead of each time computing the loss for the whole
#  data) as these are extra computations that are not related to the optimization itself.
#  Try a few learning rates and min-batch sizes and see how they influence the performance
#  (submit the graphs only for your best option, but also write about your tries). Run as
#  many iterations of SGD as you see that is needed (i.e., more iterations do not improve
#  the accuracy, even if the learning rate decreases).

""" Part 2"""

# Todo: Write the code for the standard neural network.
#  Including the forward pass and backward pass (the computation of the “Jacobian transpose times vector”). See that the
#  Jacobian tests work and submit the tests (using plots, as demonstrated in the course
#  notes). This part should not be overlooked. Remark: for the Jacobian tests use
#  the tanh() activation function, as it is differentiable and will behave properly in the
#  gradient tests. The ReLU function is piecewise linear and non-smooth and may lead
#  to weird-looking gradient tests. After the gradient test passes, you may use either
#  activation functions in the network.

# Todo: Repeat the previous section for the residual neural network.

# Todo: Now, after we’ve verified that all the individual parts are OK, we will examine that
#  their combination together in the whole network also works—the forward pass and
#  the backward pass of the network with L layers (L is a parameter). See that the
#  gradient of the whole network (softmax + layers) passes the gradient test. Submit
#  this verification.
#  In terms of implementation, you may define the weights of the network (the union
#  of the parameters for all the layers) as a list of arrays (different array for each layer)
#  or as one long vector that you slice along the way. Both ways can be used for the
#  perturbation vector that you use in the grad test (you just need to define the inner
#  product correctly). Also - note that since the network ends with the loss (a scalar
#  function) this is a gradient test and not a Jecobian test.

# Todo: Repeat section 2.1.3 for the entire network. Try a few network lengths and see how
#  this influences the performance (write the details of your networks, hyperparameters
#  and experiments). Write your conclusions and demonstrate them.

# Todo: Repeat the previous section, only now use only 200 data points for training (sample
#  them randomly). How do the results change, if at all?
import numpy as np
import util


def main():
    X = np.array([[1, 0, 9, -1, 0], [0, 5, 0, 0, 1], [1, 0, 2, -2, 1], [4, 4, 1, 4, 4], [9, 2, 8, 3, 7]])
    W = np.random.uniform(-1, 1, (5, 3))
    C = np.zeros((5, 3))
    for row in C:
        row[np.random.randint(0, len(row))] = 1
    print(util.gradient_test_X(X, W, C))


if __name__ == "__main__":
    main()



