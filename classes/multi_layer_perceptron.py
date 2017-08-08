from numpy.random import RandomState
import theano.tensor as T
from theano.tensor import TensorType

from classes.hidden_layer import HiddenLayer
from classes.logistic_regression import LogisticRegression


class MultiLayerPerceptron(object):
    """
    Multi-Layer Perceptron Class

    A multilayer perceptron is a feed-forward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a soft-max layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng: RandomState, input: TensorType, n_in: int, n_hidden: int, n_out: int):
        """
        Initialize the parameters for the multilayer perceptron

        :param rng: a random number generator used to initialize weights
        :param input: symbolic variable that describes the inputs of the architecture (one mini-batch)
        :param n_in: number of inputs units, the dimension of the space in which the data-points lie
        :param n_hidden: number of hidden units
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            inputs=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        # The logistic regression layer gets as inputs the hidden units of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        # square of L2 norm ; one regularization option is to enforce square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        # the parameters of the model are the parameters of the two layer it is made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        # keep track of model inputs
        self.input = input
