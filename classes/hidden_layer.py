from numpy import asarray, sqrt, zeros
from numpy.random import RandomState
import theano
import theano.tensor as T
from theano.tensor import dmatrix


class HiddenLayer(object):

    def __init__(self, rng: RandomState, inputs: dmatrix,
                 n_in: int, n_out: int, weights=None, biases=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix weights is of shape (n_in, n_out)
        and the bias vector biases is of shape (n_out,).

        NOTE : The non-linearity used here is tanh

        Hidden unit activation is given by: tanh(dot(inputs, weights) + biases)

        :param rng: a random number generator used to initialize weights
        :param inputs: a symbolic tensor of shape (n_examples, n_in)
        :param n_in: dimensionality of inputs
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: non-linearity to be applied in the hidden layer
        """
        self.inputs = inputs
        # `weights` is initialized with `weights_values` which is uniformly sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if weights is None:
            weights_values = asarray(
                rng.uniform(
                    low=-sqrt(6. / (n_in + n_out)),
                    high=sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                weights_values *= 4

            weights = theano.shared(value=weights_values, name='weights', borrow=True)

        if biases is None:
            b_values = zeros((n_out,), dtype=theano.config.floatX)
            biases = theano.shared(value=b_values, name='biases', borrow=True)

        self.weights = weights
        self.biases = biases

        lin_output = T.dot(inputs, self.weights) + self.biases
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.weights, self.biases]
