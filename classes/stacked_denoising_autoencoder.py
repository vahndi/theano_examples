from typing import List

from numpy.random import RandomState
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.type import TensorType

from classes.denoising_autoencoder import DenoisingAutoEncoder
from classes.hidden_layer import HiddenLayer
from classes.logistic_regression import LogisticRegression


class StackedDenoisingAutoEncoder(object):
    """
    Stacked denoising auto-encoder class (StackedDenoisingAutoEncoder)

    A stacked denoising auto-encoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the inputs of
    the dA at layer `i+1`. The first layer dA gets as inputs the inputs of
    the StackedDenoisingAutoEncoder, and the hidden layer of the last dA represents the output.
    Note that after pre-training, the StackedDenoisingAutoEncoder is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """
    def __init__(
        self,
        numpy_rng: RandomState,
        theano_rng: RandomStreams=None,
        n_ins: int=784,
        hidden_layers_sizes: List[int]=[500, 500],
        n_outs: int=10,
        corruption_levels: List[float]=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :param numpy_rng: numpy random number generator used to draw initial weights
        :param theano_rng: Theano random generator; if None is given one is generated
            based on a seed drawn from `rng`
        :param n_ins: dimension of the inputs to the sdA
        :param hidden_layers_sizes: intermediate layers size, must contain at least one value
        :param n_outs: dimension of the output of the network
        :param corruption_levels: amount of corruption to use for each layer
        """
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

        # The StackedDenoisingAutoEncoder is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising auto-encoders
        # We will first construct the StackedDenoisingAutoEncoder as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising auto-encoder that shares weights with that layer
        # During pre-training we will train these auto-encoders (which will
        # lead to changing the weights of the MLP as well)
        # During fine-tuning we will finish training the StackedDenoisingAutoEncoder by doing
        # stochastic gradient descent on the MLP

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the inputs is either the number of hidden units of
            # the layer below or the inputs size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the inputs to this layer is either the activation of the hidden
            # layer below or the inputs of the StackedDenoisingAutoEncoder if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng, inputs=layer_input,
                n_in=input_size, n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the StackedDenoisingAutoEncoder
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising auto-encoder that shared weights with this layer
            da_layer = DenoisingAutoEncoder(
                numpy_rng=numpy_rng, theano_rng=theano_rng,
                inputs=layer_input,
                n_visible=input_size, n_hidden=hidden_layers_sizes[i],
                weights=sigmoid_layer.weights, hidden_bias=sigmoid_layer.biases
            )
            self.dA_layers.append(da_layer)
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            inputs=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.fine_tune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # mini-batch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pre_training_functions(self, train_set_x: TensorType, batch_size: int):
        """
        Generates a list of functions, each of them implementing one
        step in training the dA corresponding to the layer with same index.
        The function will require as inputs the mini-batch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all mini-batch indexes.

        :param train_set_x: Shared variable that contains all data-points used
            for training the dA
        :param batch_size: size of a [mini]batch
        """
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a mini-batch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # beginning of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pre_train_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pre_train_fns.append(fn)

        return pre_train_fns

    def build_fine_tune_functions(self, datasets, batch_size, learning_rate):
        """
        Generates a function `train` that implements one step of
        fine-tuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         data-points, the other for the labels

        :type batch_size: int
        :param batch_size: size of a mini-batch

        :type learning_rate: float
        :param learning_rate: learning rate used during fine-tune stage
        """

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of mini-batches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        g_params = T.grad(self.fine_tune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - g_param * learning_rate)
            for param, g_param in zip(self.params, g_params)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.fine_tune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            inputs=[index],
            outputs=self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
