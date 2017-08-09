import numpy
import theano
import theano.tensor as T
from numpy.random import RandomState
from theano.tensor import TensorType
from theano.tensor.shared_randomstreams import RandomStreams


class DenoisingAutoEncoder(object):
    """
    Denoising Auto-Encoder class (dA)

    A denoising autoencoder tries to reconstruct the inputs from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the inputs space. Please refer to Vincent et al.,2008
    for more details. If x is the inputs then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the inputs into the latent space. Equation (3)
    computes the reconstruction of the inputs, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(weights \tilde{x} + biases)                                           (2)

        x = s(weights' y  + biases')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """
    def __init__(
        self,
        numpy_rng: RandomState, theano_rng=RandomStreams,
        inputs: TensorType=None, n_visible: int=784, n_hidden: int=500,
        weights: TensorType=None, hidden_bias: TensorType=None, visible_bias: TensorType=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the inputs ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the inputs, weights and
        bias. Such a symbolic variables are useful when, for example the inputs
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as inputs the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :param numpy_rng: number random generator used to generate weights
        :param theano_rng: Theano random generator; if None is given one is generated based on a seed
            drawn from `numpy_rng`
        :param inputs: a symbolic description of the inputs or None for standalone dA
        :param n_visible: number of visible units
        :param n_hidden:  number of hidden units
        :param weights: Theano variable pointing to a set of weights that should be
            shared between the dA and another architecture; if dA should
            be standalone set this to None
        :param hidden_bias: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared between dA and another
                     architecture; if dA should be standalone set this to None
        :param visible_bias: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared between dA and another
                     architecture; if dA should be standalone set this to None
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : weights' was written as `W_prime` and biases' as `b_prime`
        if not weights:
            # weights is initialized with `initial_weights` which is uniformly sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runnable on GPU
            initial_weights = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            weights = theano.shared(value=initial_weights, name='weights', borrow=True)

        if not visible_bias:
            visible_bias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not hidden_bias:
            hidden_bias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='biases',
                borrow=True
            )

        self.W = weights
        # biases corresponds to the bias of the hidden
        self.b = hidden_bias
        # b_prime corresponds to the bias of the visible
        self.b_prime = visible_bias
        # tied weights, therefore W_prime is weights transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no inputs is given, generate a variable representing the inputs
        if inputs is None:
            # we use a matrix because we expect a mini-batch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='inputs')
        else:
            self.x = inputs

        self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, inputs, corruption_level):
        """
        This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``corruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the inputs
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.
        """
        return self.theano_rng.binomial(size=inputs.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * inputs

    def get_hidden_values(self, inputs):
        """
        Computes the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(inputs, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed inputs given the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """
        This function computes the cost and the updates for one training step of the dA
        """
        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        mini-batches, L will be a vector, with one entry per
        #        example in mini-batch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the mini-batch. We need to
        #        compute the average of all these to get the cost of
        #        the mini-batch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        g_params = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * g_param)
            for param, g_param in zip(self.params, g_params)
        ]

        return cost, updates
