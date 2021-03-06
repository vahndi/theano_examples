import numpy
import theano
import theano.tensor as T
from theano.tensor import TensorType


class LogisticRegression(object):
    """
    Multi-class Logistic Regression Class.

    The logistic regression is fully described by a weight matrix :math:`weights`
    and bias vector :math:`biases`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, inputs: TensorType, n_in: int, n_out: int):
        """
        Initialize the parameters of the logistic regression

        :param inputs: symbolic variable that describes the inputs of the architecture (one mini-batch)
        :param n_in: number of inputs units, the dimension of the space in which the data-points lie
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """
        # initialize with 0 the weights weights as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='weights',
            borrow=True
        )
        # initialize the biases biases as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='biases',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # weights is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents inputs training sample-j
        # biases is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(inputs, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model inputs
        self.input = inputs

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{weights,biases\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, weights,biases)) \\
            \ell (\theta=\{weights,biases\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the mini-batch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across mini-batch examples) of the elements in v,
        # i.e., the mean log-likelihood across the mini-batch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y: TensorType):
        """Return a float representing the number of errors in the mini_batch
        over the total number of examples of the mini-batch ; zero one
        loss over the size of the mini-batch

        :param y: corresponds to a vector that gives for each example the correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', type(y), 'y_pred', type(self.y_pred))
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
