"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
from numpy import inf, mean
from numpy.random import RandomState
import os
import sys
import theano
import theano.tensor as T
import timeit

from classes.multi_layer_perceptron import MultiLayerPerceptron
from io import load_data


def test_mlp(learning_rate: float=0.01, l1_reg: float=0.00, l2_reg: float=0.0001,
             n_epochs: int=1000, dataset: str='mnist.pkl.gz', batch_size: int=20, n_hidden: int=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer perceptron

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used (factor for the stochastic gradient)
    :param l1_reg: L1-norm's weight when added to the cost (see regularization)
    :param l2_reg: L2-norm's weight when added to the cost (see regularization)
    :param n_epochs: maximal number of epochs to run the optimizer
    :param dataset: the path of the MNIST dataset file from
        http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    :param batch_size: the size of a mini-batch
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = RandomState(1234)

    # construct the MLP class
    classifier = MultiLayerPerceptron(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + l1_reg * classifier.L1
        + l2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a mini-batch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list g_params
    g_params = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * g_param)
        for param, g_param in zip(classifier.params, g_params)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience // 2)
        # go through this many mini-batches before checking the network
        # on the validation set; in this case we check every epoch

    best_validation_loss = inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for mini_batch_index in range(n_train_batches):

            mini_batch_avg_cost = train_model(mini_batch_index)
            # iteration number
            iteration = (epoch - 1) * n_train_batches + mini_batch_index

            if (iteration + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = mean(validation_losses)

                print(
                    'epoch %i, mini-batch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        mini_batch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iteration * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iteration

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = mean(test_losses)

                    print(('     epoch %i, mini-batch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, mini_batch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iteration:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file %s ran for %.2fm' %
           (os.path.split(__file__)[1], (end_time - start_time) / 60.)),
          file=sys.stderr)


if __name__ == '__main__':

    test_mlp()
