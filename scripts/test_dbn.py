import os
import numpy
import sys
import timeit

from classes.deep_belief_network import DeepBeliefNetwork
from data import load_data


def test_dbn(fine_tune_lr: float=0.1, pre_training_epochs: int=100,
             pre_train_lr: float=0.01, k: int=1, training_epochs: int=1000,
             dataset: str='mnist.pkl.gz', batch_size: int=10):
    """
    Demonstrates how to train and test a Deep Belief Network.
    This is demonstrated on MNIST.

    :param fine_tune_lr: learning rate used in the fine-tune stage
    :param pre_training_epochs: number of epoch to do pre-training
    :param pre_train_lr: learning rate to be used during pre-training
    :param k: number of Gibbs steps in CD/PCD
    :param training_epochs: maximal number of iterations ot run the optimizer
    :param dataset: path to the pickled dataset
    :param batch_size: the size of a mini-batch
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    # construct the Deep Belief Network
    dbn = DeepBeliefNetwork(
        numpy_rng=numpy_rng, n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=10
    )

    ##########################
    # PRE-TRAINING THE MODEL #
    ##########################
    print('... getting the pre-training functions')
    pre_training_fns = dbn.pre_training_functions(
        train_set_x=train_set_x, batch_size=batch_size, k=k
    )

    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    for i in range(dbn.n_layers):
        # go through pre-training epochs
        for epoch in range(pre_training_epochs):
            # go through the training set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(pre_training_fns[i](index=batch_index,
                                             lr=pre_train_lr))
            print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
            print(numpy.mean(c))

    end_time = timeit.default_timer()
    print(('The pre-training code for file %s ran for %.2fm' %
           (os.path.split(__file__)[1], (end_time - start_time) / 60.)),
          file=sys.stderr)

    #########################
    # FINE-TUNING THE MODEL #
    #########################

    # get the training, validation and testing function for the model
    print('... getting the fine-tuning functions')
    train_fn, validate_model, test_model = dbn.build_fine_tune_functions(
        datasets=datasets, batch_size=batch_size, learning_rate=fine_tune_lr
    )

    print('... fine-tuning the model')
    # early-stopping parameters
    patience = 4 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.    # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
        # go through this many mini-batches before checking the network
        # on the validation set; in this case we check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for mini_batch_index in range(int(n_train_batches)):
            mini_batch_avg_cost = train_fn(mini_batch_index)
            iteration = (epoch - 1) * n_train_batches + mini_batch_index

            if (iteration + 1) % validation_frequency == 0:

                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print((
                    'epoch %i, mini-batch %i/%i, validation error %f %%'
                    % (
                        epoch,
                        mini_batch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                ))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iteration * patience_increase)
                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iteration = iteration
                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print((('     epoch %i, mini-batch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, mini_batch_index + 1, n_train_batches,
                           test_score * 100.)))

            if patience <= iteration:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print((
        (
            'Optimization complete with best validation score of %f %%, '
            'obtained at iteration %i, '
            'with test performance %f %%'
        ) % (best_validation_loss * 100., best_iteration + 1, test_score * 100.)
    ))
    print(('The fine tuning code for file %s ran for %.2fm' %
           (os.path.split(__file__)[1], (end_time - start_time) / 60.)),
          file=sys.stderr)


if __name__ == '__main__':

    test_dbn()
