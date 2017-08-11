"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising auto-encoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An auto-encoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the auto-encoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denoising auto-encoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007
"""
import numpy
import os
import sys
import timeit

from classes.stacked_denoising_autoencoder import StackedDenoisingAutoEncoder
from data import load_data


def test_sda(fine_tune_lr: float=0.1, pre_training_epochs: int=15,
             pre_train_lr: float=0.001, training_epochs: int=1000,
             dataset: str='mnist.pkl.gz', batch_size: int=1):
    """
    Demonstrates how to train and test a stochastic denoising auto-encoder.

    This is demonstrated on MNIST.

    :param fine_tune_lr: learning rate used in the fine-tune stage
        (factor for the stochastic gradient)
    :param pre_training_epochs: number of epochs to do pre-training
    :param training_epochs: number of epochs to do training
    :param pre_train_lr: learning rate to be used during pre-training
    :param dataset: path the the pickled dataset
    :param batch_size: the size of a mini-batch
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of mini-batches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising auto-encoder class
    sda = StackedDenoisingAutoEncoder(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=10
    )
    ##########################
    # PRE-TRAINING THE MODEL #
    ##########################
    print('... getting the pre-training functions')
    pre_training_fns = sda.pre_training_functions(
        train_set_x=train_set_x,
        batch_size=batch_size
    )

    print('... pre-training the model')
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in range(sda.n_layers):
        # go through pre-training epochs
        for epoch in range(pre_training_epochs):
            # go through the training set
            c = []
            for batch_index in range(int(n_train_batches)):
                c.append(pre_training_fns[i](index=batch_index,
                                             corruption=corruption_levels[i],
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
    train_fn, validate_model, test_model = sda.build_fine_tune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=fine_tune_lr
    )

    print('... fine-tuning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is found
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
                print(('epoch %i, mini-batch %i/%i, validation error %f %%' %
                      (epoch, mini_batch_index + 1, n_train_batches,
                       this_validation_loss * 100.)))

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
                    print('\tepoch %i, mini-batch %i/%i, test error of best model %f %%' %
                          (epoch, mini_batch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iteration:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print((
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iteration + 1, test_score * 100.)
    ))
    print(('The training code for file %s ran for %.2fm' %
           (os.path.split(__file__)[1], (end_time - start_time) / 60.)),
          file=sys.stderr)


if __name__ == '__main__':

    test_sda()
