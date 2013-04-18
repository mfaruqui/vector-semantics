import cPickle
import gzip
import os
import sys
import time
import math
import argparse
import numpy

import theano
import theano.tensor as T

from rprop import rprop_plus_updates
from rprop import irprop_minus_updates
from mlp import MLP

from process_parallel_data import get_datasets
        
def shared_dataset(data_x, data_y, borrow=False):
        
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    
    return shared_x, shared_y#T.cast(shared_y, 'int32')
    
def get_number_of_batches(dataset, batch_size):
    
    if len(dataset) % batch_size == 0:
        return len(dataset)/batch_size 
    else:
        return len(dataset)/batch_size + 1
        
def run_mlp(datasets, n_epochs, n_hidden, batch_size, L1_reg, L2_reg, learning_rate=None):
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    
    del datasets
    
    n_out = len(train_set_y[0])
    n_in = len(train_set_x[0])
    
    sys.stderr.write("\nNumber of nodes:-\n  Input: {0}\n  Hidden: {1}\n  Output: {2}\n".format(n_in, n_hidden, n_out))
    sys.stderr.write("\nEpochs: {0}\nBatch Size: {1}\n".format(n_epochs, batch_size))
    sys.stderr.write("\nL1_reg: {0}\nL2_reg: {1}\n".format(L1_reg, L2_reg))
    
    n_train_batches = get_number_of_batches(train_set_x, batch_size)
    n_valid_batches = get_number_of_batches(valid_set_x, batch_size)
    
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    
    sys.stderr.write('... building the model\n')

    #change here possibly
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')    # the data is presented as rasterized images
    #y = T.ivector('y')  # the labels are presented as 1D vector of
    y = T.matrix('y')   # [int] labels
    
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng, input=x, n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    
    cost = classifier.errors(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    if learning_rate is not None:
        # SGD
        sys.stderr.write('\nUsing SGD with learning rate: {0}\n'.format(learning_rate))
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]
    else:
        # RPROP+
        sys.stderr.write('\nUsing RPROP+...\n')
        updates = [(param, param_updated) for (param, param_updated) in rprop_plus_updates(classifier.params, gparams)]
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    sys.stderr.write('... training\n')

    # early-stopping parameters
    patience = 100000  # look as this many examples regardless
    validation_frequency = min(n_train_batches, patience / 2)
    
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print >> sys.stderr, ('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

    end_time = time.clock()
    print >> sys.stderr, (('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
    '''
    hidden_out = classifier.hiddenLayer.throw_output(valid_set_x[0])
    logistic_out = classifier.logRegressionLayer.throw_output(hidden_out)
    logistic_out_func = theano.function(inputs=[], outputs=logistic_out)
    output = logistic_out_func()
    
    for val1, val2 in zip(output, shut_up):
        print val1, val2
    '''

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainfile", type=str, help="Joint parallel file of two languages; sentences separated by |||")
    parser.add_argument("-val", "--valfile", type=str, help="Validation corpus in the same format as training file")
    parser.add_argument("-l1", "--l1reg", type=float, default='0.01', help="Coeff of L1-reg")
    parser.add_argument("-l2", "--l2reg", type=float, default='0.01', help="Coeff of L2-reg")
    parser.add_argument("-n", "--numepoch", type=int, default=10, help="No. of epochs")
    parser.add_argument("-b", "--batchsize", type=int, default=10, help="Batch size")
    parser.add_argument("-hid", "--hiddenlayer", type=int, default=100, help="No. of nodes in hidden layer")
    parser.add_argument("-r", "--learningrate", type=float, default=None, help="Learning rate for backprop")
    
    args = parser.parse_args()
    
    trainFileName = args.trainfile
    valFileName = args.valfile
    L1_reg = args.l1reg
    L2_reg = args.l2reg
    n_epochs = args.numepoch
    batch_size = args.batchsize
    n_hidden = args.hiddenlayer
    learning_rate = args.learningrate
    
    datasets = get_datasets(trainFileName, valFileName)
    run_mlp(datasets, n_epochs, n_hidden, batch_size, L1_reg, L2_reg, learning_rate)