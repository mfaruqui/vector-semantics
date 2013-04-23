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
from scipy.sparse import csr_matrix

from rprop import rprop_plus_updates
from rprop import irprop_minus_updates
from mlp import MLP

from process_parallel_data import get_datasets

TR_LOAD_SIZE = 100    
        
def shared_dataset(data_x, data_y, borrow=False):
        
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    
    return T.cast(shared_x, 'int32'), T.cast(shared_y, 'int32')
    
def get_number_of_batches(len_dataset, batch_size):
    
    if len_dataset % batch_size == 0:
        return len_dataset/batch_size 
    else:
        return len_dataset/batch_size + 1
        
def sprase_to_normal_spliced_shared(x, y, index, batch_size):
    
    try:
        spliced_train_set_x = x[index * TR_LOAD_SIZE:(index + 1) * TR_LOAD_SIZE]
    except:
        spliced_train_set_x = x[index * TR_LOAD_SIZE:]
        
    try:
        spliced_train_set_y = y[index * TR_LOAD_SIZE:(index + 1) * TR_LOAD_SIZE]
    except:
        spliced_train_set_y = y[index * TR_LOAD_SIZE:]
    
    spliced_train_set_x = spliced_train_set_x.todense()
    spliced_train_set_y = spliced_train_set_y.todense()
    
    n_train_batches = get_number_of_batches(len(spliced_train_set_x),  batch_size)
    spliced_train_set_x, spliced_train_set_y = shared_dataset(spliced_train_set_x, spliced_train_set_y)
    
    return spliced_train_set_x, spliced_train_set_y, n_train_batches 
        
def run_mlp(datasets, n_epochs, n_hidden, batch_size, L1_reg, L2_reg, learning_rate=None):
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    
    n_in = train_set_x.shape[1]
    n_out = train_set_y.shape[1]
    n_valid_batches = get_number_of_batches(valid_set_x.shape[0], batch_size)
    n_tr_load_batches = get_number_of_batches(train_set_x.shape[0],  TR_LOAD_SIZE)
    
    train_set_x = csr_matrix(train_set_x)
    train_set_y = csr_matrix(train_set_y)
    
    del datasets
    
    sys.stderr.write("\nNumber of nodes:-\n  Input: {0}\n  Hidden: {1}\n  Output: {2}\n".format(n_in, n_hidden, n_out))
    sys.stderr.write("\nEpochs: {0}\nBatch Size: {1}\nLoad size: {2}".format(n_epochs, batch_size, TR_LOAD_SIZE))
    sys.stderr.write("\nL1_reg: {0}\nL2_reg: {1}\n".format(L1_reg, L2_reg))
    
    valid_set_x, valid_set_y = shared_dataset(valid_set_x.todense(), valid_set_y.todense())
    
    sys.stderr.write('... building the model\n')

    # index to a [mini]batch
    index = T.lscalar()
    x = T.imatrix('x')
    y = T.imatrix('y')
    
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
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
                })

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
    
    sys.stderr.write('... training\n')
    
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                      # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                      # considered significant
    validation_frequency = min(get_number_of_batches(TR_LOAD_SIZE, batch_size), patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
    
    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    start_time = time.clock()
    
    epoch = 0
    done_looping = False
    
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        
        for load_tr_index in xrange(n_tr_load_batches):
            
            spliced_train_set_x, spliced_train_set_y, n_train_batches = sprase_to_normal_spliced_shared(\
                                                        train_set_x, train_set_y, load_tr_index, batch_size)
        
            train_model = theano.function(inputs=[index], outputs=cost, updates=updates,
                givens={
                    x: spliced_train_set_x[index*batch_size : (index+1)*batch_size],
                    y: spliced_train_set_y[index*batch_size : (index+1)*batch_size]
                })
        
            for minibatch_index in xrange(n_train_batches):
            
                minibatch_avg_cost = train_model(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index
            
                if (iter + 1) % validation_frequency == 0:
                
                    validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                
                    if learning_rate is not None:
                        print >> sys.stderr, ('epoch %i, minibatch %i/%i, validation cross entropy %f learning_rate %f' %
                         (epoch, minibatch_index + 1, n_train_batches, this_validation_loss, learning_rate))
                    else:
                        print >> sys.stderr, ('epoch %i, minibatch %i/%i, validation cross entropy %f ' %
                         (epoch, minibatch_index + 1, n_train_batches, this_validation_loss))
                     
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter
                
                    if patience <= iter:
                        done_looping = True
                        break
            
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% obtained at iteration %i, epochs ran %i ') %
              (best_validation_loss, best_iter + 1, epoch))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
                          
    return classifier

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--trainfile", type=str, help="Joint parallel file of two languages; sentences separated by |||")
    parser.add_argument("-val", "--valfile", type=str, help="Validation corpus in the same format as training file")
    parser.add_argument("-l1", "--l1reg", type=float, default='0.00', help="Coeff of L1-reg")
    parser.add_argument("-l2", "--l2reg", type=float, default='0.001', help="Coeff of L2-reg")
    parser.add_argument("-n", "--numepoch", type=int, default=10, help="No. of epochs")
    parser.add_argument("-b", "--batchsize", type=int, default=10, help="Batch size")
    parser.add_argument("-hid", "--hiddenlayer", type=int, help="No. of nodes in hidden layers")
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
    
    if learning_rate == 'None':
        outFileName = 'n'+str(n_epochs)+'_b'+str(batch_size)+'_h'+str(n_hidden)
    else:
        outFileName = 'n'+str(n_epochs)+'_b'+str(batch_size)+'_h'+str(n_hidden)+'_r'+str(learning_rate)
    
    classifier = run_mlp(datasets, n_epochs, n_hidden, batch_size, L1_reg, L2_reg, learning_rate)
    classifier.save_model_params(outFileName)