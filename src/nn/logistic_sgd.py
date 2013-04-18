import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        else:
            self.W = theano.shared(value=numpy.asarray(W, dtype=theano.config.floatX), name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        else:
            self.b = theano.shared(value=numpy.asarray(b, dtype=theano.config.floatX), name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = self.p_y_given_x

        # parameters of the model
        self.params = [self.W, self.b]
        
    def throw_output(self, inputVector):
        
        return T.nnet.sigmoid(T.dot(inputVector, self.W) + self.b)

    def errors(self, y):
        
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.abs_(y - self.y_pred))
        else:
            return T.mean(T.abs_(y - self.y_pred))