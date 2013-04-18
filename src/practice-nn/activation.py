from math import exp
import sys

def sigmoid(val):
    
    return 1./(1+exp(-1.*val))
    
def sigmoid_derivative(val):
    
    sigmoid_val = sigmoid(val)
    return sigmoid_val*(1-sigmoid_val)
    
def tanh(val):
    
    return (math.exp(val)-math.exp(-1*val))/(math.exp(val)+math.exp(-1*val))
    
def tanh_derivative(val):
    
    return 1-tanh(val)**2
    
def sign(val):
    
    if val < 0:
        return -1.
    elif val > 0:
        return 1.
    else:
        return 0.
    