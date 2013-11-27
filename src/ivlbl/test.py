from multiprocessing import Pool
import random
import numpypy
import numpy

def change(x):
    
    #x *= 2
    #return x 
    return sum(conv_back(x))

def y():
    pool = Pool(2)
    x, y = ({}, {})
    x = numpy.array([2,3])
    y = numpy.array([-1,3])
    for a in pool.imap(change, [conv_str(x), conv_str(y)]):
        print a
        
def conv_str(x):
    
    return ' '.join([str(val) for val in x])
    
def conv_back(x):
    
    return [float(val) for val in x.split()]
        
if __name__=='__main__':
    y()