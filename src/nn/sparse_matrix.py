#Implementation of operations with sparse matrices
import numpy as np
from scipy.sparse import coo_matrix
from math import sqrt

def save_sparse_matrix(fileName, x):
    
    x_coo = x.tocoo()
    row = x_coo.row
    col = x_coo.col
    data = x_coo.data
    shape = x_coo.shape
    
    outFile = open(fileName,'w')
    np.savez(outFile, row=row, col=col, data=data, shape=shape)
    outFile.close()
    
    return

def load_sparse_matrix(fileName):
    
    inFile = open(fileName,'r')
    y = np.load(inFile)
    z = coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
    inFile.close()
    
    return z.tocsr()
    
def cosine_similarity(word1Vec, word2Vec):
        
    word1Norm = sqrt(word1Vec.dot(word1Vec.transpose()).todense())
    word2Norm = sqrt(word2Vec.dot(word2Vec.transpose()).todense())
    
    if word1Norm*word2Norm == 0:
        return -1
    else:
        return 1.0*word1Vec.dot(word2Vec.transpose()).todense()/(word1Norm*word2Norm)