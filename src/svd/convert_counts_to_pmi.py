import sys
import json
import time
import numpy as np
import argparse

from operator import itemgetter
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import dok_matrix
from math import sqrt
from math import log

from upper_learning_corpus import LearningCorpus
from sparse_matrix import *
from ranking import *

def convert_counts_to_pmi2(matrix, rowSum, colSum):

    totalSum = sum(rowSum.values())
    sys.stderr.write('Converting to csc_matrix format... ')
    startTime = time.time()
    matrix = coo_matrix(matrix)
    sys.stderr.write('done. Time taken: '+str(time.time()-startTime)+' secs\n')
    totalEntries = len(matrix.row)
    sys.stderr.write('Num entries: '+str(totalEntries)+'\n')

    numEntries = 1.
    # symmetric matrix
    for r, c, val in zip(np.nditer(matrix.row), np.nditer(matrix.col), np.nditer(matrix.data, op_flags=['readwrite'])):
        pi, pj, pij = (1.*val/rowSum[str(r)], 1.*val/colSum[str(c)], 1.*val/totalSum)
        val[...] = log(pij/(pi*pj))

        if numEntries% 1000000 == 0: sys.stderr.write(str(numEntries)+' ')
        numEntries += 1

    sys.stderr.write('done!\n')
    return csc_matrix((matrix.data, (matrix.row, matrix.col)), shape=matrix.shape)

def convert_counts_to_pmi(matrix, rowSum, colSum):

    totalSum = sum(rowSum.values())
    sys.stderr.write('Converting to dok_matrix format... ')
    startTime = time.time()
    matrix = dok_matrix(matrix)
    sys.stderr.write('done. Time taken: '+str(time.time()-startTime)+' secs\n')
    totalEntries = len(matrix)
    sys.stderr.write('Num entries: '+str(totalEntries)+'\n')

    r, c = matrix.shape
    numEntries = 1.
    # symmetric matrix
    if r == c:
        for key, val in matrix.iteritems():
            i, j = key
            i, j = (str(i), str(j))
            if int(i) <= int(j):
                pi, pj, pij = (1.*val/rowSum[i], 1.*val/colSum[j], 1.*val/totalSum)
                pmi = log(pij/(pi*pj)) 
                matrix[int(i), int(j)] = pmi
                matrix[int(j), int(i)] = pmi
            else:
                pass
 
            if numEntries% 1000000 == 0: sys.stderr.write(str(numEntries)+' ')
            numEntries += 1
    else:
        for key, val in matrix.iteritems():
            i, j = key
            i, j = (str(i), str(j))
            pi, pj, pij = (1.*val/rowSum[i], 1.*val/colSum[j], 1.*val/totalSum)
            matrix[int(i), int(j)] = log(pij/(pi*pj))

            if numEntries% 1000000 == 0: sys.stderr.write(str(numEntries)+' ')
            numEntries += 1

    sys.stderr.write('done!\n')
    return csc_matrix(matrix)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--matrixfile", type=str, default=None, help="Matrix file name")
    parser.add_argument("-d", "--dictfile", type=str, help="Dictionary file name")
    parser.add_argument("-o", "--outputfile", type=str, default=None, help="Output file name")
                        
    args = parser.parse_args()
   
    outFileName = args.outputfile
    dictFile = open(args.dictfile, 'r')
    values = dictFile.readline().strip().split()
    if len(values) == 3:
        colCutoff, rowCutoff, windowSize = values
    else:
        colCutoff, windowSize = values
        rowCutoff = 0.

    vocab = json.loads(dictFile.readline())
    wordFeatures = json.loads(dictFile.readline())
    rowSum = json.loads(dictFile.readline())
    colSum = json.loads(dictFile.readline())
    contextMat = load_sparse_matrix(args.matrixfile)
        
    sys.stderr.write("windowSize: "+str(windowSize)+" colCutoff: "+str(colCutoff)+" rowCutoff: "+str(rowCutoff)+'\n')
    sys.stderr.write("featLen: "+str(len(wordFeatures))+" vocabLen: "+str(len(vocab))+'\n')
    sys.stderr.write('Read the matrix!\n')

    ''' Convert the matrix here! '''
    contextMat = convert_counts_to_pmi(contextMat, rowSum, colSum)
    
    sys.stderr.write('Writing the matrix now... ')
    if outFileName is None: outFileName = args.dictfile.replace('.dict', '_pmi')
    save_sparse_matrix(outFileName, contextMat)
    sys.stderr.write('done!\n')
