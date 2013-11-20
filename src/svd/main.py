import sys
import json
import time
import numpy as np
import argparse

from operator import itemgetter
from scipy.sparse import csr_matrix
from math import sqrt

from upper_learning_corpus import LearningCorpus
from sparse_matrix import *
from ranking import *

from sparsesvd import sparsesvd

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", type=str, default=None, help="Raw corpus file name")
    parser.add_argument("-v", "--matrixfile", type=str, default=None, help="Matrix file name")
    parser.add_argument("-d", "--dictfile", type=str, help="Dictionary file name")
    parser.add_argument("-c", "--colcutoff", type=int, default=10, help="Min frequency for a word to be a feature")
    parser.add_argument("-r", "--rowcutoff", type=int, default=0, help="Min frequency for a word to be a row")
    parser.add_argument("-w", "--window", type=int, default=5, help="Size of the word context window (one side)")
    parser.add_argument("-o", "--outputfile", type=str, default='outFile', help="Output file name")
    parser.add_argument("-n", "--numdim", type=int, default=80, help="Num of dimensions required in the SVD")
                        
    args = parser.parse_args()
   
    if args.inputfile is not None:
        
        test = LearningCorpus(args.colcutoff, args.rowcutoff, args.window, args.inputfile)
        dictionaryFileName = args.outputfile+'_w'+str(args.window)+'_r'+str(args.rowcutoff)+'_c'+str(args.colcutoff)+'.dict'
        contextMatFileName = args.outputfile+'_w'+str(args.window)+'_r'+str(args.rowcutoff)+'_c'+str(args.colcutoff)+'.npz'
        test.save_whole_corpus(dictionaryFileName, contextMatFileName)
        sys.exit()

    elif args.matrixfile is not None:

        dictFile = open(args.dictfile, 'r')
        values = dictFile.readline().strip().split()
        if len(values) == 3:
            colCutoff, rowCutoff, windowSize = values
            vocab = json.loads(dictFile.readline())
            wordFeatures = json.loads(dictFile.readline())
            rowSum = json.loads(dictFile.readline())
            colSum = json.loads(dictFile.readline())
            contextMat = load_sparse_matrix(args.matrixfile)
            test = LearningCorpus(colCutoff, rowCutoff, windowSize, None, vocab, wordFeatures, contextMat, rowSum, colSum)
        else:
            colCutoff, windowSize = values
            vocab = json.loads(dictFile.readline())
            wordFeatures = json.loads(dictFile.readline())
            contextMat = load_sparse_matrix(args.matrixfile)
            test = LearningCorpus(colCutoff, rowCutoff, windowSize, None, vocab, wordFeatures, contextMat)
            rowCutoff = 0.

        sys.stderr.write("windowSize: "+str(windowSize)+" colCutoff: "+str(colCutoff)+" rowCutoff: "+str(rowCutoff)+'\n')
        sys.stderr.write("featLen: "+str(test.featLen)+" vocabLen: "+str(test.vocabLen)+'\n')

        del vocab, wordFeatures, contextMat
        sys.stderr.write('Read the matrix!\n')
 
        sys.stderr.write('\nFinding SVD with '+str(args.numdim)+' factors...\n')
        start_time = time.time()
        ut, s, vt = sparsesvd(test.contextMat, args.numdim)
        sys.stderr.write('SVD found, Time taken: '+str(time.time()-start_time)+'\n')

        sys.stderr.write('Calculating word vectors...\n')
        svdContextMat = np.dot(ut.T, np.diag(s))

        sys.stderr.write('Writing down the vectors\n')
        outFile = open(args.outputfile, 'w')
        for word, [wordId, freq] in test.vocab.iteritems():
            try:
                outFile.write(word.encode('utf-8')+' ')
                for val in svdContextMat[wordId]:
                    outFile.write('%.4f' %(val)+' ')
                outFile.write('\n')
            except:
                pass

    else:

        print "Unrecognized format"
        sys.exit()
