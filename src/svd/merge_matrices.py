import sys
import json
import time
import numpy as np
import argparse

from operator import itemgetter
from scipy.sparse import csr_matrix

from learning_corpus import LearningCorpus
from sparse_matrix import *

def read_matrix_files(dictFile, matrixFile):

    dictFile = open(args.dictfile1, 'r')
    colCutoff, rowCutoff, windowSize = dictFile.readline().strip().split()
    rowWordsDict = json.loads(dictFile.readline())
    colWordsDict = json.loads(dictFile.readline())
    rowSum = json.loads(dictFile.readline())
    colSum = json.loads(dictFile.readline())
    contextMat = load_sparse_matrix(args.matrixfile)

    corpus = LearningCorpus(colCutoff, rowCutoff, windowSize, None, rowWordsDict, colWordsDict, contextMat, rowSum, colSum)
 
    return corpus

def merge_corpora_matrices(corpus1, corpus2, rowCutoff, colCutoff):

    # Generating the merged col features
    extraColWords = set(corpus2.wordFeatures.keys()).difference(corpus1.wordFeatures.keys())
    newColDict = dict(corpus1.wordFeatures)
    beginIndex = len(corpus1.wordFeatures)
    for offset, word in enumerate(extraColWords): newColDict[word] = beginIndex + offset
    
    # Generating the merged row words
    extraRowWords = set(corpus2.vocab.keys()).difference(corpus1.vocab.keys())
    newRowDict = dict(corpus1.vocab)
    beginIndex = len(corpus1.vocab)
    for offset, word in enumerate(extraRowWords): newRowDict[word] = beginIndex + offset

    


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--matrixfile1", type=str, default=None, help="Matrix file name")
    parser.add_argument("-d1", "--dictfile1", type=str, help="Dictionary file name")
    parser.add_argument("-m2", "--matrixfile2", type=str, default=None, help="Matrix file name")
    parser.add_argument("-d2", "--dictfile2", type=str, help="Dictionary file name")
    parser.add_argument("-r", "--rowcutoff", type=int, help="Row freq cutoff")
    parser.add_argument("-c", "--colcutoff", type=int, help="Col freq cutoff")
                        
    args = parser.parse_args()
   
    corpus1 = read_matrix_file(args.dictfile1, args.matrixfile1)
    corpus2 = read_matrix_file(args.dictfile2, args.matrixfile2)
    rowCutoff, colCutoff = (args.rowcutoff, args.colcutoff)    

    newCorpus = merge_corpora_matrices(corpus1, corpus2, rowCutoff, colCutoff)
    newCorpus.save_whole_corpus('merge.dict', 'merge.npy')
