import numpy as np
import gzip
import json
import sys
import re
import time

from math import sqrt
from math import log
from operator import itemgetter
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import dok_matrix
from collections import Counter
from sparse_matrix import *

isNumber = re.compile(r'\d+.*')
def normalize_word(word):
    word = word.lower()
    if isNumber.search(word):
       return '---num---'
    else:
       return word

def convert_triangular_dict_to_csc_matrix(contextDict, rowLen, rowSum, diagSum):

    # converting to csr as getrow is much faster than getcol
    startTime = time.time()
    totalEntries = len(contextDict)
    sys.stderr.write("Number entries: "+str(totalEntries)+'\n')
    
    totalSum = 2*sum(contextDict.values()) - diagSum
    assert totalSum > 0
    sys.stderr.write("Total countSum: "+str(totalSum)+" diagSum: "+str(diagSum)+"\n")

    row, col, val = ([], [], [])
    numEntries = 0

    for (i,j), v in contextDict.iteritems():
        row.append(i)
        col.append(j)
        pi, pj, pij = (1.*v/rowSum[i], 1.*v/rowSum[j], 1.*v/totalSum)
        pmi = log(pij/(pi*pj))
        val.append(pmi)

        if i != j:
           row.append(j)
           col.append(i)
           val.append(pmi)
        
        numEntries += 1
        if numEntries % 1000000 == 0: sys.stderr.write(str(numEntries/1000000)+"m ")

    mat = csc_matrix((val, (row, col)), shape=(rowLen, rowLen))
    del row, col, val
    sys.stderr.write('\nTotal time taken in inflating and converting to csc: '+str(time.time()-startTime)+' secs\n')
    return mat

class LearningCorpus:
    
    def __init__(self, freqCutoff, rowCutoff, windowSize, corpusFileName=None, vocab=None, wordFeatures=None, contextMat=None, rowSum=None, colSum=None):
        
        if corpusFileName == None:
            self.vocab = dict(vocab)
            self.vocabLen = len(self.vocab)
            self.wordFeatures = dict(wordFeatures)
            self.featLen = len(self.wordFeatures)
            self.rowSum = dict(rowSum)
            self.colSum = dict(colSum)

            self.windowSize = windowSize
            self.rowCutoff = rowCutoff
            self.contextMat = csc_matrix(contextMat)
        else:
            self.corpusName = corpusFileName
            self.windowSize = windowSize
            self.rowCutoff = rowCutoff
            self.freqCutoffForFeatSelection = freqCutoff
            
            self.vocab = self.read_corpus()
            self.vocabLen = len(self.vocab)
            self.wordFeatures = {word:val[0] for word, val in self.vocab.iteritems()}
            self.featLen = len(self.wordFeatures)
        
            self.contextMat, self.rowSum, self.colSum = self.get_context_vectors()

    def read_corpus(self):
        
        vocab = {}
        wordId = 0
        lineNum = 0
        startTime = time.time()
        if self.corpusName.endswith('.gz'): fileReader = gzip.open(self.corpusName, 'r')
        else: fileReader = open(self.corpusName, 'r')

        for line in fileReader:
            line = unicode(line.strip(), "utf-8")
            for word in line.split():
                word = normalize_word(word)
                if word in vocab:
                    vocab[word][1] += 1
                else:
                    vocab[word] = [wordId, 1]
                    wordId += 1
            
        sys.stderr.write("Time taken to compute vocab (secs): "+str(time.time()-startTime)+"\n")
        sys.stderr.write("VocabLen: "+str(len(vocab))+"\n")

        if self.rowCutoff > 0:
            index = 0
            for word in vocab.keys():
                if vocab[word][1] < self.rowCutoff:
                    del vocab[word]
                else:
                    vocab[word] = [index, vocab[word][1]]
                    index += 1
            sys.stderr.write("\nAfter truncating elements with "+str(self.rowCutoff)+" frequency, numRows: "+str(len(vocab))+"\n")
        
        return vocab
        
    # Will work properly only if window size if lesser than the number of tokens
    def get_context_vectors(self):
        
        window = []
        matrixRowColVal, rowSum = ( Counter(), Counter() )
        firstWindow, wordNum, diagSum = (1, 0, 0)
        sys.stderr.write("Processing words:\n")

        if self.corpusName.endswith('.gz'): fileReader = gzip.open(self.corpusName, 'r')
        else: fileReader = open(self.corpusName, 'r')

        startTime = time.time()
        for word in fileReader.read().strip().split()+['.']:
            word = normalize_word(unicode(word,"utf-8"))

            wordNum += 1
            if wordNum % 50000000 == 0:
                sys.stderr.write(str(wordNum)+' ')

            #collecting words to make the window full
            if len(window) < self.windowSize + 1:
                window.append(word)
    
            #processing the first window
            elif firstWindow == 1:
                firstWindow = 0
                for i in range(0, len(window)-1):
                    if window[i] not in self.vocab: continue
                    for j in range(i+1, len(window)):
                        if window[j] not in self.vocab: continue
                        w_i, w_j = (self.vocab[window[i]][0], self.vocab[window[j]][0])
                        
                        if w_i < w_j: matrixRowColVal[(w_i, w_j)] += 1
                        else: matrixRowColVal[(w_j, w_i)] += 1
                        
                        # although we are computing the upper triangle we still 
                        # compute the whole rowSum (colSum is exactly the same)
                        rowSum[w_i] += 1
                        rowSum[w_j] += 1
                        if w_i == w_j: diagSum += 1

                #remove the first word and insert the new word in the window
                garbage = window.pop(0)
                window.append(word)
                
            #processing all intermediate windows
            else:
                if window[-1] in self.vocab:
                    for i in range(0,len(window)-1):
                        if window[i] not in self.vocab: continue
                        w_i, w_j = (self.vocab[window[i]][0], self.vocab[window[-1]][0])
                        
                        if w_j < w_i: matrixRowColVal[(w_j, w_i)] += 1
                        else: matrixRowColVal[(w_i, w_j)] += 1

                        rowSum[w_i] += 1
                        rowSum[w_j] += 1
                        if w_i == w_j: diagSum += 1
                
                #remove the first word and insert the new word in the window
                garbage = window.pop(0)
                window.append(word)
        
        sys.stderr.write('Created upper triangle. Time taken: '+str(time.time()-startTime)+' secs\nNow inflating... ')
        contextMat = convert_triangular_dict_to_csc_matrix(matrixRowColVal, self.vocabLen, rowSum, diagSum)
        del matrixRowColVal

        # as for a symm matrix sumValRow and sumColRow are same
        return contextMat, rowSum, rowSum

    # Return the ppmi/pmi value as per Turney 2012
    def get_pmi(self, val, sumValRow, sumValCol, totalSum):

        pij, pi, pj = (1.*val/totalSum, 1.*val/sumValRow, 1.*val/sumValCol)
        pmi = log(pij/(pi*pj))
        return pmi

    def save_whole_corpus(self, dictionaryFile, contextMatFile):
 
        dictFile = open(dictionaryFile,'w')
        dictFile.write(str(self.freqCutoffForFeatSelection)+' '+str(self.rowCutoff)+' '+str(self.windowSize)+'\n')
        dictFile.write(json.dumps(self.vocab)+'\n')
        dictFile.write(json.dumps(self.wordFeatures)+'\n')
        dictFile.write(json.dumps(self.rowSum)+'\n')
        dictFile.write(json.dumps(self.colSum)+'\n')
        dictFile.close()

        startTime = time.time()
        save_sparse_matrix(contextMatFile, self.contextMat)
        sys.stderr.write("Time taken in saving the matrix: "+str(time.time()-startTime)+' secs\n')
        
    def get_nth_order_information(self):

        alpha = 0.1
        rows, columns = self.contextMat.shape
        nthContext = lil_matrix(self.contextMat.shape)

        for row in range(rows):
            rowSum = self.contextMat.getrow(row).sum()
            rowArray = np.array(self.contextMat.getrow(row).todense())[0]
            tempSum = np.zeros(rowArray.shape)
            
            iterRow = coo_matrix(self.contextMat.getrow(row))
            for r, c, val in zip(iterRow.row, iterRow.col, iterRow.data):
                tempSum += val*np.array(self.contextMat.getrow(c).todense())[0]
            
            #for i, val in enumerate(rowArray):
            #    if val == 0.: continue
            #    tempSum = tempSum + val*np.array(self.contextMat.getrow(i).todense())
    
            tempSum /= rowSum
            nthContext[row, :tempSum.size] = rowArray + alpha * tempSum

            if row % 1000 == 0: sys.stderr.write(str(row)+'\r')

        self.contextMat = csr_matrix(nthContext)
        del nthContext
