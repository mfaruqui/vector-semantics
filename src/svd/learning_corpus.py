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
            self.freqCutoffForFeatSelection = freqCutoff
            self.contextMat = csc_matrix(contextMat)
        else:
            self.corpusName = corpusFileName
            self.windowSize = windowSize
            self.rowCutoff = rowCutoff
            self.freqCutoffForFeatSelection = freqCutoff
            
            self.vocab = self.read_corpus()
            self.vocabLen = len(self.vocab)
        
            self.wordFeatures = self.select_feat_words()
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
        
    def select_feat_words(self):
        
        # Remove the top 100 words by frequency from feature list
        top100 = {wordIDAndFreq[0] for wordIDAndFreq in sorted(self.vocab.values(), key=itemgetter(1), reverse=True)[:100]}

        wordFeatures = {}
        index = 0
        for word, [wordId, freq] in self.vocab.iteritems():
            if freq >= self.freqCutoffForFeatSelection and wordId not in top100:
                wordFeatures[word] = index
                index += 1
        
        sys.stderr.write("After truncating elements with "+str(self.freqCutoffForFeatSelection)+" frequency, numCols: "+str(len(wordFeatures))+"\n")
        del top100
        return wordFeatures
    
    # Will work properly only if window size if lesser than the number of tokens
    def get_context_vectors(self):
        
        window = []
        matrixRowColVal = dok_matrix((self.vocabLen, self.featLen))#Counter()
        # sumRow and sumCol for calculating P-PMI if needed
        sumValRow = Counter()
        sumValCol = Counter()
        firstWindow = 1
        wordNum = 0
        sys.stderr.write("Processing words:\n")

        if self.corpusName.endswith('.gz'): fileReader = gzip.open(self.corpusName, 'r')
        else: fileReader = open(self.corpusName, 'r')

        for word in fileReader.read().strip().split()+['.']:
            word = normalize_word(unicode(word,"utf-8"))

            wordNum += 1
            if wordNum % 50000000 == 0:
                sys.stderr.write(str(wordNum)+' ')

            # do not process a word that we do not want!
            if word not in self.vocab: continue
                
            #collecting words to make the window full
            if len(window) < self.windowSize + 1:
                window.append(word)
                    
            #processing the first window
            elif firstWindow == 1:
                firstWindow = 0
                for i in range(0, len(window)):
                    for j in range(0, len(window)):
                        if window[j] in self.wordFeatures and j != i:
                            matrixRowColVal[self.vocab[window[i]][0], self.wordFeatures[window[j]]] += 1
                            sumValRow[self.vocab[window[i]][0]] += 1
                            sumValCol[self.wordFeatures[window[j]]] += 1
                #remove the first word and insert the new word in the window
                garbage = window.pop(0)
                window.append(word)
                
            #processing all intermediate windows
            else:
                for i in range(0,len(window)-1):
                    if window[i] in self.wordFeatures:
                        matrixRowColVal[self.vocab[window[-1]][0], self.wordFeatures[window[i]]] += 1
                        sumValRow[self.vocab[window[-1]][0]] += 1
                        sumValCol[self.wordFeatures[window[i]]] += 1

                    if window[-1] in self.wordFeatures:
                        matrixRowColVal[self.vocab[window[i]][0], self.wordFeatures[window[-1]]] += 1
                        sumValRow[self.vocab[window[i]][0]] += 1
                        sumValCol[self.wordFeatures[window[-1]]] += 1

                #remove the first word and insert the new word in the window
                garbage = window.pop(0)
                window.append(word)
                    
        totalSum = sum(sumValRow.values())
        # The sum of all values in all rows should be equal to sum of all values in all columns
        assert sum(sumValRow.values()) == sum(sumValCol.values())

        contextMat = csc_matrix(matrixRowColVal)
        del matrixRowColVal
        return contextMat, sumValRow, sumValCol

    # Return the ppmi/pmi value as per Turney 2012
    def get_pmi(self, val, sumValRow, sumValCol, totalSum):

        pij, pi, pj = (1.*val/totalSum, 1.*val/sumValRow, 1.*val/sumValCol)
        pmi = log(pij/(pi*pj))
        #if pmi > 0: return pmi
        #else: return 0
        return pmi

    def save_whole_corpus(self, dictionaryFile, contextMatFile):
 
        dictFile = open(dictionaryFile,'w')
        dictFile.write(str(self.freqCutoffForFeatSelection)+' '+str(self.rowCutoff)+' '+str(self.windowSize)+'\n')
        dictFile.write(json.dumps(self.vocab)+'\n')
        dictFile.write(json.dumps(self.wordFeatures)+'\n')
        dictFile.write(json.dumps(self.rowSum)+'\n')
        dictFile.write(json.dumps(self.colSum)+'\n')
        dictFile.close()

        save_sparse_matrix(contextMatFile, self.contextMat)
        
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
