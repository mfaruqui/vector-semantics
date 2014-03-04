import numpypy
import numpy
import sys
import time
import multiprocessing
import math

from itertools import repeat

from vlbl import grad_bias, grad_context_word, grad_word
from vlbl import diff_score_word_and_noise, get_noise_words
from vlbl import logistic

from support_functions import create_vocab, filter_and_reindex
from support_functions import normalize_word, random_array
from support_functions import get_file_object

SMOOTHING = math.pow(10, -20)

class LearningCorpus:
    
    def __init__(self, rowCutoff, windowSize, numNoisySamples, numDimensions, corpusFileName):
        
        self.rowCutoff = rowCutoff
        self.windowSize = windowSize
        self.numNoisySamples = numNoisySamples
        self.vectorLen = numDimensions
        self.corpusName = corpusFileName
        self.vocab, self.word2norm = self.get_vocab()
        self.vocabList = self.vocab.keys() #to select noisy words by index, find a cleaner way
        self.noiseDist = self.get_noise_distribution()
        self.wordVectors = random_array(len(self.vocab), self.vectorLen)
        self.wordBiases = random_array(len(self.vocab))
        self.adagradVecMem = SMOOTHING*numpy.ones((len(self.vocab), self.vectorLen))
        self.adagradBiasMem = SMOOTHING*numpy.ones(len(self.vocab))
            
    def get_vocab(self):
        
        startTime = time.time()
        vocab, word2norm = create_vocab(self.corpusName)
        sys.stderr.write("\nTime taken to compute vocab (secs): "+str(time.time()-startTime))
        sys.stderr.write("\nVocabLen: "+str(len(vocab)))

        if self.rowCutoff > 1:
            vocab = filter_and_reindex(vocab, self.rowCutoff)
            sys.stderr.write("\nAfter truncating elements with "+str(self.rowCutoff)+" frequency, vocabLen: "+str(len(vocab)))
        
        return vocab, word2norm
        
    def get_noise_distribution(self):
        
        '''Unigram distribution'''
        corpusSize = 1.*sum([self.vocab[word][1] for word in self.vocab.iterkeys()])
        return {word:self.vocab[word][1]/corpusSize for word in self.vocab.iterkeys()}
        
    def train_on_corpus(self, numIterations, learningRate, batchSize):
        
        for i in range(numIterations):
            startTime = time.time()
            rate = learningRate/(i+1)
            sys.stderr.write('\n\nIteration: '+str(i+1))
            sys.stderr.write('\nLearning rate: '+str(rate))
            self.train_word_vectors(rate, batchSize)
            sys.stderr.write('\nTime taken: '+str(time.time()-startTime)+' secs')
            
        return
        
    def train_word_vectors(self, rate, batchSize):
        
        sys.stderr.write("\nProcessing words:\n")
        numWords, printIf, updateIf = (0, 500000, batchSize)
        batchUpdates = {}
        for line in get_file_object(self.corpusName):
            words = [self.word2norm[word] for word in line.strip().split() if self.word2norm[word] in self.vocab]
            batchUpdates = self.update_word_vectors(words, rate, batchUpdates)
            numWords += len(words)
            if numWords > printIf:
                sys.stderr.write(str(printIf)+' ')
                printIf += 500000
            if numWords > updateIf:
                self.add_gradient_to_words_adagrad(batchUpdates, rate)
                updateIf = 0.
                batchUpdates = {}
            
        return
        
    def update_word_vectors(self, words, rate, batchUpdates):

        noiseWords = get_noise_words(words, self.numNoisySamples, self.vocabList)
        for i, word in enumerate(words):
            if i < self.windowSize: contextWords = words[0:i] + words[i+1:i+self.windowSize+1]
            else: contextWords = words[i-self.windowSize:i] + words[i+1:i+self.windowSize+1]

            wordContextScore = logistic(diff_score_word_and_noise(word, contextWords, self.numNoisySamples, self.noiseDist, self.wordBiases, self.wordVectors, self.vocab))
            noiseScores = [logistic(diff_score_word_and_noise(noiseWord, contextWords, self.numNoisySamples, self.noiseDist, self.wordBiases, self.wordVectors, self.vocab)) for noiseWord in noiseWords]

            updateInWordBias = 1-wordContextScore-sum(noiseScores)
            updateInWordVector = (1-wordContextScore)*grad_word(word, contextWords, self.wordVectors, self.vocab) - \
                                sum([noiseScores[j]*grad_word(noiseWord, contextWords, self.wordVectors, self.vocab) for j, noiseWord in enumerate(noiseWords)])

            wordIndex = self.vocab[word][0]
            if wordIndex not in batchUpdates: 
                batchUpdates[wordIndex] = [updateInWordVector, updateInWordBias, 1.]
            else: 
                batchUpdates[wordIndex][0] += updateInWordVector
                batchUpdates[wordIndex][1] += updateInWordBias
                batchUpdates[wordIndex][2] += 1.
        
        #self.add_gradient_to_words_adagrad(batchUpdates, rate)

        return batchUpdates
        
    def add_gradient_to_words_sgd(self, batchUpdates, rate):
        
        for wordIndex in batchUpdates.iterkeys():
            self.wordVectors[wordIndex] += (rate/batchUpdates[wordIndex][2]) * batchUpdates[wordIndex][0]
            self.wordBiases[wordIndex] += (rate/batchUpdates[wordIndex][2]) * batchUpdates[wordIndex][1]
            
        return
        
    def add_gradient_to_words_adagrad(self, batchUpdates, rate):
        
        for wordIndex in batchUpdates.iterkeys():
            
            # Average the updates individually for every word; Upgrade the adagrad history
            self.adagradVecMem[wordIndex] += numpy.square(batchUpdates[wordIndex][0]/batchUpdates[wordIndex][2])
            self.adagradBiasMem[wordIndex] += numpy.square(batchUpdates[wordIndex][1]/batchUpdates[wordIndex][2])
            
            # Update the parameters
            self.wordVectors[wordIndex] += rate * numpy.divide(batchUpdates[wordIndex][0], numpy.sqrt(self.adagradVecMem[wordIndex]))
            self.wordBiases[wordIndex] += rate * batchUpdates[wordIndex][1]/numpy.sqrt(self.adagradBiasMem[wordIndex])
            
        return