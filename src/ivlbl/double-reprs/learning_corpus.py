import numpypy
import numpy
import sys
import time
import multiprocessing
import math

from itertools import repeat

from ivlbl import grad_bias, grad_context_word, grad_target_word
from ivlbl import diff_score_word_and_noise, get_noise_words
from ivlbl import logistic, get_unnormalized_score

from support_functions import create_vocab, filter_and_reindex
from support_functions import normalize_word, random_array
from support_functions import get_file_object

SMOOTHING = math.pow(10, -20)

class LearningCorpus:
    
    def __init__(self, freqCutoff, windowSize, numNoisySamples, numDimensions, corpusFileName):
        
        self.freqCutoff = freqCutoff
        self.windowSize = windowSize
        self.numNoisySamples = numNoisySamples
        self.vectorLen = numDimensions
        self.corpusName = corpusFileName
        self.vocab, self.word2norm = self.get_vocab()
        self.vocabList = self.vocab.keys() #to select noisy words by index, find a cleaner way
        self.noiseDist = self.get_noise_distribution()
        
        # Vectors that represent the word
        self.targetWordVectors = random_array(len(self.vocab), self.vectorLen)
        
        # Vectors that represent the word in context
        self.contextWordVectors = random_array(len(self.vocab), self.vectorLen)
        self.contextWordBiases = random_array(len(self.vocab))
        
        # Adagrad memory to be stored for updates
        self.adagradTargetVecMem = SMOOTHING*numpy.ones((len(self.vocab), self.vectorLen))
        self.adagradContextVecMem = SMOOTHING*numpy.ones((len(self.vocab), self.vectorLen))
        self.adagradContextBiasMem = SMOOTHING*numpy.ones(len(self.vocab))
            
    def get_vocab(self):
        
        startTime = time.time()
        vocab, word2norm = create_vocab(self.corpusName)
        sys.stderr.write("\nTime taken to compute vocab (secs): "+str(time.time()-startTime))
        sys.stderr.write("\nVocabLen: "+str(len(vocab)))

        if self.freqCutoff > 1:
            vocab = filter_and_reindex(vocab, self.freqCutoff)
            sys.stderr.write("\nAfter truncating elements with "+str(self.freqCutoff)+" frequency, vocabLen: "+str(len(vocab)))
        
        return vocab, word2norm
        
    def get_noise_distribution(self):
        
        '''Unigram distribution'''
        corpusSize = 1.*sum([self.vocab[word][1] for word in self.vocab.iterkeys()])
        return {word:self.vocab[word][1]/corpusSize for word in self.vocab.iterkeys()}
        
    def train_on_corpus(self, numIterations, learningRate):
        
        for i in range(numIterations):
            startTime = time.time()
            rate = learningRate/(i+1)
            sys.stderr.write('\n\nIteration: '+str(i+1))
            sys.stderr.write('\nLearning rate: '+str(rate))
            self.train_word_vectors(rate)
            sys.stderr.write('\nTime taken: '+str(time.time()-startTime)+' secs')
            #logLikelihood = get_unnormalized_score(self.corpusName, self.wordVectors, self.wordBiases, self.vocab, self.word2norm, self.windowSize)
            #sys.stderr.write('\nUnnorm score: '+str(logLikelihood))
            
        return
        
    def train_word_vectors(self, rate):
        
        sys.stderr.write("\nProcessing words:\n")
        numWords, printIf = (0, 500000)
        for line in get_file_object(self.corpusName):
            words = [self.word2norm[word] for word in line.strip().split() if self.word2norm[word] in self.vocab]
            self.update_word_vectors(words, rate)
            numWords += len(words)
            if numWords > printIf:
                sys.stderr.write(str(printIf)+' ')
                printIf += 500000
            
        return
        
    def update_word_vectors(self, words, rate):

        noiseWords = get_noise_words(words, self.numNoisySamples, self.vocabList)
        batchUpdatesContext, batchUpdatesTarget = ({}, {})
        for i, word in enumerate(words):
            if i < self.windowSize: contextWords = words[0:i] + words[i+1:i+self.windowSize+1]
            else: contextWords = words[i-self.windowSize:i] + words[i+1:i+self.windowSize+1]

            wordContextScore = logistic(diff_score_word_and_noise(word, contextWords, self.numNoisySamples, self.noiseDist, self.contextWordBiases, self.contextWordVectors, self.targetWordVectors, self.vocab))
            noiseScores = [logistic(diff_score_word_and_noise(noiseWord, contextWords, self.numNoisySamples, self.noiseDist, self.contextWordBiases, self.contextWordVectors, self.targetWordVectors, self.vocab)) for noiseWord in noiseWords]

            # Update in all contextWord vectors and biases is same for all context words
            updateInContextWordBias = 1-wordContextScore-sum(noiseScores)
            updateInContextWordVector = (1-wordContextScore)*grad_context_word(word, contextWords, self.targetWordVectors, self.vocab) - \
                                sum([noiseScores[j]*grad_context_word(noiseWord, contextWords, self.targetWordVectors, self.vocab) for j, noiseWord in enumerate(noiseWords)])
            updateInTargetWordVector = (1-wordContextScore)*grad_target_word(word, contextWords, self.contextWordVectors, self.vocab) - \
                                sum([noiseScores[j]*grad_target_word(noiseWord, contextWords, self.contextWordVectors, self.vocab) for j, noiseWord in enumerate(noiseWords)])

            
            # Update the cache of updates for context words
            for contextWord in contextWords:
                wordIndex = self.vocab[contextWord][0]
                if contextWord not in batchUpdatesContext: 
                    batchUpdatesContext[wordIndex] = [updateInContextWordVector, updateInContextWordBias, 1.]
                else: 
                    batchUpdatesContext[wordIndex][0] += updateInContextWordVector
                    batchUpdatesContext[wordIndex][1] += updateInContextWordBias
                    batchUpdatesContext[wordIndex][2] += 1.
        
            # Update the cache of updates for target word
            wordIndex = self.vocab[word][0]
            if wordIndex not in batchUpdatesTarget: 
                batchUpdatesTarget[wordIndex] = [updateInTargetWordVector, 1.]
            else:
                batchUpdatesTarget[wordIndex][0] += updateInTargetWordVector
                batchUpdatesTarget[wordIndex][1] += 1.
            
        self.add_gradient_to_words_adagrad(batchUpdatesContext, batchUpdatesTarget, rate)

        return
        
    def add_gradient_to_words_sgd(self, batchUpdates, rate):
        
        for wordIndex in batchUpdates.iterkeys():
            self.wordVectors[wordIndex] += (rate/batchUpdates[wordIndex][2]) * batchUpdates[wordIndex][0]
            self.wordBiases[wordIndex] += (rate/batchUpdates[wordIndex][2]) * batchUpdates[wordIndex][1]
            
        return
        
    def add_gradient_to_words_adagrad(self, batchUpdatesContext, batchUpdatesTarget, rate):
        
        for wordIndex in batchUpdatesTarget.iterkeys():
            
            # Average the updates indidvidually for every word; Upgrade the adagrad history
            self.adagradTargetVecMem[wordIndex] += numpy.square(batchUpdatesTarget[wordIndex][0]/batchUpdatesTarget[wordIndex][1])
            # Update the parameters
            self.targetWordVectors[wordIndex] += rate * numpy.divide(batchUpdatesTarget[wordIndex][0], numpy.sqrt(self.adagradTargetVecMem[wordIndex]))
        
        for wordIndex in batchUpdatesContext.iterkeys():
            
            # Average the updates indidvidually for every word; Upgrade the adagrad history
            self.adagradContextVecMem[wordIndex] += numpy.square(batchUpdatesContext[wordIndex][0]/batchUpdatesContext[wordIndex][2])
            self.adagradContextBiasMem[wordIndex] += numpy.square(batchUpdatesContext[wordIndex][1]/batchUpdatesContext[wordIndex][2])
            
            # Update the parameters
            self.contextWordVectors[wordIndex] += rate * numpy.divide(batchUpdatesContext[wordIndex][0], numpy.sqrt(self.adagradContextVecMem[wordIndex]))
            self.contextWordBiases[wordIndex] += rate * batchUpdatesContext[wordIndex][1]/numpy.sqrt(self.adagradContextBiasMem[wordIndex])
            
        return