import numpypy
import numpy
import sys
import time
import multiprocessing

from itertools import repeat

from ivlbl import grad_bias, grad_context_word, grad_word
from ivlbl import diff_score_word_and_noise, get_noise_words
from ivlbl import logistic

from support_functions import create_vocab, filter_and_reindex
from support_functions import normalize_word
from support_functions import random_array
from support_functions import get_file_object, get_words_to_update

'''
To be fixed: 
learningRate parameter
'''

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
        self.train_on_corpus()
            
    def get_vocab(self):
        
        startTime = time.time()
        vocab, word2norm = create_vocab(self.corpusName)
        sys.stderr.write("Time taken to compute vocab (secs): "+str(time.time()-startTime)+"\n")
        sys.stderr.write("VocabLen: "+str(len(vocab))+"\n")

        if self.rowCutoff > 0:
            vocab = filter_and_reindex(vocab, self.rowCutoff)
            sys.stderr.write("\nAfter truncating elements with "+str(self.rowCutoff)+" frequency, vocabLen: "+str(len(vocab))+"\n")
        
        return vocab, word2norm
        
    def get_noise_distribution(self):
        
        '''Unigram distribution'''
        corpusSize = 1.*sum([self.vocab[word][1] for word in self.vocab.iterkeys()])
        return {word:self.vocab[word][1]/corpusSize for word in self.vocab.iterkeys()}
        
    def train_on_corpus(self):
        
        numIterations = 1
        for i in range(numIterations):
            startTime = time.time()
            learningRate = 0.5/(i+1)
            sys.stderr.write('\nIteration: '+str(i+1)+'\n')
            self.train_word_vectors(learningRate)
            sys.stderr.write('\nTime taken: '+str(time.time()-startTime)+' secs\n')
            
        return
        
    def train_word_vectors(self, learningRate):
        
        window, numWords = ([], 0)
        sys.stderr.write("Processing words:\n")
        startTime = time.time()
        
        printIf = 100000
        for line in get_file_object(self.corpusName):
            words = [self.word2norm[word] for word in line.strip().split() if self.word2norm[word] in self.vocab]
            self.update_word_vectors(words, learningRate)
            numWords += len(words)
            if numWords > printIf:
                sys.stderr.write(str(printIf)+' ')
                printIf += 100000
            
        return
        
    def update_word_vectors(self, words, learningRate):
        
        noiseWords = get_noise_words(words, self.numNoisySamples, self.vocabList)
        #numWordsToBeUpdated = min(1, len(words))
        for i, word in enumerate(words):
            if i < self.windowSize: contextWords = words[0:i] + words[i+1:i+self.windowSize+1]
            else: contextWords = words[i-self.windowSize:i] + words[i+1:i+self.windowSize+1]
            
            wordContextScore = logistic(diff_score_word_and_noise(word, contextWords, self.numNoisySamples, self.noiseDist, self.wordBiases, self.wordVectors, self.vocab))
            noiseScores = [logistic(diff_score_word_and_noise(noiseWord, contextWords, self.numNoisySamples, self.noiseDist, self.wordBiases, self.wordVectors, self.vocab)) for noiseWord in noiseWords]
            
            # Update in all contextWord vectors and biases is same for all context words
            updateInContextWordBias = 1-wordContextScore-sum(noiseScores)
            updateInContextWordVector = (1-wordContextScore)*grad_context_word(word, contextWords, self.wordVectors, self.vocab) - \
                                sum([noiseScores[i]*grad_context_word(noiseWord, contextWords, self.wordVectors, self.vocab) for i, noiseWord in enumerate(noiseWords)])
            
            #self.add_gradient_to_words(get_words_to_update(numWordsToBeUpdated, contextWords), updateInContextWordVector, updateInContextWordBias, learningRate)
            for contextWord in contextWords:
                self.wordVectors[self.vocab[contextWord][0]] += learningRate * updateInContextWordVector
                self.wordBiases[self.vocab[contextWord][0]] += learningRate * updateInContextWordBias
                
        return
        
    def add_gradient_to_words(self, words, vectorUpdate, biasUpdate, learningRate):
        
        for word in words:
            self.wordVectors[self.vocab[word][0]] += learningRate * vectorUpdate
            self.wordBiases[self.vocab[word][0]] += learningRate * biasUpdate
        
        return