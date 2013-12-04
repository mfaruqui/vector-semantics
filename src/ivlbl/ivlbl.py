import math
import sys
import random
from collections import Counter

def logistic(val):
    
    if val > 20: return 1.
    elif val < -20: return 0.
    else: return 1./(1+math.exp(-1*val))

def score_word_pair(wordVector, contextWordVector, contextWordBias):
    
    return sum(wordVector * contextWordVector) + contextWordBias

def score_word_in_context(word, contextWords, wordBiases, wordVectors, vocab):
    
    return sum([score_word_pair(wordVectors[vocab[word][0]], wordVectors[vocab[contextWord][0]], wordBiases[vocab[contextWord][0]]) for contextWord in contextWords])
    
def diff_score_word_and_noise(word, contextWords, numNoiseWords, noiseDist, wordBiases, wordVectors, vocab):
    
    return score_word_in_context(word, contextWords, wordBiases, wordVectors, vocab) - math.log(numNoiseWords*noiseDist[word])

def grad_bias(word, contextWords, wordVectors, vocab):
    
    return 1.
    
def grad_context_word(word, contextWords, wordVectors, vocab):
    
    return wordVectors[vocab[word][0]]
    
def grad_word(word, contextWords, wordVectors, vocab):
    
    return sum([wordVectors[vocab[contextWord][0]] for contextWord in contextWords])    

''' Select words that are not in the given list of words '''
def get_noise_words(contextWords, numNoiseWords, vocabList):
    
    noiseWords = []
    while len(noiseWords) != numNoiseWords:
        randomWord = random.choice(vocabList) 
        if randomWord not in contextWords: 
            noiseWords.append(randomWord)
        
    return noiseWords
    
def get_unnormalized_score(fileName, wordVectors, wordBiases, wordVocab, word2norm, windowSize):
    
    score = 0.
    for line in open(fileName, 'r'):
        words = [word2norm[word] for word in line.strip().split() if word2norm[word] in wordVocab]
        
        for i, word in enumerate(words):
            
            if i < windowSize: contextWords = words[0:i] + words[i+1:i+windowSize+1]
            else: contextWords = words[i-windowSize:i] + words[i+1:i+windowSize+1]
            
            # You are calculating the score of the context here wrt the target word
            contextScore = score_word_in_context(word, contextWords, wordBiases, wordVectors, wordVocab)
            score += contextScore
            
    return score
    
def get_loglikelihood(fileName, wordVectors, wordBiases, wordVocab, word2norm, windowSize):
    
    # For every word compute the sum of its score with the whole vocabulary
    wordVocabScore = Counter()
    for word1 in wordVocab.iterkeys():
        for word2 in wordVocab.iterkeys():
            wordVector, contextWordVector, contextWordBias = (wordVectors[wordVocab[word1][0]], wordVectors[wordVocab[word2][0]], wordBiases[wordVocab[word2][0]])
            wordVocabScore[word1] += math.exp(score_word_pair(wordVector, contextWordVector, contextWordBias))
    
    for word in wordVocabScore.iterkeys(): wordVocabScore[word] = math.log(wordVocabScore[word])        
    
    score = 0.
    for line in open(fileName, 'r'):
        words = [word2norm[word] for word in line.strip().split() if word2norm[word] in wordVocab]
        
        for i, word in enumerate(words):
            if i < windowSize: contextWords = words[0:i] + words[i+1:i+windowSize+1]
            else: contextWords = words[i-windowSize:i] + words[i+1:i+windowSize+1]
            
            # You are calculating the score of the context with respect to the target word
            contextScore = score_word_in_context(word, contextWords, wordBiases, wordVectors, wordVocab)
            score += contextScore - len(contextWords)*wordVocabScore[word]
            
    return score