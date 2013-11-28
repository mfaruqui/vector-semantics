import math
import random

def logistic(val):
    
    if val > 20: return 1.
    elif val < -20: return 0.
    else: return 1./(1+math.exp(-1*val))

def score_word_pair(targetWordVector, contextWordVector, contextWordBias):
    
    return sum(targetWordVector * contextWordVector) + contextWordBias

def score_word_in_context(word, contextWords, contextWordBiases, contextWordVectors, targetWordVectors, vocab):
    
    return sum([score_word_pair(targetWordVectors[vocab[word][0]], contextWordVectors[vocab[contextWord][0]], contextWordBiases[vocab[contextWord][0]]) for contextWord in contextWords])
    
def diff_score_word_and_noise(word, contextWords, numNoiseWords, noiseDist, contextWordBiases, contextWordVectors, targetWordVectors, vocab):
    
    return score_word_in_context(word, contextWords, contextWordBiases, contextWordVectors, targetWordVectors, vocab) - math.log(numNoiseWords*noiseDist[word])

def grad_bias(word, contextWords, wordVectors, vocab):
    
    return 1.
    
def grad_context_word(word, contextWords, wordVectors, vocab):
    
    return wordVectors[vocab[word][0]]
    
def grad_target_word(word, contextWords, wordVectors, vocab):
    
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
            
            wordContextScore = score_word_in_context(word, contextWords, wordBiases, wordVectors, wordVocab)
            score += wordContextScore
            
    return score