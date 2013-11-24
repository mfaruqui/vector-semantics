import math
import random

def logistic(val):
    
    if val > 20: return 1.
    elif val < -20: return 0.
    else: return 1./(1+math.exp(-1*val))

def score_word_pair(wordVector, contextWordVector, contextWordBias):
    
    return sum(wordVector * contextWordVector) + contextWordBias

def score_word_in_context(word, contextWords, wordBiases, wordVectors, vocab):
    
    # Ideally this function should be parallellized
    return sum([score_word_pair(wordVectors[vocab[word][0]], wordVectors[vocab[contextWord][0]], wordBiases[vocab[contextWord][0]]) for contextWord in contextWords])
    
def diff_score_word_and_noise(word, contextWords, numNoiseWords, noiseDist, wordBiases, wordVectors, vocab):
    
    return score_word_in_context(word, contextWords, wordBiases, wordVectors, vocab) - math.log(numNoiseWords*noiseDist[word])
    
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