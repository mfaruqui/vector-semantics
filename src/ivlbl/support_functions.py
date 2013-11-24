import sys
import gzip
import re
import numpypy
import numpy
import random
import math

from operator import itemgetter

def get_file_object(fileName):
    
    if fileName.endswith('.gz'): return gzip.open(fileName, 'r')
    else: return open(fileName, 'r')

isNumber = re.compile(r'\d+.*')
def normalize_word(word):
    
    word = unicode(word, "utf-8")
    if isNumber.search(word.lower()): return '---num---'
    elif re.sub(r'\W+', '', word) == '': return '---punc---'
    else: return word.lower()

# vocab = {word:[index, freq], ...}    
def create_vocab(fileName):
    
    vocab, word2norm, wordId = ({}, {}, 0)
    for line in get_file_object(fileName):
        for word in line.split():
            if word not in word2norm: 
                word2norm[word] = normalize_word(word)
                word = word2norm[word]
            else:
                word = word2norm[word]
                
            if word in vocab:
                vocab[word][1] += 1
            else:
                vocab[word] = [wordId, 1]
                wordId += 1
                
    return (vocab, word2norm)
    
# vocab = {word:[index, freq], ...}
def filter_and_reindex(vocab, thresh):
    
    index = 0
    for word in vocab.keys():
        if vocab[word][1] < thresh:
            del vocab[word]
        else:
            vocab[word] = [index, vocab[word][1]]
            index += 1
            
    return vocab
    
def print_vectors(vocab, wordVectors, outFileName):
    
    sys.stderr.write('Writing down the vectors in '+outFileName+'\n')
    outFile = open(outFileName, 'w')
    
    # Sort the frequency by frequency and then print the vectors in that order
    for word, values in sorted(vocab.items(), key=itemgetter(1, 1), reverse=False):
        index, freq = (values[0], values[1])
        outFile.write(word.encode('utf-8')+' ')
        for val in wordVectors[index]:
            outFile.write('%.4f' %(val)+' ')
        outFile.write('\n')
        
    outFile.close()
    
# works only if the values in the array are floats
def normalize_array(array):
    
    array /= math.sqrt((array**2).sum())
    return array

def random_array(row, col=None):
    
    if col == None:
        return normalize_array(numpy.array([random.uniform(0, 1) for i in range(row)]))
    else:
        return numpy.array([normalize_array(numpy.array([random.uniform(0, 1) for i in range(col)])) for j in range(row)])
        
def get_words_to_update(numWords, contextWords):
    
    if contextWords == []: return []
    else: return [random.choice(contextWords) for i in range(numWords)]
    