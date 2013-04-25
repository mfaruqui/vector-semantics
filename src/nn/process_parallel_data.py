import sys
from scipy.sparse import csr_matrix
import numpy
import re
from collections import Counter

number = '[0-9]+'
isNumber = re.compile(number)

FREQ_THRESH = 5

def normalize_word(word):
    
    if isNumber.search(word):
        return '---$$$---'
    else:
        return word
        
def trim_vocab(vocab):
    
    new_index = 0
    for word, freq in vocab.items():
        if freq <= FREQ_THRESH:
            del vocab[word]
        else:
            vocab[word] = new_index
            new_index += 1
            
    return vocab

def get_vocab(fileName, lang1Vocab=Counter(), lang2Vocab=Counter()):
    
    numLines = 0
    
    for line in open(fileName, 'r'):
        numLines += 1  
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
        
        for word in lang1.split():
            word = normalize_word(word)
            lang1Vocab[word] += 1 

        for word in lang2.split():
            word = normalize_word(word)
            lang2Vocab[word] += 1

    #trim the vocab by frequency and replace frequency by unique number
    return numLines, trim_vocab(lang1Vocab), trim_vocab(lang2Vocab)
    
def convert_dict_to_csr_matrix(matrixDict, sizeData, langVocab):
    
    row = numpy.zeros(len(matrixDict), dtype=int)
    col = numpy.zeros(len(matrixDict), dtype=int)
    values = numpy.zeros(len(matrixDict), dtype=int)
    
    index = 0
    for (r, c), val in matrixDict.iteritems():
        row[index] = r
        col[index] = c
        values[index] = val
        index += 1
    
    matrixLang = csr_matrix((values,(row,col)), shape=(sizeData,len(langVocab)))
    return matrixLang

def get_parallel_cooccurence_arrays(fileName, lang1Vocab, lang2Vocab, sizeData):
    
    matrixDict1 = Counter()
    numLine = 0
    for line in open(fileName, 'r'):
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
        
        for word in lang1.split():
            word = normalize_word(word)
            if word in lang1Vocab:
                # we want count of the words on the input
                matrixDict1[(numLine,lang1Vocab[word])] += 1
                
        numLine += 1
    
    matrixLang1 = convert_dict_to_csr_matrix(matrixDict1, sizeData, lang1Vocab)  
    del matrixDict1
    
    matrixDict2 = Counter()
    numLine = 0
    for line in open(fileName, 'r'):
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
                
        for word in lang2.split():
            word = normalize_word(word)
            if word in lang2Vocab:
                # we want probability of occurrence on the output
                matrixDict2[(numLine,lang2Vocab[word])] = 1
            
        numLine += 1
    
    matrixLang2 = convert_dict_to_csr_matrix(matrixDict2, sizeData, lang2Vocab)  
    del matrixDict2
    
    return (matrixLang1, matrixLang2)
    
def get_datasets(trFile, valFile):
    
    sizeTrData, lang1Vocab, lang2Vocab = get_vocab(trFile)
    sizeValData, lang1Vocab, lang2Vocab = get_vocab(valFile, lang1Vocab, lang2Vocab)
    
    sys.stderr.write("\nFiles read...\n")
    sys.stderr.write("Total vocab sizes: lang1 = {0}, lang2 = {1}\n".format(len(lang1Vocab), len(lang2Vocab)))
    sys.stderr.write("Size of files: Train = {0}, Val = {1}\n".format(sizeTrData, sizeValData))
    
    datasets = []
    datasets.append(get_parallel_cooccurence_arrays(trFile, lang1Vocab, lang2Vocab, sizeTrData))
    datasets.append(get_parallel_cooccurence_arrays(valFile, lang1Vocab, lang2Vocab, sizeValData))
    
    return datasets