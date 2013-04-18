import sys
from scipy.sparse import *
import numpy

def make_vocab_vectors(fileName, lang1Vocab={}, lang2Vocab={}):
    
    numLines = 0
    
    for line in open(fileName, 'r'):
        numLines += 1  
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
        
        for word in lang1.split():
            if word not in lang1Vocab:
                lang1Vocab[word] = len(lang1Vocab)

        for word in lang2.split():
            if word not in lang2Vocab:
                lang2Vocab[word] = len(lang2Vocab)

    return numLines, lang1Vocab, lang2Vocab

def get_parallel_cooccurence_arrays(fileName, lang1Vocab, lang2Vocab, sizeData):
    
    matrixLang1 = numpy.zeros((sizeData, len(lang1Vocab)), dtype=numpy.float)
    matrixLang2 = numpy.zeros((sizeData, len(lang2Vocab)), dtype=numpy.float)
    
    numLine = 0
    for line in open(fileName, 'r'):
        lang1, lang2 = line.split('|||')
        lang1 = unicode(lang1.strip().lower(), 'utf-8')
        lang2 = unicode(lang2.strip().lower(), 'utf-8')
        
        for word in lang1.split():
            matrixLang1[numLine][lang1Vocab[word]] += 1
            
        for word in lang2.split():
            matrixLang2[numLine][lang2Vocab[word]] = 1
            
        numLine += 1
    
    return (matrixLang1, matrixLang2)
    
def get_datasets(trFile, valFile):
    
    sizeTrData, lang1Vocab, lang2Vocab = make_vocab_vectors(trFile)
    sizeValData, lang1Vocab, lang2Vocab = make_vocab_vectors(valFile, lang1Vocab, lang2Vocab)
    
    sys.stderr.write("\nFiles read...\n")
    sys.stderr.write("Total vocab sizes: lang1 = {0}, lang2 = {1}\n".format(len(lang1Vocab), len(lang2Vocab)))
    sys.stderr.write("Size of files: Train = {0}, Val = {1}\n".format(sizeTrData, sizeValData))
    
    datasets = []
    datasets.append(get_parallel_cooccurence_arrays(trFile, lang1Vocab, lang2Vocab, sizeTrData))
    datasets.append(get_parallel_cooccurence_arrays(valFile, lang1Vocab, lang2Vocab, sizeValData))
    
    return datasets
    