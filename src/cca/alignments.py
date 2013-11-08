import sys
import argparse
import gzip
import numpy
import math

from collections import Counter
from operator import itemgetter

'''
For every word in lang2 this method finds a coressponding vector
using weighted average of the vectors that word is aligned to in lang1
'''
def get_aligned_vectors(wordAlignFile, lang1WordVectors, lang2WordVectors):
    
    '''
    The number of aligned vectors will be less than or equal to the lang2
    and will have the dimensions of lang1
    '''
    alignedVectors = {}
    lenLang1Vector = len(lang1WordVectors[lang1WordVectors.keys()[0]])
    tookTop = 0
    for line in open(wordAlignFile, 'r'):
        lang2Word, rest = line.strip().split(" ||| ")
        lang1Words, lang1WordsFreq = ([], [])

        if lang2Word not in lang2WordVectors: continue
        
        for wordFreq in rest.split():
            word, freq = wordFreq.rsplit('__', 1)
            freq = float(freq)
            lang1Words.append(word)
            lang1WordsFreq.append(freq)
            
        if lang1Words[0] not in lang1WordVectors: continue

        alignedVectors[lang2Word] = numpy.zeros(lenLang1Vector, dtype=float)
        alignedVectors[lang2Word] += lang1WordVectors[lang1Words[0]]

    sys.stderr.write("No. of aligned vectors found: "+str(len(alignedVectors))+'\n')
        
    return alignedVectors

''' Read all the word vectors and normalize them '''
def read_word_vectors(filename):
    
    wordVectors = {}

    if filename.endswith('.gz'):
        for lineNum, line in enumerate(gzip.open(filename, 'r')):
            line = line.strip().lower()
            word = line.split()[0]
            wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
            for index, vecVal in enumerate(line.split()[1:]):
                wordVectors[word][index] = float(vecVal)

    else:
        for lineNum, line in enumerate(open(filename, 'r')):
            line = line.strip().lower()
            word = line.split()[0]
            wordVectors[word] = numpy.zeros(len(line.split())-1, dtype=float)
            for index, vecVal in enumerate(line.split()[1:]):
                wordVectors[word][index] = float(vecVal)
            
    ''' normalize weight vector '''
    wordVectors[word] /= math.sqrt((wordVectors[word]**2).sum() + 1e-6)
            
    sys.stderr.write("Vectors read from: "+filename+" \n")
    return wordVectors
    
'''
Takes as input a biiig parallel corpus and its word aligned file
along with the word projections of the two languages. 

Output: alignment counts of words in the word projection file from
the biig corpus.
'''
def count_word_alignments(parallelFile, alignmentFile, lang1WordVectors, lang2WordVectors):
    
    wordAlignDict = {}
    lineNum = 1
    for pLine, aLine in zip(gzip.open(parallelFile, 'r'),gzip.open(alignmentFile, 'r')):
        l1, l2 = pLine.lower().strip().split(' ||| ')
        l1Words, l2Words = (l1.split(), l2.split())
        
        for wordIndexPair in aLine.strip().split():
            i, j = wordIndexPair.split('-')
            i, j = (int(i), int(j))
            ''' count alignment only if both words have word vectors '''
            if l1Words[i] in lang1WordVectors and l2Words[j] in lang2WordVectors:
                if l2Words[j] in wordAlignDict:
                    if l1Words[i] in wordAlignDict[l2Words[j]]:
                        wordAlignDict[l2Words[j]][l1Words[i]] += 1
                    else:
                        wordAlignDict[l2Words[j]][l1Words[i]] = 1
                else:
                    wordAlignDict[l2Words[j]] = {l1Words[i]: 1}
        
        if lineNum%10000 == 0: sys.stderr.write(str(lineNum)+' ')
        lineNum += 1

    sys.stderr.write(str(len(wordAlignDict))+"\n")
    return wordAlignDict

def read_files(args):
    
    lang1WordVectors = read_word_vectors(args.wordproj1)
    lang2WordVectors = read_word_vectors(args.wordproj2)
    
    wordAlignDict = count_word_alignments(args.parallelfile, args.alignfile, lang1WordVectors, lang2WordVectors)
    
    for word in wordAlignDict.iterkeys():
        print word, '|||',
        for alignedWord, freq in sorted(wordAlignDict[word].items(), key=itemgetter(1), reverse=True):
            print alignedWord+'__'+str(freq),
        print
        
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parallelfile", type=str, help="Joint parallel file")
    parser.add_argument("-a", "--alignfile", type=str, help="Word alignment file")
    parser.add_argument("-w1", "--wordproj1", type=str, help="Word proj of lang1")
    parser.add_argument("-w2", "--wordproj2", type=str, help="Word proj of lang2")
    
    args = parser.parse_args()
    read_files(args)
