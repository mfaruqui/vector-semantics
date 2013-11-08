from math import sqrt
from operator import itemgetter

from ranking import *

def cosine(a, b):

    numr = 0
    aNorm = 0
    bNorm = 0
    for ai, bi in zip(a,b):
        numr = ai*bi
        aNorm += ai**2
        bNorm += bi**2

    return numr/(sqrt(aNorm)*sqrt(bNorm))

if __name__=='__main__':

    manualDict = {}
    autoDict = {}
    neg = 0

    newDict = {}
    lineNum = 1
    for line in open('packages/rnn/FeatureAugmentedRNNToolkit/word_projections.txt', 'r'):
        if lineNum != 1:
           word, rest = line.split()[0], line.split()[1:]
           newDict[word] = []
           for val in rest:
               newDict[word].append(float(val))
        lineNum += 1

    for line in open('corpus/wordsim353/combined.tab','r'):
        line = line.strip().lower()
        word1, word2, val = line.split('\t')
        
	if word1 not in newDict or word2 not in newDict:
            pass
        else:
	    autoVal = cosine(newDict[word1], newDict[word2])
            manualDict[(word1, word2)] = val
            autoDict[(word1, word2)] = autoVal

    print
    print "rho = ", spearmans_rho(assign_ranks(manualDict), assign_ranks(autoDict))

for (word1, word2), val in sorted(autoDict.items(), key=itemgetter(1), reverse=True):
    print word1, word2, val
