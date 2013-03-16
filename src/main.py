import sys
import json
import numpy as np
from learning_corpus import LearningCorpus
import time

if __name__=='__main__':
    
    if len(sys.argv) == 4:
        corpusFileName = sys.argv[1]
        freqCutoff = int(sys.argv[2])
        windowSize = int(sys.argv[3])
        
        test = LearningCorpus(windowSize, freqCutoff, corpusFileName)
        dictionaryFileName = 'data/1m-sents.sample'+'_w'+str(windowSize)+'_f'+str(freqCutoff)+'_.dict'
        contextMatFileName = 'data/1m-sents.sample'+'_w'+str(windowSize)+'_f'+str(freqCutoff)+'_.npy'
        test.save_whole_corpus(dictionaryFileName, contextMatFileName)
        
    elif len(sys.argv) == 3:
        dictFile = open(sys.argv[1], 'r')
        contextMatFile = open(sys.argv[2], 'r')
        
        freqCutoff, windowSize = dictFile.readline().strip().split()
        vocab = json.loads(dictFile.readline())
        wordFeatures = json.loads(dictFile.readline())
        contextMat = np.load(contextMatFile)
        test = LearningCorpus(freqCutoff, windowSize, None, vocab, wordFeatures, contextMat)
        
        del freqCutoff, windowSize, vocab, wordFeatures, contextMat
        
        print "Now sleeping"
        time.sleep(20)
    else:
        print "Unrecognized format"
        sys.exit()
    
    