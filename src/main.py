import sys
from learning_corpus import LearningCorpus

corpusFileName = sys.argv[1]
freqCutoff = int(sys.argv[2])
windowSize = int(sys.argv[3])

if __name__=='__main__':
    
    test = LearningCorpus(corpusFileName, windowSize, freqCutoff)
    print test.vocabLen, test.featLen
    outFileName = corpusFileName+'_w'+str(windowSize)+'_f'+str(freqCutoff)+'_.npy'
    outFile = open(outFileName,'w')
    np.save(outFile, test.contextMat)
