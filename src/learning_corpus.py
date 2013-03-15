import numpy as np
import gzip

class LearningCorpus:
    
    def __init__(self, corpusFileName, windowSize, freqCutoff):
        
        self.corpusName = corpusFileName
        self.vocab = self.read_corpus()
        self.vocabLen = len(self.vocab)
        
        self.windowSize = windowSize
        self.freqCutoffForFeatSelection = freqCutoff
        self.wordFeatures = self.select_feat_words()
        self.featLen = len(self.wordFeatures)
        
        self.contextMat = np.zeros((self.vocabLen, self.featLen), dtype=int)
        self.get_context_vectors()
        
    def read_corpus(self):
        
        vocab = {}
        wordId = 0
        for line in gzip.open(self.corpusName, 'r'):
            line = unicode(line.strip(), "utf-8")
            for word in line.split():
                if word in vocab:
                    vocab[word][1] += 1
                else:
                    vocab[word] = [wordId, 1]
                    wordId += 1
                    
        return vocab
        
    def select_feat_words(self):
        
        wordFeatures = {}
        index = 0
        for word, [wordId, freq] in self.vocab.iteritems():
            if freq > self.freqCutoffForFeatSelection:
                wordFeatures[word] = index
                index += 1
                
        return wordFeatures
    
    #Will work properly only if window size if lesser than the number of tokens
    def get_context_vectors(self):
        
        window = []
        firstWindow = 1
        with gzip.open(self.corpusName, 'r') as f:
            for word in f.read().strip().split()+['.']:
                word = unicode(word,"utf-8")
                
                #collecting words to make the window full
                if len(window) < self.windowSize + 1:
                    window.append(word)
                    
                #processing the first window
                elif firstWindow == 1:
                    firstWindow = 0
                    for i in range(0, len(window)):
                        for j in range(0, len(window)):
                            if window[j] in self.wordFeatures and j != i:
                                self.contextMat[self.vocab[window[i]][0]][self.wordFeatures[window[j]]] += 1
                    #remove the first word and insert the new word in the window
                    garbage = window.pop(0)
                    window.append(word)
                
                #processing all intermediate windows
                else:
                    for i in range(0,len(window)-1):
                        if window[i] in self.wordFeatures:
                            self.contextMat[self.vocab[window[len(window)-1]][0]][self.wordFeatures[window[i]]] += 1
                        if window[len(window)-1] in self.wordFeatures:
                            self.contextMat[self.vocab[window[i]][0]][self.wordFeatures[window[len(window)-1]]] += 1
                    #remove the first word and insert the new word in the window
                    garbage = window.pop(0)
                    window.append(word)
