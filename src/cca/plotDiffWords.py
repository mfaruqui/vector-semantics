import sys
import numpy

from alignments import read_word_vectors
from findMatch import cosine_sim
from ranking import spearmans_rho
from ranking import assign_ranks
from tsne import make_graph

if __name__=='__main__':
    
    wordSimFile = sys.argv[1]
    wordVectorFile1 = sys.argv[2]
    wordVectorFile2 = sys.argv[3]
    plotFile = sys.argv[4]
    rankDiff = int(sys.argv[5])

    wordVectors1 = read_word_vectors(wordVectorFile1)
    wordVectors2 = read_word_vectors(wordVectorFile2)

    manualDict = {}
    autoDict1 = {}
    autoDict2 = {}
    notFound = 0
    for line in open(wordSimFile,'r'):
        line = line.strip().lower()
        word1, word2, val = line.split('\t')
        if word1 in wordVectors1 and word2 in wordVectors1:
            manualDict[(word1, word2)] = float(val)
            autoDict1[(word1, word2)] = cosine_sim(wordVectors1[word1], wordVectors1[word2])
            autoDict2[(word1, word2)] = cosine_sim(wordVectors2[word1], wordVectors2[word2])
        else:
            notFound += 1
    print
    #print 'No. pairs not found: ', notFound
    #print "rho = ", spearmans_rho(assign_ranks(manualDict), assign_ranks(autoDict)) 

    rankedDict1 = assign_ranks(autoDict1)
    rankedDict2 = assign_ranks(autoDict2)

    X1 = []
    X2 = []
    Y = []
    wordsCovered = {}
    for (word1, word2) in rankedDict1.iterkeys():
        # If rank difference > rankDiff plot those points
        if abs(rankedDict1[(word1, word2)] - rankedDict2[(word1, word2)]) >= rankDiff:
            if word1 not in wordsCovered:
                X1.append( wordVectors1[word1] )
                X2.append( wordVectors2[word1] )
                wordsCovered[word1] = 0
                Y.append( word1 )

            if word2 not in wordsCovered:
                X1.append( wordVectors1[word2] )
                X2.append( wordVectors2[word2] )
                wordsCovered[word2] = 0
                Y.append( word2 )

    sys.stderr.write('Words to be plotted: '+str(len(Y)))

    with open('inter_vectors1.txt', 'w') as vectorFile:
         for x in X1:
             for i, val in enumerate(x):
                 if i != len(x)-1:
                     vectorFile.write(str(val)+' ')
                 else:
                     vectorFile.write(str(val))
             vectorFile.write('\n')   
    vectorFile.close()

    with open('inter_vectors2.txt', 'w') as vectorFile:
         for x in X2:
             for i, val in enumerate(x):
                 if i != len(x)-1:
                     vectorFile.write(str(val)+' ')
                 else:
                     vectorFile.write(str(val))
             vectorFile.write('\n')   
    vectorFile.close()

    with open('inter_words.txt', 'w') as wordFile:
         for word in Y:
             wordFile.write(word+'\n')
    wordFile.close()
    #numpy.savetxt('inter_vectors.txt', X, delimiter=' ')
                 
    make_graph('inter_vectors1.txt', 'inter_words.txt', plotFile+'1.pdf')
    make_graph('inter_vectors2.txt', 'inter_words.txt', plotFile+'2.pdf')
