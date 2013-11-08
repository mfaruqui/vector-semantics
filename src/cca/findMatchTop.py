import sys
import io
import numpy
import gzip
import math

from numpy.linalg import norm

from alignments import read_word_vectors

''' 
Calculates the cosime sim between two numpy arrays
'''
def cosine_sim(vec1, vec2):
    
    return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

''' 
Returns the best match of the given vector but makes
sure that the best match is not one of the input vectors itself
'''
def find_best_match(wordVecToBeMatched, excludeList, wordVectors):

    bestSim = -10
    bestWord = ''
    
    for word in wordVectors.iterkeys():
        sim = cosine_sim(wordVectors[word],wordVecToBeMatched)
        if sim > bestSim and word not in excludeList:
            bestWord = word
            bestSim = sim
            
    return bestWord
    
'''
Performs the entire task of relationship extraction and returns the accuracy
'''
def task_evaluation(wordVecFile, quesFile, vocabFile):
    
    wordVectors = read_word_vectors(wordVecFile)

    # search among only top 50,000 words
    delWords, index = ([], 1.)
    for line in open(vocabFile, 'r'):
        word, freq = line.split()
        if index > 50000: delWords.append(word)
        index += 1
     
    for word in delWords:
        try:
           del wordVectors[word]
        except:
           pass

    print len(delWords)
    print 'Search space size:', len(wordVectors)

    correct, total, notFound = (0., 0., 0.)
    
    for line in io.open(quesFile, 'r', encoding='utf-8'):
         
        a, b, c, answer = line.strip().lower().split()
        if a not in wordVectors or b not in wordVectors or c not in wordVectors:
            sys.stderr.write(line.strip()+" : Not found\n")
            notFound += 1
            continue
            
        wordVecToBeMatched = wordVectors[b] - wordVectors[a] + wordVectors[c]
        excludeList = [b, a, c]
        bestMatch = find_best_match(wordVecToBeMatched, excludeList, wordVectors)
        print a, b, c, bestMatch, answer
     
        if bestMatch == answer:
            correct += 1
        total += 1
        
    print 'Not found:', notFound
    return 1.*correct/total
    
if __name__=='__main__':
    
    wordVecFile, quesFile, vocabFile = (sys.argv[2], sys.argv[1], sys.argv[3]) 
    print task_evaluation(wordVecFile, quesFile, vocabFile)
