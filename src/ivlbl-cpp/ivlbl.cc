#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <string>
#include <algorithm>
#include <tr1/unordered_map>
#include <Eigen/Core>
#include "utils.h"

float logistic(float val){
    
    if (val > 20) return 1;
    else if (val < -20) return 0.;
    else return 1/(1+exp(-1*val));
}

float score_word_pair(RowVectorXf& wordVector, RowVectorXf& contextWordVector, float contextWordBias){
    
    return wordVector.dot(contextWordVector) + contextWordBias;
}

float score_word_in_context(unsigned int word, vector<unsigned int>& contextWords, RowVectorXf& wordBiases, vector<RowVectorXf>& wordVectors){
    
    float sumScore = 0;
    for (int i=0; i<contextWords.size(); i++)
        sumScore += score_word_pair(wordVectors[word], wordVectors[contextWords[i]], wordBiases[contextWords[i]]);
    
    return sumScore;
}

float diff_score_word_and_noise(unsigned int word, vector<unsigned int>& contextWords, int numNoiseWords, mapUintFloat& noiseDist, RowVectorXf& wordBiases, vector<RowVectorXf >& wordVectors){
    
    return score_word_in_context(word, contextWords, wordBiases, wordVectors) - log(numNoiseWords*noiseDist[word]);
}

float grad_bias(unsigned int word, vector<unsigned int>& contextWords, vector<RowVectorXf >& wordVectors){
    
    return 1;
}

RowVectorXf grad_context_word(unsigned int word, vector<unsigned int>& contextWords, vector<RowVectorXf >& wordVectors){
    
    return wordVectors[word];
}

RowVectorXf grad_word(unsigned int word, vector<unsigned int>& contextWords, vector<RowVectorXf>& wordVectors){

    RowVectorXf sumVec(wordVectors[0].size());
    sumVec.setZero(sumVec.size());
    for (int i=0; i<contextWords.size(); i++)
        sumVec += wordVectors[contextWords[i]];
    
    return sumVec;
}

// Can be made better by using a map
vector<unsigned int> get_noise_words(vector<unsigned int>& contextWords, int numNoiseWords, unsigned int vocabSize){

    vector<unsigned int> noiseWords(numNoiseWords, -1);
    unsigned int selectedWords = 0, randIndex;
    while (selectedWords != numNoiseWords){
        randIndex = rand() % vocabSize;
        //if (contextWords.find(randIndex) == contextWords.end())
        if (find(contextWords.begin(), contextWords.end(), randIndex) == contextWords.end())
            noiseWords[selectedWords++] = randIndex;
    }
    
    return noiseWords;
}