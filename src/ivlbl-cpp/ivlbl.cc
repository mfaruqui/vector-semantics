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

float logistic(float val) {
    
    if (val > 20) return 1;
    else if (val < -20) return 0.;
    else return 1/(1+exp(-1*val));
}

float score_word_in_context(int word, vector<int>& contextWords,
                            RowVectorXf& wordBiases, vector<RowVectorXf>& wordVectors) {
    
    float sumScore = 0;
    for (int i=0; i<contextWords.size(); i++)
        sumScore += wordVectors[word].dot(wordVectors[contextWords[i]]) + wordBiases[contextWords[i]];
    
    return sumScore;
}

float diff_score_word_and_noise(int word, vector<int>& contextWords,
                                int numNoiseWords, mapIntFloat& noiseDist, 
                                RowVectorXf& wordBiases, vector<RowVectorXf >& wordVectors) {
    
    return score_word_in_context(word, contextWords, wordBiases, wordVectors) - log(numNoiseWords*noiseDist[word]);
}

float grad_bias(int word, vector<int>& contextWords, 
                vector<RowVectorXf >& wordVectors) {
    
    return 1;
}

RowVectorXf grad_context_word(int word, vector<int>& contextWords, 
                              vector<RowVectorXf >& wordVectors) {
    
    return wordVectors[word];
}

RowVectorXf grad_word(int word, vector<int>& contextWords, 
                      vector<RowVectorXf>& wordVectors) {

    RowVectorXf sumVec(wordVectors[0].size());
    sumVec.setZero(sumVec.size());
    for (int i=0; i<contextWords.size(); i++)
        sumVec += wordVectors[contextWords[i]];
    
    return sumVec;
}

// Can be made better by using a map
vector<int> get_noise_words(vector<int>& contextWords, 
                                     int numNoiseWords, int vocabSize) {

    vector<int> noiseWords(numNoiseWords, -1);
    int selectedWords = 0, randIndex;
    while (selectedWords != numNoiseWords){
        randIndex = rand() % vocabSize;
        //if (contextWords.find(randIndex) == contextWords.end())
        if (find(contextWords.begin(), contextWords.end(), randIndex) == contextWords.end())
            noiseWords[selectedWords++] = randIndex;
    }
    
    return noiseWords;
}