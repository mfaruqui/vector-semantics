#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <string>
#include <algorithm>
#include <tr1/unordered_map>
#include "utils.h"
#include "vecops.h"

using namespace std;

typedef std::tr1::unordered_map<string, unsigned int> mapStrUint;
typedef std::tr1::unordered_map<unsigned int, float> mapUintFloat;
typedef std::tr1::unordered_map<unsigned int, unsigned int> mapUintUint;

float logistic(float val){
    
    if (val > 20) return 1.;
    else if (val < -20) return 0.;
    else return 1./(1+exp(-1*val));
}

float score_word_pair(vector<float>& wordVector, vector<float>& contextWordVector, float contextWordBias){
    
    float innerProduct = inner_product(wordVector.begin(), wordVector.end(), contextWordVector.begin(), 0.0);
    return  innerProduct + contextWordBias;
}

float score_word_in_context(unsigned int word, vector<unsigned int>& contextWords, vector<float>& wordBiases, vector<vector<float> >& wordVectors){
    
    float sumScore = 0;
    for (int i=0; i<contextWords.size(); i++)
        sumScore += score_word_pair(wordVectors[word], wordVectors[contextWords[i]], wordBiases[contextWords[i]]);
    
    return sumScore;
}

float diff_score_word_and_noise(unsigned int word, vector<unsigned int>& contextWords, int numNoiseWords, mapUintFloat& noiseDist, vector<float>& wordBiases, vector<vector<float> >& wordVectors){
    
    return score_word_in_context(word, contextWords, wordBiases, wordVectors) - log(numNoiseWords*noiseDist[word]);
}

float grad_bias(unsigned int word, vector<unsigned int>& contextWords, vector<vector<float> >& wordVectors){
    
    return 1;
}

vector<float> grad_context_word(unsigned int word, vector<unsigned int>& contextWords, vector<vector<float> >& wordVectors){
    
    return wordVectors[word];
}

vector<float> grad_word(unsigned int word, vector<unsigned int>& contextWords, vector<vector<float> >& wordVectors){

    vector<float> sumVec(wordVectors[0].size(), 0.0);
    for (int i=0; i<contextWords.size(); i++)
        vec_plus_equal(sumVec, wordVectors[contextWords[i]]);
    
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