#ifndef IVLBL_H_INCLUDED
#define IVLBL_H_INCLUDED

#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <string>
#include <tr1/unordered_map>

using namespace std;
typedef std::tr1::unordered_map<string, unsigned int> mapStrUint;
typedef std::tr1::unordered_map<unsigned int, float> mapUintFloat;
typedef std::tr1::unordered_map<unsigned int, unsigned int> mapUintUint;

float logistic(float val);

float score_word_pair(vector<float>& wordVector, vector<float>& contextWordVector, float contextWordBias);
float score_word_in_context(unsigned int word, vector<unsigned int>& contextWords, vector<float>& wordBiases, vector<vector<float> >& wordVectors);
float diff_score_word_and_noise(unsigned int word, vector<unsigned int>& contextWords, int numNoiseWords, mapUintFloat& noiseDist, vector<float>& wordBiases, vector<vector<float> >& wordVectors);

float grad_bias(unsigned int word, vector<unsigned int>& contextWords, vector<vector<float> >& wordVectors);
vector<float> grad_context_word(unsigned int word, vector<unsigned int>& contextWords, vector<vector<float> >& wordVectors);
vector<float> grad_word(unsigned int word, vector<unsigned int>& contextWords, vector<vector<float> >& wordVectors);

vector<unsigned int> get_noise_words(vector<unsigned int>& contextWords, int numNoiseWords, unsigned int vocabSize);

#endif