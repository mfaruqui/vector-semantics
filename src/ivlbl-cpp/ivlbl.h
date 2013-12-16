#ifndef IVLBL_H_INCLUDED
#define IVLBL_H_INCLUDED

#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

typedef std::tr1::unordered_map<string, int> mapStrInt;
typedef std::tr1::unordered_map<int, float> mapIntFloat;
typedef std::tr1::unordered_map<int, int> mapIntInt;

float logistic(float val);

float score_word_pair(RowVectorXf& wordVector, RowVectorXf& contextWordVector,
                      float contextWordBias);
                      
float score_word_in_context(int word, vector<int>& contextWords,
                            RowVectorXf& wordBiases, 
                            vector<RowVectorXf>& wordVectors);
                            
float diff_score_word_and_noise(int word, vector<int>& contextWords, 
                                int numNoiseWords, mapIntFloat& noiseDist,
                                RowVectorXf& wordBiases, 
                                vector<RowVectorXf >& wordVectors);

float grad_bias(int word, vector<int>& contextWords,
                vector<RowVectorXf>& wordVectors);
                
RowVectorXf grad_context_word(int word, vector<int>& contextWords,
                              vector<RowVectorXf>& wordVectors);
                              
RowVectorXf grad_word(int word, vector<int>& contextWords,
                      vector<RowVectorXf>& wordVectors);

vector<int> get_noise_words(vector<int>& contextWords, int numNoiseWords, 
                            int vocabSize);

#endif