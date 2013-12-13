#ifndef TRAIN_H_INCLUDED
#define TRAIN_H_INCLUDED

#include <iostream>
#include <vector>
#include "utils.h"
#include "vecops.h"
#include "ivlbl.h"

using namespace std;

vector<unsigned int> words_in_window(vector<unsigned int>& words, unsigned int wordIndex, unsigned int windowSize);

void add_grad_to_words_adagrad(vector<unsigned int>& contextWords, float rate, vector<float>& updateVec, 
                                float updateBias, vector<vector<float> >& adagradVecMem, vector<float>& adagradBiasMem,
                                vector<vector<float> >& wordVectors, vector<float>& wordBiases);
                                
void train_word_vectors(vector<unsigned int>& words, vector<vector<float> >& wordVectors, vector<float>& wordBiases,
                        vector<vector<float> >& adagradVecMem, vector<float>& adagradBiasMem,
                        mapStrUint& indexedVocab, mapUintFloat& noiseDist, unsigned int numNoiseWords, unsigned int vocabSize,
                        unsigned int windowSize, float rate);

pair<vector<vector<float> >, mapStrUint> train_on_corpus(char* fileName, unsigned int numIter, float learningRate, 
                                        unsigned int numNoiseWords, unsigned int windowSize, unsigned int freqCutoff,
                                        unsigned int vecLen);
                                        
#endif