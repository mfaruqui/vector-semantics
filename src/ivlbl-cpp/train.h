#ifndef TRAIN_H_INCLUDED
#define TRAIN_H_INCLUDED

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "utils.h"
#include "ivlbl.h"

vector<unsigned int> words_in_window(vector<unsigned int>& words, 
                                     unsigned int wordIndex, unsigned int windowSize);

void add_grad_to_words_adagrad(vector<unsigned int>& contextWords, float rate, 
                               RowVectorXf& updateVec, float updateBias, 
                               vector<RowVectorXf>& adagradVecMem, RowVectorXf& adagradBiasMem,
                               vector<RowVectorXf>& wordVectors, RowVectorXf& wordBiases);
                                
void train_word_vectors(vector<unsigned int>& words, vector<RowVectorXf>& wordVectors,
                        RowVectorXf& wordBiases, vector<RowVectorXf>& adagradVecMem, 
                        RowVectorXf& adagradBiasMem, mapStrUint& indexedVocab,
                        mapUintFloat& noiseDist, unsigned int numNoiseWords,
                        unsigned int vocabSize, unsigned int windowSize, float rate);

pair<vector<RowVectorXf >, mapStrUint> train_on_corpus(char* fileName, 
                                                       unsigned int numIter, 
                                                       float learningRate, 
                                                       unsigned int numNoiseWords, 
                                                       unsigned int windowSize, 
                                                       unsigned int freqCutoff,
                                                       unsigned int vecLen);
                                        
#endif