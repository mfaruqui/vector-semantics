#include <iostream>
#include <fstream>
#include <vector>
#include "utils.h"
#include "vecops.h"
#include "ivlbl.h"

using namespace std;

vector<unsigned int> words_in_window(vector<unsigned int>& words, unsigned int wordIndex, unsigned int windowSize){
    
    vector<unsigned int> wordsInWindow;
    unsigned int sentLen = words.size();
    
    if (wordIndex < windowSize){
        if (wordIndex + windowSize + 1 < sentLen)
            wordsInWindow.insert(wordsInWindow.begin(), words.begin()+wordIndex+1, words.begin()+wordIndex+1+windowSize);
        else
            wordsInWindow.insert(wordsInWindow.begin(), words.begin()+wordIndex+1, words.end());
        wordsInWindow.insert(wordsInWindow.begin(), words.begin(), words.begin()+wordIndex);
    }
    else{
        if (wordIndex + windowSize + 1 < sentLen)
            wordsInWindow.insert(wordsInWindow.begin(), words.begin()+wordIndex+1, words.begin()+wordIndex+1+windowSize);
        else
            wordsInWindow.insert(wordsInWindow.begin(), words.begin()+wordIndex+1, words.end());
        wordsInWindow.insert(wordsInWindow.begin(), words.begin()+wordIndex-windowSize, words.begin()+wordIndex);
    }
    
    return wordsInWindow;
}

void add_grad_to_words_adagrad(vector<unsigned int>& contextWords, float rate, vector<float>& updateVec, 
                                float updateBias, vector<vector<float> >& adagradVecMem, vector<float>& adagradBiasMem,
                                vector<vector<float> >& wordVectors, vector<float>& wordBiases){
    
    for(int i=0; i<contextWords.size(); i++){
        
        unsigned int word = contextWords[i];
        
        // Update the adagrad memory first 
        vec_plus_equal(adagradVecMem[word], vec_square(updateVec));
        adagradBiasMem[word] += pow(updateBias, 2);
        
        // Now apply the updates
        // This: wordVectors[word] += rate * updateVec / sqrt(adagradVecMem[word])
        vec_div_equal(updateVec, vec_sqrt(adagradVecMem[word]));
        vec_prod_equal(updateVec, rate);
        vec_plus_equal(wordVectors[word], updateVec);
        
        wordBiases[word] += rate * updateBias / sqrt(adagradBiasMem[word]);
    }
    return;
}

void train_word_vectors(vector<unsigned int>& words, vector<vector<float> >& wordVectors, vector<float>& wordBiases,
                        vector<vector<float> >& adagradVecMem, vector<float>& adagradBiasMem,
                        mapUintFloat& noiseDist, unsigned int numNoiseWords, unsigned int vocabSize,
                        unsigned int windowSize, float rate){

    vector<unsigned int> noiseWords = get_noise_words(words, numNoiseWords, vocabSize);
    unsigned int word, noiseWord;
    vector<unsigned int> contextWords;
    
    float wordContextScore, updateInContextBias, noiseScore;
    vector<float> noiseScores, updateInContextVec, noiseScoreGradProd(wordVectors[0].size(), 0), temp;
    
    
    for (int i=0; i<words.size(); i++){
        
        word = words[i];
        contextWords = words_in_window(words, i, windowSize);
        wordContextScore = logistic(diff_score_word_and_noise(word, contextWords, numNoiseWords, noiseDist, wordBiases, wordVectors));
        
        for (int j=0; j<noiseWords.size(); j++){
            noiseWord = noiseWords[j];
            noiseScore = logistic(diff_score_word_and_noise(noiseWord, contextWords, numNoiseWords, noiseDist, wordBiases, wordVectors));
            noiseScores.push_back(noiseScore);
            temp = grad_context_word(noiseWord, contextWords, wordVectors);
            vec_plus_equal(noiseScoreGradProd, vec_prod(temp, noiseScore));
        }
                                                        
        updateInContextBias = 1 - wordContextScore - accumulate(noiseScores.begin(), noiseScores.end(), 0);
        temp = grad_context_word(word, contextWords, wordVectors);
        vec_prod_equal(temp, 1 - wordContextScore);
        updateInContextVec = vec_plus(temp, noiseScoreGradProd);
        
        add_grad_to_words_adagrad(contextWords, rate, updateInContextVec, updateInContextBias, 
                                    adagradVecMem, adagradBiasMem, wordVectors, wordBiases);
    }
    
    return;   
}

pair<vector<vector<float> >, mapStrUint> train_on_corpus(char* fileName, unsigned int numIter, float learningRate, 
                                        unsigned int numNoiseWords, unsigned int windowSize, unsigned int freqCutoff,
                                        unsigned int vecLen){
    
    mapStrUint vocab, indexedVocab;
    mapUintFloat noiseDist;
    mapStrStr word2norm;
    unsigned vocabSize;
    float rate;
    
    pair<mapStrUint, mapStrStr> vocabPair = get_vocab(fileName);
    vocab = vocabPair.first;
    word2norm = vocabPair.second;
    
    filter_vocab(vocab, freqCutoff);
    indexedVocab = reindex_vocab(vocab);
    noiseDist = get_unigram_dist(vocab, indexedVocab);
    vocabSize = indexedVocab.size();
    
    vector<vector<float> > wordVectors = random_vector(vocabSize, vecLen), adagradVecMem = epsilon_vector(vocabSize, vecLen);
    vector<float> wordBiases = random_vector(vocabSize), adagradBiasMem = epsilon_vector(vocabSize);
    
    for(int i=0; i<numIter; i++){
        
        // adjust learning rate
        rate = learningRate/(i+1);
        cerr << "Iteration: " << i+1 << "\n";
        cerr << "Learning rate: " << rate << "\n";
        
        ifstream inputFile(fileName);
        string line, normWord;
        vector<string> tokens;
        vector<unsigned int> words;
        unsigned int numWords = 0, printIf = 1000;
    
        if (inputFile.is_open()) {
            
            while(getline(inputFile, line)) {
                
                // extract normalized words from sentences
                tokens = split_line(line, ' ');
                for(int i=0; i<tokens.size(); i++)
                    if (word2norm.find(tokens[i]) != word2norm.end())
                        words.push_back(indexedVocab[word2norm[tokens[i]]]);
                
                // Train word vectors now
                train_word_vectors(words, wordVectors, wordBiases, adagradVecMem, adagradBiasMem, 
                                    noiseDist, numNoiseWords, vocabSize, windowSize, rate);
                
                numWords += words.size();
                words.erase(words.begin(), words.end());
                if (numWords > printIf){
                    cerr << numWords << " ";
                    printIf += 1000;
                }
            }
            inputFile.close();
            cerr << "\n";
        }
        else
            cout << "\nUnable to open file\n";
    }
    
    return make_pair(wordVectors, indexedVocab);
}
