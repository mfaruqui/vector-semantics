/*  Copyright 2013 Manaal Faruqui. All Rights Reserved.
    
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
            
                  http://www.apache.org/licenses/LICENSE-2.0
  
  Code to compute word vector representations using the 
  Inverse Log bilinear Model presented in (Mnih and Kavukcuoglu, 2013)
  and trained using Noise Contrastive Estimation (Gutmann and Hyvarinen 2010).
*/

#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <numeric>
#include <cmath>
#include <ctime>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>
#include <iomanip>

using namespace std;
using namespace Eigen;

typedef std::tr1::unordered_map<string, int> mapStrInt;
typedef std::tr1::unordered_map<string, string> mapStrStr;
typedef std::tr1::unordered_map<int, float> mapIntFloat;
typedef std::tr1::unordered_map<int, int> mapIntInt;

/* =================== Utility functions begin =================== */

float EPSILON = 0.00000000000000000001;

string normalize_word(string& word) {
    if (std::string::npos != word.find_first_of("0123456789"))
        return "---num---";
    for (int i=0; i<word.length(); ++i)
        if (isalnum(word[i])){
            transform(word.begin(), word.end(), word.begin(), ::tolower);
            return word;
        }
    return "---punc---";
}

/* Try splitting over all whitespaces not just space */
vector<string> split_line(string& line, char delim) {
    vector<string> words;
    stringstream ss(line);
    string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            words.push_back(item);
    }
    return words;
}

pair<mapStrInt, mapStrStr> get_vocab(string filename) {
    string line, normWord;
    vector<string> words;
    mapStrInt vocab;
    mapStrStr word2norm;
    ifstream myfile(filename.c_str());
    if (myfile.is_open()) {
        while(getline(myfile, line)) {
            words = split_line(line, ' ');
            for(int i=0; i<words.size(); i++){
                normWord = normalize_word(words[i]);
                if (word2norm.find(words[i]) == word2norm.end())
                    word2norm[words[i]] = normWord;
                vocab[normWord]++;
            }
        }
        myfile.close();
    }
    else
        cout << "Unable to open file";
    return make_pair(vocab, word2norm);
}

/* 
It is not deleting stuff, dont know why !
while printing its still there ! x-(
http://stackoverflow.com/questions/17036428/c-map-element-doesnt-get-erased-if-i-refer-to-it
*/
mapStrInt filter_vocab(mapStrInt& vocab, const int freqCutoff) {
    mapStrInt filtVocab;
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        if (! (it->second < freqCutoff) )
            filtVocab[it->first] = it->second;
    return filtVocab;
}

mapStrInt reindex_vocab(mapStrInt& vocab) {
    int index = 0;
    mapStrInt indexedVocab;
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        indexedVocab[it->first] = index++;
    return indexedVocab;
}

mapIntFloat get_log_unigram_dist(mapStrInt& vocab, mapStrInt& indexedVocab) {
    float sumFreq = 0;
    mapIntFloat unigramDist;
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        sumFreq += it->second;
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        unigramDist[indexedVocab[it->first]] = log(it->second/sumFreq);
    return unigramDist;
}

void print_map(mapStrInt& vocab) {
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        cout << it->first << " " << it->second << "\n";
}

void print_map(mapIntFloat& vocab) {
    for (mapIntFloat::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
}

void print_map(mapStrStr& vocab) {
    for (mapStrStr::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
}

RowVectorXf epsilon_vector(int row) {
    RowVectorXf nonZeroVec(row);
    nonZeroVec.setOnes(row);
    nonZeroVec *= EPSILON;
    return nonZeroVec;
}

vector<RowVectorXf> epsilon_vector(int row, int col) {
    vector<RowVectorXf> epsilonVec;
    RowVectorXf vec = epsilon_vector(col);
    for (int i=0; i<row; ++i)
        epsilonVec.push_back(vec);
    return epsilonVec;
}

RowVectorXf random_vector(const int length) {
    RowVectorXf randVec(length);
    for (int i=0; i<randVec.size(); ++i)
        randVec[i] = (rand()/(double)RAND_MAX);
    randVec /= randVec.norm();
    return randVec;
}

vector<RowVectorXf> random_vector(int row, int col) {
    vector<RowVectorXf> randVec;
    for (int i=0; i<row; ++i){
        randVec.push_back(random_vector(col));
    }
    return randVec;
}

void print_vectors(char* fileName, vector<RowVectorXf>& wordVectors, 
                   mapStrInt& indexedVocab) {
    ofstream outFile(fileName);
    for (mapStrInt::iterator it=indexedVocab.begin(); it!= indexedVocab.end(); it++){
        /* This check is wrong but I have to put it, coz of the elements not getting deleted :(
         By this we will be missing the word at index 0. */
         if (it->second != 0){
            outFile << it->first << " ";
            for (int i=0; i != wordVectors[it->second].size(); ++i)
                outFile << wordVectors[it->second][i] << " ";
            outFile << "\n";
        }
    }
}

float logistic(float val) {
    if (val > 20) return 1;
    else if (val < -20) return 0.;
    else return 1/(1+exp(-1*val));
}

/* =================== Utility functions end =================== */

/* Selects words randomly for now */
vector<int> get_noise_words(int numNoiseWords, int vocabSize) {
    vector<int> noiseWords;
    int selectedWords = 0;
    while (selectedWords != numNoiseWords){
        noiseWords.push_back(rand() % vocabSize);
        ++selectedWords;
    }
    return noiseWords;
}

vector<int> words_in_window(vector<int>& words, int wordIndex, 
                            int windowSize) {
    vector<int> wordsInWindow;
    int start, end, sentLen = words.size();
    start = (wordIndex <= windowSize) ? 0 : wordIndex-windowSize;
    end = (sentLen - wordIndex <= windowSize) ? sentLen-1 : wordIndex + windowSize;
    for (int i=start; i<wordIndex; ++i)
        wordsInWindow.push_back(words[i]);
    for (int i=wordIndex+1; i<=end; ++i)
        wordsInWindow.push_back(words[i]);
    return wordsInWindow;
}

float diff_score_word_and_noise(int word, vector<int>& contextWords,
                                int numNoiseWords, mapIntFloat& noiseDist,
                                RowVectorXf& wordBiases,
                                vector<RowVectorXf >& wordVectors,
                                float logNumNoiseWords) {                     
    float sumScore = 0;
    #pragma omp parallel for reduction(+:sumScore) num_threads(2)
    for (int i=0; i<contextWords.size(); ++i)
        sumScore += wordVectors[word].dot(wordVectors[contextWords[i]]) + wordBiases[contextWords[i]];

    return sumScore - logNumNoiseWords - noiseDist[word];
}

/* Main class definition that learns the word vectors */

class WordVectorLearner {
    mapStrInt vocab;
    mapIntFloat noiseDist;
    mapStrStr word2norm;
    int vocabSize, windowSize, numNoiseWords, freqCutoff, vecLen;
    float learningRate, logNumNoiseWords;
    string corpusName;
    vector<RowVectorXf> adagradVecMem;
    RowVectorXf wordBiases, adagradBiasMem;
    
  public:
      
    vector<RowVectorXf> wordVectors;
    mapStrInt indexedVocab;
    WordVectorLearner(int window, int freq, int noiseWords, int vectorLen) {
        windowSize = window;
        freqCutoff = freq;
        numNoiseWords = noiseWords;
        logNumNoiseWords = log(numNoiseWords);
        vecLen = vectorLen;
    }
      
    void preprocess_vocab(string corpusName) {
        pair<mapStrInt, mapStrStr> vocabPair = get_vocab(corpusName);
        vocab = vocabPair.first;
        word2norm = vocabPair.second;
        cerr << "Orig vocab " << vocab.size() << "\n";
        vocab = filter_vocab(vocab, freqCutoff);
        cerr << "Filtered vocab " << vocab.size() << "\n";
        indexedVocab = reindex_vocab(vocab);
        vocabSize = indexedVocab.size();
    }
    
    void init_vectors(int vocabSize, int vecLen) {
        wordVectors = random_vector(vocabSize, vecLen); 
        wordBiases = random_vector(vocabSize);
        adagradVecMem = epsilon_vector(vocabSize, vecLen);
        adagradBiasMem = epsilon_vector(vocabSize);
    }
    
    void set_noise_dist() { 
        noiseDist = get_log_unigram_dist(vocab, indexedVocab);
    }
    
    void train_word_vectors(vector<int>& words, float rate) {
        for (int i=0; i<words.size(); ++i) {
            vector<int> noiseWords = get_noise_words(numNoiseWords, vocabSize);
            /* Get the words in the window context of the target word */
            vector<int> contextWords = words_in_window(words, i, windowSize);
            /* Get the diff of score of the word in context and the noise dist */
            float wordContextScore = logistic(diff_score_word_and_noise(words[i], contextWords, numNoiseWords,
                                                                  noiseDist, wordBiases, wordVectors, logNumNoiseWords));
            /* Get the diff of score of the noise words in context and the noise dist */     
            RowVectorXf noiseScoreGradProd(wordVectors[0].size());
            noiseScoreGradProd.setZero(noiseScoreGradProd.size());
            float noiseScoreSum = 0;
            for (int j=0; j<noiseWords.size(); ++j) {
                float noiseScore = logistic(diff_score_word_and_noise(noiseWords[j], contextWords, numNoiseWords,
                                                                noiseDist, wordBiases, wordVectors, logNumNoiseWords));
                noiseScoreSum += noiseScore;
                noiseScoreGradProd += noiseScore * wordVectors[noiseWords[j]];
            }
            /* Grad wrt bias is one, grad wrt contextWord is the target word */
            float updateInBias = 1 - wordContextScore - noiseScoreSum;
            RowVectorXf updateInVec = (1 - wordContextScore) * wordVectors[words[i]] - noiseScoreGradProd;
            /* Update adagrad params and add the updates to the context words now */
            RowVectorXf updateVecSquare = updateInVec.array().square();
            float updateBiasSquare = pow(updateInBias, 2);
            for (int k=0; k<contextWords.size(); ++k) {
                int contextWord = contextWords[k];
                /* Update the adagrad memory first */
                adagradVecMem[contextWord] += updateVecSquare;
                adagradBiasMem[contextWord] += updateBiasSquare;
                /* Now apply the updates */
                RowVectorXf temp = adagradVecMem[contextWord].array().sqrt();
                wordVectors[contextWord] += rate * updateInVec.cwiseQuotient(temp);
                wordBiases[contextWord] += rate * updateInBias / sqrt(adagradBiasMem[contextWord]);
            }
        }
    }

    void train_on_corpus(string corpusName, int iter, float learningRate) {
        preprocess_vocab(corpusName);
        init_vectors(vocabSize, vecLen);
        set_noise_dist();
        for (int i=0; i<iter; ++i) {
            float rate = learningRate/(i+1);
            cerr << "Iteration: " << i+1 << "\n";
            cerr << "Learning rate: " << rate << "\n";
            ifstream inputFile(corpusName.c_str());
            string line, normWord, token;
            vector<string> tokens;
            vector<int> words;
            int numWords = 0;
            if (inputFile.is_open()) {
                while (getline(inputFile, line)) {
                    /* Extract normalized words from sentences */
                    tokens = split_line(line, ' ');
                    for (int i=0; i<tokens.size(); ++i){
                        token = tokens[i];
                        if (word2norm.find(token) != word2norm.end())
                           words.push_back(indexedVocab[word2norm[token]]);
                    }
                    /* Train word vectors now */
                    train_word_vectors(words, rate);
                    numWords += words.size();
                    cerr << numWords << "\r";
                    words.clear();
                }
                inputFile.close();
                cerr << "\n";
            }
            else
                cout << "\nUnable to open file\n";
        }
    }
};

int main(){
    string corpus = "../news.2011.en.norm";
    int window = 5, freqCutoff = 10, noiseWords = 10, vectorLen = 80, numIter = 1;
    float rate = 0.05;
    
    WordVectorLearner obj (window, freqCutoff, noiseWords, vectorLen);
    obj.train_on_corpus(corpus, numIter, rate);
    print_vectors("news-n2-80.txt", obj.wordVectors, obj.indexedVocab);
    return 1;
}