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

typedef std::tr1::unordered_map<string, unsigned> mapStrunsigned;
typedef std::tr1::unordered_map<string, string> mapStrStr;
typedef std::tr1::unordered_map<unsigned, double> mapunsigneddouble;
typedef std::tr1::unordered_map<unsigned, unsigned> mapunsignedunsigned;

/* =================== Utility functions begin =================== */

double EPSILON = 0.00000000000000000001;

string normalize_word(string& word) {
    if (std::string::npos != word.find_first_of("0123456789"))
        return "---num---";
    for (unsigned i=0; i<word.length(); ++i)
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

pair<mapStrunsigned, mapStrStr> get_vocab(string filename) {
    string line, normWord;
    vector<string> words;
    mapStrunsigned vocab;
    mapStrStr word2norm;
    ifstream myfile(filename.c_str());
    if (myfile.is_open()) {
        while(getline(myfile, line)) {
            words = split_line(line, ' ');
            for(unsigned i=0; i<words.size(); i++){
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
while prunsigneding its still there ! x-(
http://stackoverflow.com/questions/17036428/c-map-element-doesnt-get-erased-if-i-refer-to-it
*/
mapStrunsigned filter_vocab(mapStrunsigned& vocab, const unsigned freqCutoff) {
    mapStrunsigned filtVocab;
    for (mapStrunsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
        if (! (it->second < freqCutoff) )
            filtVocab[it->first] = it->second;
    return filtVocab;
}

mapStrunsigned reindex_vocab(mapStrunsigned& vocab) {
    unsigned index = 0;
    mapStrunsigned indexedVocab;
    for (mapStrunsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
        indexedVocab[it->first] = index++;
    return indexedVocab;
}

mapunsigneddouble get_log_unigram_dist(mapStrunsigned& vocab, mapStrunsigned& indexedVocab) {
    double sumFreq = 0;
    mapunsigneddouble unigramDist;
    for (mapStrunsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
        sumFreq += it->second;
    for (mapStrunsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
        unigramDist[indexedVocab[it->first]] = log(it->second/sumFreq);
    return unigramDist;
}

void prunsigned_map(mapStrunsigned& vocab) {
    for (mapStrunsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
        cout << it->first << " " << it->second << "\n";
}

void prunsigned_map(mapunsigneddouble& vocab) {
    for (mapunsigneddouble::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
}

void prunsigned_map(mapStrStr& vocab) {
    for (mapStrStr::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
}

RowVectorXf epsilon_vector(unsigned row) {
    RowVectorXf nonZeroVec(row);
    nonZeroVec.setOnes(row);
    nonZeroVec *= EPSILON;
    return nonZeroVec;
}

vector<RowVectorXf> epsilon_vector(unsigned row, unsigned col) {
    vector<RowVectorXf> epsilonVec;
    RowVectorXf vec = epsilon_vector(col);
    for (unsigned i=0; i<row; ++i)
        epsilonVec.push_back(vec);
    return epsilonVec;
}

RowVectorXf random_vector(const unsigned length) {
    RowVectorXf randVec(length);
    for (unsigned i=0; i<randVec.size(); ++i)
        randVec[i] = (rand()/(double)RAND_MAX);
    randVec /= randVec.norm();
    return randVec;
}

vector<RowVectorXf> random_vector(unsigned row, unsigned col) {
    vector<RowVectorXf> randVec;
    for (unsigned i=0; i<row; ++i){
        randVec.push_back(random_vector(col));
    }
    return randVec;
}

void prunsigned_vectors(char* fileName, vector<RowVectorXf>& wordVectors, 
                   mapStrunsigned& indexedVocab) {
    ofstream outFile(fileName);
    for (mapStrunsigned::iterator it=indexedVocab.begin(); it!= indexedVocab.end(); it++){
        /* This check is wrong but I have to put it, coz of the elements not getting deleted :(
         By this we will be missing the word at index 0. */
         if (it->second != 0){
            outFile << it->first << " ";
            for (unsigned i=0; i != wordVectors[it->second].size(); ++i)
                outFile << wordVectors[it->second][i] << " ";
            outFile << "\n";
        }
    }
}

double logistic(double val) {
    if (val > 20) return 1;
    else if (val < -20) return 0.;
    else return 1/(1+exp(-1*val));
}

/* =================== Utility functions end =================== */

/* Selects words randomly for now */
vector<unsigned> get_noise_words(unsigned numNoiseWords, unsigned vocabSize) {
    vector<unsigned> noiseWords;
    unsigned selectedWords = 0;
    while (selectedWords != numNoiseWords){
        noiseWords.push_back(rand() % vocabSize);
        ++selectedWords;
    }
    return noiseWords;
}

vector<unsigned> words_in_window(vector<unsigned>& words, unsigned wordIndex, 
                            unsigned windowSize) {
    vector<unsigned> wordsInWindow;
    unsigned start, end, sentLen = words.size();
    start = (wordIndex <= windowSize) ? 0 : wordIndex-windowSize;
    end = (sentLen - wordIndex <= windowSize) ? sentLen-1 : wordIndex + windowSize;
    for (unsigned i=start; i<wordIndex; ++i)
        wordsInWindow.push_back(words[i]);
    for (unsigned i=wordIndex+1; i<=end; ++i)
        wordsInWindow.push_back(words[i]);
    return wordsInWindow;
}

double diff_score_word_and_noise(unsigned word, vector<unsigned>& contextWords,
                                unsigned numNoiseWords, mapunsigneddouble& noiseDist,
                                RowVectorXf& wordBiases,
                                vector<RowVectorXf >& wordVectors,
                                double logNumNoiseWords) {                     
    double sumScore = 0;
    #pragma omp parallel for reduction(+:sumScore) num_threads(5)
    for (unsigned i=0; i<contextWords.size(); ++i)
        sumScore += wordVectors[word].dot(wordVectors[contextWords[i]]) + wordBiases[contextWords[i]];

    return sumScore - logNumNoiseWords - noiseDist[word];
}

/* Main class definition that learns the word vectors */

class WordVectorLearner {
    mapStrunsigned vocab;
    mapunsigneddouble noiseDist;
    mapStrStr word2norm;
    unsigned vocabSize, windowSize, numNoiseWords, freqCutoff, vecLen;
    double learningRate, logNumNoiseWords;
    string corpusName;
    vector<RowVectorXf> adagradVecMem;
    RowVectorXf wordBiases, adagradBiasMem;
    
  public:
      
    vector<RowVectorXf> wordVectors;
    mapStrunsigned indexedVocab;
    WordVectorLearner(unsigned window, unsigned freq, unsigned noiseWords, unsigned vectorLen) {
        windowSize = window;
        freqCutoff = freq;
        numNoiseWords = noiseWords;
        logNumNoiseWords = log(numNoiseWords);
        vecLen = vectorLen;
    }
      
    void preprocess_vocab(string corpusName) {
        pair<mapStrunsigned, mapStrStr> vocabPair = get_vocab(corpusName);
        vocab = vocabPair.first;
        word2norm = vocabPair.second;
        cerr << "Orig vocab " << vocab.size() << "\n";
        vocab = filter_vocab(vocab, freqCutoff);
        cerr << "Filtered vocab " << vocab.size() << "\n";
        indexedVocab = reindex_vocab(vocab);
        vocabSize = indexedVocab.size();
    }
    
    void init_vectors(unsigned vocabSize, unsigned vecLen) {
        wordVectors = random_vector(vocabSize, vecLen); 
        wordBiases = random_vector(vocabSize);
        adagradVecMem = epsilon_vector(vocabSize, vecLen);
        adagradBiasMem = epsilon_vector(vocabSize);
    }
    
    void set_noise_dist() { 
        noiseDist = get_log_unigram_dist(vocab, indexedVocab);
        
    }
    
    void train_word_vectors(vector<unsigned>& words, double rate) {
        for (unsigned i=0; i<words.size(); ++i) {
            vector<unsigned> noiseWords = get_noise_words(numNoiseWords, vocabSize);
            /* Get the words in the window context of the target word */
            vector<unsigned> contextWords = words_in_window(words, i, windowSize);
            /* Get the diff of score of the word in context and the noise dist */
            double wordContextScore = logistic(diff_score_word_and_noise(words[i], contextWords, numNoiseWords,
                                                                  noiseDist, wordBiases, wordVectors, logNumNoiseWords));
            /* Get the diff of score of the noise words in context and the noise dist */     
            RowVectorXf noiseScoreGradProd(wordVectors[0].size());
            noiseScoreGradProd.setZero(noiseScoreGradProd.size());
            double noiseScoreSum = 0;
            for (unsigned j=0; j<noiseWords.size(); ++j) {
                double noiseScore = logistic(diff_score_word_and_noise(noiseWords[j], contextWords, numNoiseWords,
                                                                noiseDist, wordBiases, wordVectors, logNumNoiseWords));
                noiseScoreSum += noiseScore;
                noiseScoreGradProd += noiseScore * wordVectors[noiseWords[j]];
            }
            /* Grad wrt bias is one, grad wrt contextWord is the target word */
            double updateInBias = 1 - wordContextScore - noiseScoreSum;
            RowVectorXf updateInVec = (1 - wordContextScore) * wordVectors[words[i]] - noiseScoreGradProd;
            /* Update adagrad params and add the updates to the context words now */
            RowVectorXf updateVecSquare = updateInVec.array().square();
            double updateBiasSquare = updateInBias*updateInBias;
            for (unsigned k=0; k<contextWords.size(); ++k) {
                unsigned contextWord = contextWords[k];
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

    void train_on_corpus(string corpusName, unsigned iter, double learningRate) {
        preprocess_vocab(corpusName);
        init_vectors(vocabSize, vecLen);
        set_noise_dist();
        for (unsigned i=0; i<iter; ++i) {
            double rate = learningRate/(i+1);
            cerr << "Iteration: " << i+1 << "\n";
            cerr << "Learning rate: " << rate << "\n";
            ifstream inputFile(corpusName.c_str());
            string line, normWord, token;
            vector<string> tokens;
            vector<unsigned> words;
            unsigned numWords = 0;
            if (inputFile.is_open()) {
                while (getline(inputFile, line)) {
                    /* Extract normalized words from sentences */
                    tokens = split_line(line, ' ');
                    for (unsigned i=0; i<tokens.size(); ++i){
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
    string corpus = "../10k";
    unsigned window = 5, freqCutoff = 2, noiseWords = 10, vectorLen = 80, numIter = 1;
    double rate = 0.05;
    
    WordVectorLearner obj (window, freqCutoff, noiseWords, vectorLen);
    obj.train_on_corpus(corpus, numIter, rate);
    prunsigned_vectors("news-n5-80.txt", obj.wordVectors, obj.indexedVocab);
    return 1;
}