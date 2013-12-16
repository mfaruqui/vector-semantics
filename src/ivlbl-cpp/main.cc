#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Core>
#include "utils.h"
#include "ivlbl.h"
using namespace std;

class WordVectorLearner {
    mapStrInt vocab;
    mapIntFloat noiseDist;
    mapStrStr word2norm;
    int vocabSize, windowSize, numNoiseWords, freqCutoff, vecLen;
    float learningRate;
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
        vecLen = vectorLen;
    }
      
    void preprocess_vocab(string corpusName) {
        pair<mapStrInt, mapStrStr> vocabPair = get_vocab(corpusName);
        vocab = vocabPair.first;
        word2norm = vocabPair.second;
        cerr << "Orig vocab " << vocab.size();
        filter_vocab(vocab, freqCutoff);
        cerr << "Filtered vocab " << vocab.size();
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
        noiseDist = get_unigram_dist(vocab, indexedVocab);
    }
    
    void add_grad_to_words_adagrad(vector<int>& contextWords, 
                                   RowVectorXf& updateVec, 
                                   float updateBias, float rate) {
        RowVectorXf updateVecSquare = updateVec.array().square(), temp;
        float updateBiasSquare = pow(updateBias, 2);
        
        for (int i=0; i<contextWords.size(); i++) {
            int word = contextWords[i];
            /* Update the adagrad memory first */
            adagradVecMem[word] += updateVecSquare;
            adagradBiasMem[word] += updateBiasSquare;
            /* Now apply the updates */
            temp = adagradVecMem[word].array().sqrt();
            wordVectors[word] += rate * updateVec.cwiseQuotient(temp);
            wordBiases[word] += rate * updateBias / sqrt(adagradBiasMem[word]);
        }
    }
    
    void train_word_vectors(vector<int>& words, float rate) {
        vector<int> noiseWords = get_noise_words(words, numNoiseWords, vocabSize);
        int word, noiseWord;
        vector<int> contextWords;
        float wordContextScore, updateInBias, noiseScore, noiseScoreSum;
        RowVectorXf updateInVec, noiseScoreGradProd(wordVectors[0].size()), temp;
    
        for (int i=0; i<words.size(); i++) {
            noiseScoreSum = 0;
            noiseScoreGradProd.setZero(noiseScoreGradProd.size());
            word = words[i];
            contextWords = words_in_window(words, i, windowSize);
            wordContextScore = logistic(diff_score_word_and_noise(word, contextWords, numNoiseWords,
                                                                  noiseDist, wordBiases, wordVectors));
        
            for (int j=0; j<noiseWords.size(); j++) {
                noiseWord = noiseWords[j];
                noiseScore = logistic(diff_score_word_and_noise(noiseWord, contextWords, numNoiseWords,
                                                                noiseDist, wordBiases, wordVectors));
                noiseScoreSum += noiseScore;
                noiseScoreGradProd += noiseScore * grad_context_word(noiseWord, contextWords, 
                                                                     wordVectors);
            }
                                                        
            updateInBias = 1 - wordContextScore - noiseScoreSum;
            updateInVec = (1 - wordContextScore) * grad_context_word(word, contextWords, wordVectors)
                                 - noiseScoreGradProd;
            add_grad_to_words_adagrad(contextWords, updateInVec, updateInBias, rate);
        }  
    }
    
    void train_on_corpus(string corpusName, int iter, float learningRate) {
        preprocess_vocab(corpusName);
        init_vectors(vocabSize, vecLen);
        set_noise_dist();
    
        for (int i=0; i<iter; i++) {
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
                    // extract normalized words from sentences
                    tokens = split_line(line, ' ');
                    for (int i=0; i<tokens.size(); i++){
                        token = tokens[i];
                        if (word2norm.find(token) != word2norm.end())
                           words.push_back(indexedVocab[word2norm[token]]);
                    }
                
                    // Train word vectors now
                    train_word_vectors(words, rate);
                    numWords += words.size();
                    words.erase(words.begin(), words.end());
                    cerr << numWords << "\r";
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
    
    string corpus = "../1k";
    int window = 5, freqCutoff = 10, noiseWords = 10, vectorLen = 80, numIter = 10;
    float rate = 0.05;
    
    WordVectorLearner obj (window, freqCutoff, noiseWords, vectorLen);
    obj.train_on_corpus(corpus, numIter, rate);
    print_vectors("news-try.txt", obj.wordVectors, obj.indexedVocab);
    
    return 1;
}