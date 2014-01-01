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
#include <random>
#include "utils.h"
#include "logadd.h"
#include "alias_sampler.h"

using namespace std;
using namespace Eigen;

#define MAX_EXP 10

vector<unsigned> 
context(vector<unsigned>& words, unsigned tgtWrdIx, unsigned windowSize) {
  vector<unsigned> contextWords;
  unsigned start, end, sentLen = words.size(), tgtWord=words[tgtWrdIx];
  start = (tgtWrdIx <= windowSize)? 0: tgtWrdIx-windowSize;
  end = (sentLen-tgtWrdIx <= windowSize)? sentLen-1: tgtWrdIx+windowSize;
  for (unsigned i=start; i<=end; ++i)
    if (i != tgtWrdIx)
       contextWords.push_back(words[i]);
  return contextWords;
}

/* Main class definition that learns the word vectors */

class WordVectorLearner {
  mapStrUnsigned vocab;
  vector<string> filtVocabList;
  mapUnsignedDouble noiseDist;
  mapStrStr word2norm;
  unsigned vocabSize, windowSize, numNoiseWords, freqCutoff, vecLen;
  double learningRate, logNumNoiseWords;
  vector<RowVectorXf> adagradVecMem;
  RowVectorXf adagradBiasMem;
  /* For sampling noise words */
  AliasSampler sampler;
    
public:
      
  vector<RowVectorXf> wordVectors;
  RowVectorXf wordBiases;
  mapStrUnsigned indexedVocab;
      
  WordVectorLearner(unsigned window, unsigned freq, unsigned noiseWords, unsigned vectorLen) {
    windowSize = window;
    freqCutoff = freq;
    numNoiseWords = noiseWords;
    logNumNoiseWords = log(numNoiseWords);
    vecLen = vectorLen;
  }
      
  void preprocess_vocab(string corpusName) {
    pair<mapStrUnsigned, mapStrStr> vocabPair = get_vocab(corpusName);
    vocab = vocabPair.first;
    word2norm = vocabPair.second;
    cerr << "Orig vocab " << vocab.size() << "\n";
    filtVocabList = filter_vocab(vocab, freqCutoff);
    cerr << "Filtered vocab " << filtVocabList.size() << "\n";
    indexedVocab = reindex_vocab(filtVocabList);
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
    vector<double> multinomial(vocabSize, 0.0);
    for (mapUnsignedDouble::iterator it = noiseDist.begin(); it != noiseDist.end(); ++it)
      multinomial[it->first] = exp(it->second);
    sampler.initialise(multinomial);
  }
  
  double diff_score_word_noise(unsigned word, vector<unsigned>& contextWords) {                   
    double sumScore = 0;
    for (unsigned i=0; i<contextWords.size(); ++i) {
      sumScore += wordVectors[word].dot(wordVectors[contextWords[i]]);
      sumScore += wordBiases[contextWords[i]];
    }
    return sumScore - logNumNoiseWords - noiseDist[word];
  }
  
  double log_lh(string corpus, unsigned nCores) {
    mapUnsignedDouble wordVocabScore;
    for (unsigned i=0; i<filtVocabList.size(); ++i) {
      unsigned word1 = indexedVocab[filtVocabList[i]], j;
      double logExpSum = 0;
      #pragma omp parallel for num_threads(2*nCores) shared(logExpSum) private(j)
      for (j=0; j<filtVocabList.size(); ++j) {
        if (j == i) continue;
        unsigned word2 = indexedVocab[filtVocabList[j]];
        double pairScore = wordVectors[word1].dot(wordVectors[word2]);
        pairScore += wordBiases[word2];
        logExpSum = log_add(pairScore, logExpSum);
      }
      wordVocabScore[word1] = logExpSum;
    }
    
    double lh = 0;
    ifstream inputFile(corpus.c_str());
    string line, normWord, token;
    vector<string> tokens;
    vector<unsigned> words;
    if (inputFile.is_open()) {
      while (getline(inputFile, line)) {
        /* Extract normalized words from sentences */
        tokens = split_line(line, ' ');
        words.clear();
        for (unsigned j=0; j<tokens.size(); ++j) {
          token = tokens[j];
          if (word2norm.find(token) != word2norm.end())
            words.push_back(indexedVocab[word2norm[token]]);
        }
        #pragma omp parallel for num_threads(nCores) shared(lh)
        for (unsigned tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
          unsigned tgtWord = words[tgtWrdIx];
          /* Get the words in the window context of the target word */
          vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
          /* Score word in context now */
          double x = diff_score_word_noise(tgtWord, contextWords);
          double contextScore = (x>MAX_EXP)? 1: (x<-MAX_EXP? 0: 1/(1+exp(-x)));
          /* log likelihood */
          lh += contextScore - contextWords.size()*wordVocabScore[tgtWord];
        }
      }
    inputFile.close();
    }
    else
      cout << "\nUnable to open file\n";
  return lh;
  }
    
  void train_word_vec(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      /* Get the words in the window context of the target word */
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      /* Get the diff of score of the word in context and the noise dist */
      double x = diff_score_word_noise(tgtWord, contextWords);
      double wordContextScore = (x>MAX_EXP)? 1: (x<-MAX_EXP? 0: 1/(1+exp(-x)));
      /* Select noise words for this target word */
      unsigned noiseWords[numNoiseWords];
      for (unsigned selWrds=0; selWrds<numNoiseWords; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* Get the diff of score of noise words in context and the noise dist */
      RowVectorXf noiseScoreGradProd(wordVectors[0].size());
      noiseScoreGradProd.setZero(noiseScoreGradProd.size());
      double noiseScoreSum=0;
      for (unsigned j=0; j<numNoiseWords; ++j) {
        double y = diff_score_word_noise(noiseWords[j], contextWords);
        double noiseScore = (y>MAX_EXP)? 1: (y<-MAX_EXP? 0: 1/(1+exp(-y)));
        noiseScoreSum += noiseScore;
        noiseScoreGradProd += noiseScore * wordVectors[noiseWords[j]];
      }
      /* Grad wrt bias is one, grad wrt contextWord is the target word */
      double updateInBias = 1 - wordContextScore - noiseScoreSum;
      RowVectorXf updateInVec = (1-wordContextScore) * wordVectors[tgtWord] - noiseScoreGradProd;
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

  void train_on_corpus(string corpus, unsigned iter, unsigned nCores, double lRate) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    for (unsigned i=0; i<iter; ++i) {
      double rate = lRate/(i+1);
      cerr << "Iteration: " << i+1 << "\n";
      cerr << "Learning rate: " << rate << "\n";
      ifstream inputFile(corpus.c_str());
      string line, normWord, token;
      vector<string> tokens;
      vector<unsigned> words;
      unsigned numWords = 0;
      if (inputFile.is_open()) {
        while (getline(inputFile, line)) {
          /* Extract normalized words from sentences */
          tokens = split_line(line, ' ');
          for (unsigned j=0; j<tokens.size(); ++j) {
            token = tokens[j];
            if (word2norm.find(token) != word2norm.end())
              words.push_back(indexedVocab[word2norm[token]]);
          }
          /* Train word vectors now */
          train_word_vec(words, nCores, rate);
          numWords += words.size();
          cerr << int(numWords/1000) << "K\r";
          words.clear();
        }
      inputFile.close();
      //cerr << "Log likelihood: " << log_lh(corpus, nCores);
      cerr << "\n";
      }
      else {
        cerr << "\nUnable to open file\n";
        break;
      }
    }
  }
};

int main(int argc, char **argv){
  string corpus = "../10k";
  unsigned window = 5, freqCutoff = 2, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 5, numCores = 6;
  double rate = 0.05;
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate);
  print_vectors("y.txt", obj.wordVectors, obj.indexedVocab);
  return 1;
}