#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <numeric>
#include <cmath>
#include <time.h>
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
      
  WordVectorLearner(unsigned window, unsigned freq, unsigned noiseWords, 
                    unsigned vectorLen) {
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
    mapUnsignedDouble::iterator it;
    for (it = noiseDist.begin(); it != noiseDist.end(); ++it)
      multinomial[it->first] = exp(it->second);
    sampler.initialise(multinomial);
  }
  
  double score_word_context(unsigned word, vector<unsigned>& contextWords) {       
    double sumScore = 0;
    for (unsigned i=0; i<contextWords.size(); ++i) 
      sumScore += wordVectors[word].dot(wordVectors[contextWords[i]]);
    sumScore /= contextWords.size();
    sumScore += wordBiases[word];
    return sumScore;
  }
  
  double log_lh(string corpus, unsigned nCores) {
    double lh = 0;
    ifstream inputFile(corpus.c_str());
    string line, normWord;
    vector<unsigned> words;
    if (inputFile.is_open()) {
      while (getline(inputFile, line)) {
        /* Extract normalized words from sentences */
        vector<string> tokens = split_line(line, ' ');
        words.clear();
        for (unsigned j=0; j<tokens.size(); ++j)
          if (word2norm.find(tokens[j]) != word2norm.end())
            words.push_back(indexedVocab[word2norm[tokens[j]]]);
        
        unsigned tgtWrdIx;
        #pragma omp parallel num_threads(nCores) shared(lh)
        #pragma omp for nowait private(tgtWrdIx)
        for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
          unsigned tgtWord = words[tgtWrdIx];
          vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
          double contextScore = score_word_context(tgtWord, contextWords);
          /* get vocab context score */
          double vocabContextScore = 0;
          for (unsigned v=0; v<filtVocabList.size(); ++v) {
            unsigned vWord = indexedVocab[filtVocabList[v]];
            double scoreWordContext = score_word_context(vWord, contextWords);
            vocabContextScore = log_add(vocabContextScore, scoreWordContext);
          }
          #pragma omp critical
          {lh += contextScore - vocabContextScore;}
        }
      }
    inputFile.close();
    }
    else
      cout << "\nUnable to open file\n";
  return lh;
  }
  
  void update(unsigned tgtWord, vector<unsigned>& context, )
  
  void train_nce(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      /* Get the words in the window context of the target word */
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      /* Get the ratio of probab scores of theta and noise */
      double pThetaTgt = exp(score_word_context(tgtWord, contextWords));
      double pNoiseTgt = numNoiseWords*exp(noiseDist[tgtWord]);
      double pNoiseToThetaTgt = pNoiseTgt/(pNoiseTgt+pThetaTgt);
      /* Select noise words for this target word */
      unsigned noiseWords[numNoiseWords];
      for (unsigned selWrds=0; selWrds<numNoiseWords; ++selWrds) {
        noiseWords[selWrds] = sampler.Draw();
      }
      /* Get the sum of context vectors */
      RowVectorXf contextVecAvg(vecLen);contextVecAvg.setZero(vecLen);
      for (unsigned c=0; c<contextWords.size(); ++c)
        contextVecAvg += wordVectors[contextWords[c]];
      contextVecAvg /= contextWords.size();
      /* Get the diff of score of noise words in context and the noise dist */
      RowVectorXf pThetaToNoiseGradProd(vecLen);
      pThetaToNoiseGradProd.setZero(vecLen);
      double pThetaToNoiseSum=0;
      for (unsigned j=0; j<numNoiseWords; ++j) {
        double pTheta = exp(score_word_context(noiseWords[j], contextWords));
        double pNoise = numNoiseWords*exp(noiseDist[noiseWords[j]]);
        double pThetaToNoise = pTheta/(pNoise+pTheta);
        pThetaToNoiseSum += pThetaToNoise;
        pThetaToNoiseGradProd += pThetaToNoise*wordVectors[noiseWords[j]];
      }
      /* Update adagrad params and add the updates to the context words now */
      RowVectorXf delContextVec = pNoiseToThetaTgt * wordVectors[tgtWord];
      delContextVec -= pThetaToNoiseGradProd;
      delContextVec /= contextWords.size();
      RowVectorXf delContextVecSquare = delContextVec.array().square();
      for (unsigned k=0; k<contextWords.size(); ++k) {
        unsigned contextWord = contextWords[k];
        adagradVecMem[contextWord] += delContextVecSquare;
        RowVectorXf temp = adagradVecMem[contextWord].array().sqrt();
        wordVectors[contextWord] += rate * delContextVec.cwiseQuotient(temp);
      }
      /* Update the target word vector */
      RowVectorXf delTgtVec = pNoiseToThetaTgt*contextVecAvg;
      RowVectorXf delTgtVecSquare = delTgtVec.array().square();
      adagradVecMem[tgtWord] += delTgtVecSquare;
      RowVectorXf temp = adagradVecMem[tgtWord].array().sqrt();
      wordVectors[tgtWord] += rate * delTgtVec.cwiseQuotient(temp);
      /* Update target bias */
      double delTgtBias = pNoiseToThetaTgt;
      double delTgtBiasSquare = delTgtBias*delTgtBias;
      adagradBiasMem[tgtWord] += delTgtBiasSquare;
      delTgtBias *= rate / sqrt(adagradBiasMem[tgtWord]);
      wordBiases[tgtWord] += delTgtBias;
    }
  }

  void 
  train_on_corpus(string corpus, unsigned iter, unsigned nCores, double lRate) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    time_t start, end;
    time(&start);
    /*cerr << "\nLog likelihood: " << log_lh(corpus, nCores);
    time(&end);
    cerr << "\nTime taken: " << float(difftime(end,start)/3600) << " hrs";*/
    for (unsigned i=0; i<iter; ++i) {
      double rate = lRate/(i+1);
      cerr << "\n\nIteration: " << i+1;
      cerr << "\nLearning rate: " << rate << "\n";
      ifstream inputFile(corpus.c_str());
      string line, normWord;
      vector<unsigned> words;
      unsigned numWords = 0;
      if (inputFile.is_open()) {
        time(&start);
        while (getline(inputFile, line)) {
          /* Extract normalized words from sentences */
          vector<string> tokens = split_line(line, ' ');
          words.clear();
          for (unsigned j=0; j<tokens.size(); ++j)
            if (word2norm.find(tokens[j]) != word2norm.end())
              words.push_back(indexedVocab[word2norm[tokens[j]]]);
          /* Train word vectors now */
          train_nce(words, nCores, rate);
          numWords += words.size();
          cerr << int(numWords/1000) << "K\r";
        }
        inputFile.close();
        time(&end);
        cerr << "Time taken: " << float(difftime(end,start)/3600) << " hrs\n";
        /*time(&start);
        cerr << "\nLog likelihood: " << log_lh(corpus, nCores);
        time(&end);
        cerr << "\nTime taken: " << float(difftime(end,start)/3660) << " hrs";*/
      }
      else {
        cerr << "\nUnable to open file\n";
        break;
      }
    }
  }
};

int main(int argc, char **argv){
  string corpus = "corpora/news.2011.en.norm";
  unsigned window = 5, freqCutoff = 10, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 1, numCores = 16;
  double rate = 0.05;
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate);
  print_vectors("correct-news.txt", obj.wordVectors, obj.indexedVocab);
  return 1;
}