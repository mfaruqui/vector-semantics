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
    adagradVecMem = random_vector(vocabSize, vecLen);
    adagradBiasMem = random_vector(vocabSize);
  }
    
  void set_noise_dist() {
    noiseDist = get_log_unigram_dist(vocab, indexedVocab);
    vector<double> multinomial(vocabSize, 0.0);
    mapUnsignedDouble::iterator it;
    for (it = noiseDist.begin(); it != noiseDist.end(); ++it)
      multinomial[it->first] = exp(it->second);
    sampler.initialise(multinomial);
  }
  
  double prob_model_to_noise(unsigned tgtWord, RowVectorXf contextVec, 
                             double biasSum) {
    double lpThetaTgt = wordVectors[tgtWord].dot(contextVec) + biasSum;
    double lpNoiseTgt = logNumNoiseWords+noiseDist[tgtWord];
    double pNoiseToThetaTgt = 1/(1+exp(lpThetaTgt-lpNoiseTgt));
    assert (isfinite(pNoiseToThetaTgt));
    return 1-pNoiseToThetaTgt;
  }
  
  double log_lh(string corpus, unsigned nCores) {
    double lh = 0;
    ifstream inputFile(corpus.c_str());
    string line, normWord;
    vector<unsigned> words;
    if (inputFile.is_open()) {
      cerr << "Calculating LLH...\n";
      while (getline(inputFile, line)) {
        /* Extract normalized words from sentences */
        vector<string> tokens = split_line(line, ' ');
        words.clear();
        for (unsigned j=0; j<tokens.size(); ++j)
          if (word2norm.find(tokens[j]) != word2norm.end())
            words.push_back(indexedVocab[word2norm[tokens[j]]]);
        /* Calculate likelihood */
        unsigned tgtWrdIx;
        #pragma omp parallel num_threads(nCores) shared(lh)
        #pragma omp for nowait private(tgtWrdIx)
        for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
          unsigned tgtWord = words[tgtWrdIx];
          vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
          if (contextWords.size() < 1) continue;
          /* Get the sum of context vectors */
          RowVectorXf contextVec(vecLen);contextVec.setZero(vecLen);
          double biasSum = 0;
          for (unsigned c=0; c<contextWords.size(); ++c) {
            contextVec += wordVectors[contextWords[c]];
            biasSum += wordBiases[contextWords[c]];
          }
          double contextScore = wordVectors[tgtWord].dot(contextVec) + biasSum;
          /* Get vocab context score */
          double vocabContextScore = 0;
          for (unsigned v=0; v<filtVocabList.size(); ++v) {
            unsigned vWord = indexedVocab[filtVocabList[v]];
            double scoreWordContext = wordVectors[vWord].dot(contextVec) + biasSum;
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
  
  void train_nce(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      /* Get the words in the window context of the target word */
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      if (contextWords.size() < 1) continue;
      RowVectorXf contextVec(vecLen);contextVec.setZero(vecLen);
      double contextBias = 0;
      for (unsigned c=0; c<contextWords.size(); ++c) {
        contextVec += wordVectors[contextWords[c]];
        contextBias += wordBiases[contextWords[c]];
      }
      /* Get the ratio of probab scores of theta and noise */
      double pNoiseToThetaTgt = 1-prob_model_to_noise(tgtWord, contextVec,
                                                      contextBias);
      /* Select noise words for this target word */
      unsigned noiseWords[numNoiseWords];
      for (unsigned selWrds=0; selWrds<numNoiseWords; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* Get the diff of score of noise words in context and the noise dist */
      RowVectorXf pThetaToNoiseGradProd(vecLen);
      pThetaToNoiseGradProd.setZero(vecLen);
      double pThetaToNoiseSum=0;
      for (unsigned j=0; j<numNoiseWords; ++j) {
        double pThetaToNoise = prob_model_to_noise(noiseWords[j], contextVec,
                                                   contextBias);
        pThetaToNoiseSum += pThetaToNoise;
        pThetaToNoiseGradProd += pThetaToNoise * wordVectors[noiseWords[j]];
      }
      /* Calculate updates */
      RowVectorXf delTgtVec = pNoiseToThetaTgt * contextVec;
      double delConBias = pNoiseToThetaTgt - pThetaToNoiseSum;
      RowVectorXf delConVec = pNoiseToThetaTgt * wordVectors[tgtWord];
      delConVec -= pThetaToNoiseGradProd;
      /* Apply the updates */
      update(tgtWord, contextWords, delTgtVec, delConVec, delConBias, rate);
    }
  }
  
  void update(unsigned tgtWord, vector<unsigned>& contextWords, 
              RowVectorXf delTgtVec, RowVectorXf delConVec, double delConBias,
              double rate) {
    /* Update the target word vector */
    RowVectorXf delTgtVecSq = delTgtVec.array().square();
    adagradVecMem[tgtWord] += delTgtVecSq;
    RowVectorXf temp = adagradVecMem[tgtWord].array().sqrt();
    wordVectors[tgtWord] += rate * delTgtVec.cwiseQuotient(temp);
    /* Update the context word vectors */
    RowVectorXf delConVecSq = delConVec.array().square();
    double delConBiasSq = delConBias * delConBias;
    for (unsigned k=0; k<contextWords.size(); ++k) {
      unsigned conWord = contextWords[k];
      /* Update the adagrad memory first */
      adagradVecMem[conWord] += delConVecSq;
      adagradBiasMem[conWord] += delConBiasSq;
      /* Now apply the updates */
      RowVectorXf temp = adagradVecMem[conWord].array().sqrt();
      wordVectors[conWord] += rate*delConVec.cwiseQuotient(temp);
      wordBiases[conWord] += rate*delConBias/sqrt(adagradBiasMem[conWord]);
    }
  }

  void 
  train_on_corpus(string corpus, unsigned iter, unsigned nCores, double lRate) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    double corpusSize = get_corpus_size(vocab, indexedVocab);
    time_t start, end;
    time(&start);
    cerr << "\nCorpus size: " << corpusSize << "\n";
    cerr << "\nLLH per word: \n" << log_lh(corpus, nCores)/corpusSize;
    time(&end);
    cerr << "\nTime taken: " << float(difftime(end,start)/3600) << " hrs";
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
          cerr << int(numWords/10000) << "0K\r";
        }
        inputFile.close();
        time(&end);
        cerr << "Time taken: " << float(difftime(end,start)/3600) << " hrs\n";
        time(&start);
        cerr << "\nLLH per word: \n";
        cerr << log_lh(corpus, nCores)/corpusSize;
        time(&end);
        cerr << "\nTime taken: " << float(difftime(end,start)/3660) << " hrs";
      }
      else {
        cerr << "\nUnable to open file\n";
        break;
      }
    }
  }
};

int main(int argc, char **argv){
  string corpus = "../100";
  unsigned window = 5, freqCutoff = 1, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 5, numCores = 10;
  double rate = 0.05;
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate);
  //print_vectors("vectors/base-sum-vec.txt", obj.wordVectors, obj.indexedVocab);
  //print_biases("vectors/base-sum-bias.txt", obj.wordBiases, obj.indexedVocab);
  return 1;
}