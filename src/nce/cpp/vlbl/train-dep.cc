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

vector<vector<unsigned>>
dep_new_context(vector<unsigned>& words, vector<unsigned>& deps) {
  vector<vector<unsigned>> contextWords;
  for (unsigned i=0; i<words.size(); ++i)
    contextWords.push_back(vector<unsigned>());
  
  for (unsigned tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
    unsigned toWrdIx = deps[tgtWrdIx];
    if (toWrdIx != -1) {
      /* Push the word to which this word points in this word's context */
      contextWords[tgtWrdIx].push_back(words[toWrdIx]);
      /* Push this word in the context of the word this word points to */
      contextWords[toWrdIx].push_back(words[tgtWrdIx]);
    }
  }
  return contextWords;
}

vector<unsigned> 
dep_context(vector<unsigned>& words, vector<unsigned>& deps, unsigned tgtWrdIx) {
  vector<unsigned> contextWords;
  if (deps[tgtWrdIx] != -1) 
    contextWords.push_back(words[deps[tgtWrdIx]]);
  for (unsigned i=0; i<words.size(); ++i)
    if (deps[i] == tgtWrdIx)
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
    wordVectors = normal_vector(vocabSize, vecLen); 
    wordBiases = normal_vector(vocabSize);
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
  
  double prob_model_to_noise(unsigned tgtWord, RowVectorXf contextVec, 
                             double biasSum) {
    double lpThetaTgt = wordVectors[tgtWord].dot(contextVec) + biasSum;
    double lpNoiseTgt = logNumNoiseWords+noiseDist[tgtWord];
    double pNoiseToThetaTgt = 1/(1+exp(lpThetaTgt-lpNoiseTgt));
    assert (isfinite(pNoiseToThetaTgt));
    return 1-pNoiseToThetaTgt;
  }
  
  void train_nce(vector<unsigned>& words, vector<unsigned>& deps, 
                 unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    /* Get context words of all the target words in O(2*n) */
    vector<vector<unsigned>> allContextWords = dep_new_context(words, deps);
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      vector<unsigned> contextWords = allContextWords[tgtWrdIx];
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
  train_on_corpus(string corpus, string depCorpus, unsigned iter, 
                  unsigned nCores, double lRate) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    double corpusSize = get_corpus_size(vocab, indexedVocab);
    time_t start, end;
    time(&start);
    cerr << "\nCorpus size: " << corpusSize << "\n";
    for (unsigned i=0; i<iter; ++i) {
      double rate = lRate/(i+1);
      cerr << "\n\nIteration: " << i+1;
      cerr << "\nLearning rate: " << rate << "\n";
      ifstream inputFile(corpus.c_str());
      ifstream depFile(depCorpus.c_str());
      string line, depLine, normWord;
      vector<unsigned> words, deps;
      unsigned numWords = 0, lineNum = 0;
      if (inputFile.is_open()) {
        time(&start);
        while (getline(inputFile, line) && getline(depFile, depLine)) {
          /* Extract normalized words from sentences */
          vector<string> tokens = split_line(line, ' ');
          vector<string> depStrings = split_line(depLine, ' ');
          words.clear(); deps.clear();
          for (unsigned j=0; j<tokens.size(); ++j) {
            words.push_back(indexedVocab[word2norm[tokens[j]]]);
            deps.push_back(atoi(depStrings[j].c_str())-1);
          }
          lineNum += 1;
          /* length of the word sequence and the dependency sequene */
          if(words.size() != deps.size()) {
            cerr << lineNum << ": " << words.size() << " " << deps.size() << "\n";
            continue;
          }  
          /* Train word vectors now */
          train_nce(words, deps, nCores, rate);
          numWords += words.size();
          cerr << int(numWords/1000) << "K\r";
        }
        inputFile.close();
        depFile.close();
        time(&end);
        cerr << "Time taken: " << float(difftime(end,start)/3600) << " hrs\n";
      }
      else {
        cerr << "\nUnable to open file\n";
        break;
      }
    }
  }
};

int main(int argc, char **argv){
  unsigned window = 5, freqCutoff = 10, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 1;
  double rate = 0.05;
  
  if (argc != 5) {
    cerr << "Usage: "<< argv[0] << " corpusName " << "depCorpus ";
    cerr << "outVecFileName " << "numCores\n";
    exit(0);
  }
  
  // Use the normalized corpus only as input !
  string corpus = argv[1];
  string depCorpus = argv[2];
  string outFile = argv[3];
  string outVecFile = outFile+"_vec.txt", outBiasFile = outFile+"_bias.txt";
  unsigned numCores = atoi(argv[4]);
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, depCorpus, numIter, numCores, rate);
  print_vectors(outVecFile, obj.wordVectors, obj.indexedVocab);
  print_biases(outBiasFile, obj.wordBiases, obj.indexedVocab);
  return 1;
}
