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

/* Main class definition that learns the word vectors */

class WordVectorLearner {
  mapStrUnsigned vocab;
  vector<string> filtVocabList;
  mapUnsignedDouble noiseDist, logPriorDict;
  mapStrStr word2norm;
  unsigned vocabSize, windowSize, numNoiseWords, freqCutoff, vecLen;
  double learningRate, logNumNoiseWords, corpusSize, logPrior;
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
    corpusSize = get_corpus_size(vocab, indexedVocab);
  }
    
  void init_vectors(unsigned vocabSize, unsigned vecLen) {
    wordVectors = random_vector(vocabSize, vecLen); 
    wordBiases = random_vector(vocabSize);
    adagradVecMem = epsilon_vector(vocabSize, vecLen);
    adagradBiasMem = epsilon_vector(vocabSize);
  }
  
  /* Compute total prior and prior contribution of words */
  void compute_prior(mapLexParaP& pp) {
    for (auto it = pp.begin(); it != pp.end(); ++it) {
      double distance = 0;
      for (unsigned i=0; i<pp[it->first].size(); ++i) {
        unsigned ppWord = pp[it->first][i];
        RowVectorXf delVec = wordVectors[it->first]-wordVectors[ppWord];
        distance += delVec.squaredNorm();
      }
      logPriorDict[it->first] = -distance;
    }
    /* Compute the total prior */
    for (auto it = logPriorDict.begin(); it != logPriorDict.end(); ++it)
      logPrior += logPriorDict[it->first];
  }
    
  void set_noise_dist() {
    noiseDist = get_log_unigram_dist(vocab, indexedVocab);
    vector<double> multinomial(vocabSize, 0.0);
    mapUnsignedDouble::iterator it;
    for (it = noiseDist.begin(); it != noiseDist.end(); ++it)
      multinomial[it->first] = exp(it->second);
    sampler.initialise(multinomial);
  }
  
  double 
  prob_model_to_noise(unsigned tgtWord, RowVectorXf conVec, double biasSum) {
    double lpThetaTgt = wordVectors[tgtWord].dot(conVec) + biasSum;
    lpThetaTgt -= logPrior/corpusSize;
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
      cerr << "\nCalculating LLH...";
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
  
  void train_nce(vector<unsigned>& words, unsigned nCores, double rate,
                 mapLexParaP& pp) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      /* Get context vector statistics */
      RowVectorXf contextVec(vecLen);contextVec.setZero(vecLen);
      double contextBias = 0;
      vector<RowVectorXf> gradPriorCon;
      for (unsigned c=0; c<contextWords.size(); ++c) {
        unsigned conWord = contextWords[c];
        contextVec += wordVectors[conWord];
        contextBias += wordBiases[conWord];
        /* Get gradient of prior wrt context words in advance */
        RowVectorXf gPriorCon(vecLen);gPriorCon.setZero(vecLen);
        if (pp.find(conWord) != pp.end())
          for (unsigned i=0; i<pp[conWord].size(); ++i) {
            unsigned ppWord = pp[conWord][i];
            gPriorCon += 2/corpusSize * (wordVectors[conWord]-wordVectors[ppWord]);
          }
        gradPriorCon.push_back(gPriorCon);
      }
      /* Get the ratio of probab scores of theta and noise */
      double pNoiseToThetaTgt = 1-prob_model_to_noise(tgtWord, contextVec,
                                                      contextBias);
      /* Select noise words for this target word */
      unsigned noiseWords[numNoiseWords];
      for (unsigned selWrds=0; selWrds<numNoiseWords; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* Marginalization over noise words */
      double pThetaToNoiseSum=0;
      vector<double> pThetaToNoiseList;
      for (unsigned j=0; j<numNoiseWords; ++j) {
        double pThetaToNoise = prob_model_to_noise(noiseWords[j], contextVec,
                                                   contextBias);
        pThetaToNoiseSum += pThetaToNoise;
        pThetaToNoiseList.push_back(pThetaToNoise);
      }
      /* Calculate target word vector update */
      RowVectorXf gradTgt = contextVec;
      if (pp.find(tgtWord) != pp.end())
        for (unsigned i=0; i<pp[tgtWord].size(); ++i) {
          unsigned ppWord = pp[tgtWord][i];
          gradTgt -= 2/corpusSize * (wordVectors[tgtWord]-wordVectors[ppWord]);
        }
      /* Calculate context bias update */
      RowVectorXf delTgtVec = pNoiseToThetaTgt * gradTgt;
      double delConBias = pNoiseToThetaTgt - pThetaToNoiseSum;
      /* Calculate context vector updates */
      vector<RowVectorXf> delConVecs;
      for (unsigned c=0; c<contextWords.size(); ++c) {
        RowVectorXf delConVec = pNoiseToThetaTgt * wordVectors[tgtWord];
        delConVec -= pNoiseToThetaTgt * gradPriorCon[c];
        for (unsigned j=0; j<numNoiseWords; ++j)
          delConVec -= pThetaToNoiseList[j]*(wordVectors[noiseWords[j]]-gradPriorCon[c]);
        delConVecs.push_back(delConVec);
      }
      /* Apply the updates */
      update(tgtWord, contextWords, delTgtVec, delConVecs, delConBias, rate);
    }
    /* Change the prior only once per sentence to avoid computational cost */
    update_prior(words, pp);
  }
  
  void update_prior(vector<unsigned>& words, mapLexParaP& pp) {
    for (unsigned i=0; i<words.size(); ++i) {
      unsigned word = words[i];
      if (pp.find(word) != pp.end()) {
        /* Remove the old component of this word's prior */
        logPrior -= logPriorDict[word];
        /* Calculate the new component of this word's prior */
        double distance = 0;
        for (unsigned i=0; i<pp[word].size(); ++i) {
          unsigned ppWord = pp[word][i];
          RowVectorXf delVec = wordVectors[word]-wordVectors[ppWord];
          distance += delVec.squaredNorm();
        }
        /* Update the prior component dict */
        logPriorDict[word] = -distance;
        /* Update the overall prior */
        logPrior += logPriorDict[word];
      }
    }
  }
  
  void update(unsigned tgtWord, vector<unsigned>& contextWords, 
              RowVectorXf delTgtVec, vector<RowVectorXf>& delConVecs, 
              double delConBias, double rate) {
    /* Update the target word vector */
    RowVectorXf delTgtVecSq = delTgtVec.array().square();
    adagradVecMem[tgtWord] += delTgtVecSq;
    RowVectorXf temp = adagradVecMem[tgtWord].array().sqrt();
    wordVectors[tgtWord] += rate * delTgtVec.cwiseQuotient(temp); 
    /* Update the context word vectors */
    double delConBiasSq = delConBias * delConBias;
    for (unsigned k=0; k<contextWords.size(); ++k) {
      unsigned conWord = contextWords[k];
      RowVectorXf delConVecSq = delConVecs[k].array().square();
      /* Update the adagrad memory first */
      adagradVecMem[conWord] += delConVecSq;
      adagradBiasMem[conWord] += delConBiasSq;
      /* Now apply the updates */
      RowVectorXf temp = adagradVecMem[conWord].array().sqrt();
      wordVectors[conWord] += rate*delConVecs[k].cwiseQuotient(temp);
      wordBiases[conWord] += rate*delConBias/sqrt(adagradBiasMem[conWord]);
    }
  }

  void 
  train_on_corpus(string corpus, unsigned iter, unsigned nCores, double lRate,
                  string ppCorpus) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    mapLexParaP paraPhrases = read_lex_parap(ppCorpus, indexedVocab);
    compute_prior(paraPhrases);
    time_t start, end;
    time(&start);
    cerr << "\nCorpus size: " << corpusSize;
    cerr << "\nLog posterior: " << log_lh(corpus, nCores);
    cerr << "\nLog prior: " << logPrior;
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
          train_nce(words, nCores, rate, paraPhrases);
          numWords += words.size();
          cerr << int(numWords/1000) << "K\r";
        }
        inputFile.close();
        time(&end);
        cerr << "Time taken: " << float(difftime(end,start)/3600) << " hrs\n";
        time(&start);
        cerr << "\nLog posterior: " << log_lh(corpus, nCores);
        cerr << "\nLog prior: " << logPrior;
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
  string corpus = "../1k";
  string ppCorpus = "corpora/clean-1.0-s-lexical.txt";
  unsigned window = 5, freqCutoff = 2, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 5, numCores = 5;
  double rate = 0.05;
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate, ppCorpus);
  print_vectors("x.txt", obj.wordVectors, obj.indexedVocab);
  return 1;
}