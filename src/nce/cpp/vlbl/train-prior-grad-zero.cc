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
  mapUnsignedDouble noiseDist;
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
    adagradVecMem = random_vector(vocabSize, vecLen);
    adagradBiasMem = random_vector(vocabSize);
  }
  
  /* Compute total prior */
  double compute_and_set_prior(mapLexParaP& pp, double priorWt) {
    logPrior = 0;
    for (auto it = pp.begin(); it != pp.end(); ++it) 
      for (unsigned i=0; i<pp[it->first].size(); ++i) {
        unsigned ppWord = pp[it->first][i];
        RowVectorXf delVec = wordVectors[it->first]-wordVectors[ppWord];
        logPrior -= delVec.squaredNorm();
      }
    logPrior *= priorWt;
    return logPrior;
  }
  
  void adagrad_vec(unsigned word, RowVectorXf delWordVec, double rate) {
    RowVectorXf delVecSq = delWordVec.array().square();
    adagradVecMem[word] += delVecSq;
    RowVectorXf temp = adagradVecMem[word].array().sqrt();
    wordVectors[word] += rate * delWordVec.cwiseQuotient(temp); 
  }
  
  void adagrad_bias(unsigned word, double delBias, double rate) {
    adagradBiasMem[word] += delBias * delBias;
    wordBiases[word] += rate*delBias/sqrt(adagradBiasMem[word]);
  }
  
  void update(vector<unsigned>& contextWords, RowVectorXf delConVec, 
              double delConBias, double rate) {
    /* Update the context word vectors */
    for (unsigned k=0; k<contextWords.size(); ++k) {
      adagrad_vec(contextWords[k], delConVec, rate);
      adagrad_bias(contextWords[k], delConBias, rate);
    }
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
    double lpNoiseTgt = logNumNoiseWords+noiseDist[tgtWord];
    double pNoiseToThetaTgt = 1/(1+exp(lpThetaTgt-lpNoiseTgt));
    assert (isfinite(pNoiseToThetaTgt));
    return 1-pNoiseToThetaTgt;
  }
  
  void 
  train_posterior_nce(vector<unsigned>& words, unsigned nCores, double rate) {
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
      double delConBias = pNoiseToThetaTgt - pThetaToNoiseSum;
      RowVectorXf delConVec = pNoiseToThetaTgt * wordVectors[tgtWord];
      delConVec -= pThetaToNoiseGradProd;
      /* Apply the updates */
      update(contextWords, delConVec, delConBias, rate);
    }
  }
  
  void train_prior(unsigned numWords, mapLexParaP& pp,
                   unsigned nCores, vector<unsigned>& ppSrcWords) {
    unsigned wordIndx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(wordIndx)            
    for (wordIndx=0; wordIndx < ppSrcWords.size(); ++wordIndx) {
      unsigned word = ppSrcWords[wordIndx];
      if (pp[word].size() < 1) continue;
      /* Weigh the original word vec by the num paraphrases */
      RowVectorXf newVec = pp[word].size()*wordVectors[word];
      for (unsigned i=0; i<pp[word].size(); ++i) {
        unsigned ppWord = pp[word][i];
        /* weigh all paraphrases by 1 */
        newVec += wordVectors[ppWord];
      }
      /* Divide the update of prior to sum to 1 in the end */
      newVec /= 2*pp[word].size();
      wordVectors[word] = newVec;
    }
  }

  void 
  train_on_corpus(string corpus, unsigned iter, unsigned nCores, double lRate,
                  string ppCorpus, unsigned updatePrior) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    mapLexParaP paraPhrases = read_lex_parap(ppCorpus, indexedVocab);
    vector<unsigned> ppWords;
    for (auto it=paraPhrases.begin(); it != paraPhrases.end(); ++it) 
      ppWords.push_back(it->first);
    time_t start, end;
    time(&start);
    cerr << "\nCorpus size: " << corpusSize;
    for (unsigned i=0; i<iter; ++i) {
      double rate = lRate/(i+1);
      cerr << "\n\nIteration: " << i+1;
      cerr << "\nLearning rate: " << rate << "\n";
      ifstream inputFile(corpus.c_str());
      string line, normWord;
      vector<unsigned> words;
      unsigned numWords = 0, changePrior = updatePrior;
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
          train_posterior_nce(words, nCores, rate);
          if (numWords > changePrior) {
            train_prior(updatePrior, paraPhrases, nCores, ppWords);
            changePrior += updatePrior;
          }
          numWords += words.size();
          cerr << int(numWords/1000) << "K\r";
        }
        inputFile.close();
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
  /* For PPDBD training these will mostly remain the same */
  unsigned window = 5, freqCutoff = 10, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 1;
  double rate = 0.05;
  
  if (argc != 6) {
    cerr << "Usage: "<< argv[0] << " corpusName " << "ppCorpusName ";
    cerr << "outVecFileName " << "whenToUpdatePrior " <<"numCores\n";
    exit(0);
  }
  
  string corpus = argv[1];
  string ppCorpus = argv[2];
  string outFile = argv[3];
  string outVecFile = outFile+"_vec.txt", outBiasFile = outFile+"_bias.txt";
  unsigned updatePrior = atoi(argv[4]), numCores = atoi(argv[5]);
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate, ppCorpus, updatePrior);
  print_vectors(outVecFile, obj.wordVectors, obj.indexedVocab);
  print_biases(outBiasFile, obj.wordBiases, obj.indexedVocab);
  return 1;
}
