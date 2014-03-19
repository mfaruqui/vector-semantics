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

using namespace std;
using namespace Eigen;

typedef std::tr1::unordered_map<unsigned, RowVectorXf> mapUnsignedVec;

vector<unsigned> 
context(vector<unsigned>& words, unsigned tgtWrdIx, unsigned windowSize) {
  vector<unsigned> contextWords;
  unsigned start, end, sentLen = words.size();
  start = (tgtWrdIx <= windowSize)? 0: tgtWrdIx-windowSize;
  end = (sentLen-tgtWrdIx <= windowSize)? sentLen-1: tgtWrdIx+windowSize;
  for (unsigned i=start; i<=end; ++i)
    if (i != tgtWrdIx)
       contextWords.push_back(words[i]);
  return contextWords;
}

mapUnsignedVec read_vectors(string filename, mapStrUnsigned& indexedVocab) {
  string line, normWord;
  vector<string> wordAndVals;
  mapUnsignedVec wordVectors;
  ifstream myfile(filename.c_str());
  if (myfile.is_open()) {
    while(getline(myfile, line)) {
      wordAndVals = split_line(line, ' ');
      string word = wordAndVals[0];
      normWord = normalize_word(word);
      unsigned vecLen = wordAndVals.size()-1;
      RowVectorXf vector(vecLen);vector.setZero(vecLen);
      for (unsigned i=1; i<wordAndVals.size(); ++i)
        vector[i-1] = atof(wordAndVals[i].c_str());
      wordVectors[indexedVocab[normWord]] = vector;
    }
    myfile.close();
  }
  else
    cout << "Unable to open file";
  return wordVectors;
}

mapUnsignedDouble read_biases(string filename, mapStrUnsigned& indexedVocab) {
  string line, normWord;
  vector<string> wordAndVal;
  mapUnsignedDouble wordBiases;
  ifstream myfile(filename.c_str());
  if (myfile.is_open()) {
    while(getline(myfile, line)) {
      wordAndVal = split_line(line, ' ');
      string word = wordAndVal[0];
      normWord = normalize_word(word);
      wordBiases[indexedVocab[normWord]] = atof(wordAndVal[1].c_str());
    }
    myfile.close();
  }
  else
    cout << "Unable to open file";
  return wordBiases;
}

/* Main class definition that learns the word vectors */

class WordVectorLearner {
  mapStrUnsigned vocab;
  vector<string> filtVocabList;
  mapStrStr word2norm;
  unsigned vocabSize, windowSize, vecLen;
    
public:
      
  mapUnsignedVec wordVectors;
  mapUnsignedDouble wordBiases;
  mapStrUnsigned indexedVocab;
      
  WordVectorLearner(unsigned window) {
    windowSize = window;
  }
      
  void preprocess_vocab(string corpusName) {
    pair<mapStrUnsigned, mapStrStr> vocabPair = get_vocab(corpusName);
    vocab = vocabPair.first;
    word2norm = vocabPair.second;
    cerr << "\nOrig vocab " << vocab.size();
    filtVocabList = filter_vocab(vocab, 0);
    indexedVocab = reindex_vocab(filtVocabList);
    vocabSize = indexedVocab.size();
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
        for (unsigned j=0; j<tokens.size(); ++j) {
          unsigned wordId = indexedVocab[word2norm[tokens[j]]];
          if (wordVectors.find(wordId) != wordVectors.end())
            words.push_back(wordId);
          else
            words.push_back(indexedVocab["---unk---"]);
        }
        /* Calculate likelihood */
        unsigned tgtWrdIx;
        #pragma omp parallel num_threads(nCores) shared(lh)
        #pragma omp for nowait private(tgtWrdIx)
        for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
          unsigned tgtWord = words[tgtWrdIx];
          vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
          if (contextWords.size() < 1) continue;
          /* Get the sum of context vectors */
          unsigned vecLen = wordVectors[0].size();
          RowVectorXf contextVec(vecLen);contextVec.setZero(vecLen);
          double biasSum = 0;
          for (unsigned c=0; c<contextWords.size(); ++c) {
            contextVec += wordVectors[contextWords[c]];
            biasSum += wordBiases[contextWords[c]];
          }
          contextVec /= contextWords.size();
          biasSum /= contextWords.size();
          double contextScore = wordVectors[tgtWord].dot(contextVec) + biasSum;
          /* Get vocab context score */
          double vocabContextScore = 0;
          for (unsigned v=0; v<filtVocabList.size(); ++v) {
            unsigned vWord = indexedVocab[filtVocabList[v]];
            double scoreWordContext = wordVectors[vWord].dot(contextVec) + biasSum;
            vocabContextScore = log_add(scoreWordContext, vocabContextScore);
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

  void work(string corpus, string vecFile, string biasFile, unsigned nCores) {
    preprocess_vocab(corpus);
    wordVectors = read_vectors(vecFile, indexedVocab);
    wordBiases = read_biases(biasFile, indexedVocab);
    time_t start, end;
    double corpusSize = get_corpus_size(vocab, indexedVocab);
    cerr << "\nCorpus size: " << corpusSize;
    time(&start);
    double llh = log_lh(corpus, nCores);
    time(&end);
    cerr << "\nLog likelihood: " << llh;
    cerr << "\nLLH per word: " << llh/corpusSize;
    cerr << "\nTime taken: " << float(difftime(end,start)/3600) << " hrs";
  }
};

int main(int argc, char **argv){
  
  if (argc != 5) {
    cerr << "Usage: " << argv[0] << " corpus" << " vectorfile";
    cerr << " biasfile" << " numCores\n";
  }
  else {
    string corpus = argv[1];
    string vectorfile = argv[2];
    string biasfile = argv[3];
    unsigned numCores = atoi(argv[4]);
    unsigned window = 5;
    
    WordVectorLearner obj(window);
    obj.work(corpus, vectorfile, biasfile, numCores);
  }
  return 1;
}