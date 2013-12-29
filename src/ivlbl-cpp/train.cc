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
#include <random>
#include "alias_sampler.h"

using namespace std;
using namespace Eigen;

#define EPSILON 0.00000000000000000001;
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<string, string> mapStrStr;
typedef std::tr1::unordered_map<unsigned, double> mapUnsignedDouble;
typedef std::tr1::unordered_map<int, double> mapIntDouble;

mapIntDouble SIGMOID;

/* =================== Utility functions begin =================== */

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

pair<mapStrUnsigned, mapStrStr> get_vocab(string filename) {
  string line, normWord;
  vector<string> words;
  mapStrUnsigned vocab;
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
mapStrUnsigned filter_vocab(mapStrUnsigned& vocab, const unsigned freqCutoff) {
  mapStrUnsigned filtVocab;
  for (mapStrUnsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
    if (! (it->second < freqCutoff) )
      filtVocab[it->first] = it->second;
  return filtVocab;
}

mapStrUnsigned reindex_vocab(mapStrUnsigned& vocab) {
  unsigned index = 0;
  mapStrUnsigned indexedVocab;
  for (mapStrUnsigned::iterator it = vocab.begin(); it != vocab.end(); ++it){
    string word = it->first;
    indexedVocab[word] = index++;
  }
  return indexedVocab;
}

mapUnsignedDouble get_log_unigram_dist(mapStrUnsigned& vocab, mapStrUnsigned& indexedVocab) {
  double sumFreq = 0;
  mapUnsignedDouble unigramDist;
  for (mapStrUnsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
    sumFreq += it->second;
  for (mapStrUnsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
    unigramDist[indexedVocab[it->first]] = log(it->second/sumFreq);
  return unigramDist;
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
  for (unsigned i=0; i<row; ++i)
    randVec.push_back(random_vector(col));
  return randVec;
}

void print_vectors(char* fileName, vector<RowVectorXf>& wordVectors, 
                   mapStrUnsigned& indexedVocab) {
  ofstream outFile(fileName);
  for (mapStrUnsigned::iterator it=indexedVocab.begin(); it!= indexedVocab.end(); it++){
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

/* =================== Utility functions end =================== */
double diff_score_word_noise(unsigned word, vector<unsigned>& contextWords,
                             mapUnsignedDouble& noiseDist,
                             RowVectorXf& wordBiases,
                             vector<RowVectorXf>& wordVectors,
                             double logNumNoiseWords) {                     
  double sumScore = 0;
  for (unsigned i=0; i<contextWords.size(); ++i)
    sumScore += wordVectors[word].dot(wordVectors[contextWords[i]]) + wordBiases[contextWords[i]];
  return sumScore - logNumNoiseWords - noiseDist[word];
}

/* Main class definition that learns the word vectors */

class WordVectorLearner {
  mapStrUnsigned vocab;
  mapUnsignedDouble noiseDist;
  mapStrStr word2norm;
  unsigned vocabSize, windowSize, numNoiseWords, freqCutoff, vecLen;
  double learningRate, logNumNoiseWords;
  string corpusName;
  vector<RowVectorXf> adagradVecMem;
  RowVectorXf wordBiases, adagradBiasMem;
  /* For sampling noise words */
  AliasSampler sampler;
    
public:
      
  vector<RowVectorXf> wordVectors;
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
    vector<double> multinomial(vocabSize, 0.0);
    for (mapUnsignedDouble::iterator it = noiseDist.begin(); it != noiseDist.end(); ++it)
      multinomial[it->first] = exp(it->second);
    sampler.initialise(multinomial);
  }
    
  void train_word_vec(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel for num_threads(nCores) private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      /* Get the words in the window context of the target word */
      vector<unsigned> contextWords;
      unsigned start, end, sentLen = words.size(), tgtWord=words[tgtWrdIx];
      start = (tgtWrdIx <= windowSize)? 0: tgtWrdIx-windowSize;
      end = (sentLen-tgtWrdIx <= windowSize)? sentLen-1: tgtWrdIx+windowSize;
      for (unsigned i=start; i<=end; ++i)
        if (i != tgtWrdIx)
          contextWords.push_back(words[i]);
      /* Get the diff of score of the word in context and the noise dist */
      double x = diff_score_word_noise(tgtWord, contextWords,
                                       noiseDist, wordBiases, wordVectors,
                                       logNumNoiseWords);
      double wordContextScore = (x>MAX_EXP)? 1: (x<-MAX_EXP? 0: SIGMOID[(int)((x+MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))]);
      /* Select noise words for this target word */
      unsigned noiseWords[numNoiseWords];
      for (unsigned selWrds=0; selWrds<numNoiseWords; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* Get the diff of score of noise words in context and the noise dist */
      RowVectorXf noiseScoreGradProd(wordVectors[0].size());
      noiseScoreGradProd.setZero(noiseScoreGradProd.size());
      double noiseScoreSum=0;
      for (unsigned j=0; j<numNoiseWords; ++j) {
        double y = diff_score_word_noise(noiseWords[j], contextWords,
                                         noiseDist, wordBiases, wordVectors,
                                         logNumNoiseWords);
        double noiseScore = (y>MAX_EXP)? 1: (y<-MAX_EXP? 0: SIGMOID[(int)((y+MAX_EXP)*(EXP_TABLE_SIZE/MAX_EXP/2))]);
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
      cerr << "\n";
      }
      else {
        cout << "\nUnable to open file\n";
        break;
      }
    }
  }
};

int main(int argc, char **argv){
  /* pre-computing the logistic func values */
  for (unsigned i = 0; i < EXP_TABLE_SIZE; i++) {
    SIGMOID[i] = exp((i / (double)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
    SIGMOID[i] = SIGMOID[i] / (SIGMOID[i] + 1);
  }
      
  string corpus = "../10k";
  unsigned window = 5, freqCutoff = 2, noiseWords = 10, vectorLen = 80;
  unsigned numIter = 1, numCores = 6;
  double rate = 0.05;
  
  WordVectorLearner obj (window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate);
  print_vectors("y.txt", obj.wordVectors, obj.indexedVocab);
  return 1;
}