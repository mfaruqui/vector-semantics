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
  unsigned vocabSize, windowSize, noiseSize, freqCutoff, vecLen;
  double learningRate;
  vector<RowVectorXf> gConVecMem, gTgtVecMem;
  RowVectorXf gTgtBiasMem;
  /* For sampling noise words */
  AliasSampler sampler;
  vector<vector<unsigned>> noiseMem;
    
public:
      
  vector<RowVectorXf> conVecs, tgtVecs;
  RowVectorXf tgtBiases;
  mapStrUnsigned indexedVocab;
      
  WordVectorLearner(unsigned window, unsigned freq, unsigned noiseWords, 
                    unsigned vectorLen) {
    windowSize = window;
    freqCutoff = freq;
    noiseSize = noiseWords;
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
    tgtVecs = random_vector(vocabSize, vecLen);
    conVecs = random_vector(vocabSize, vecLen);
    tgtBiases = random_vector(vocabSize);
    gTgtVecMem = epsilon_vector(vocabSize, vecLen);
    gConVecMem = epsilon_vector(vocabSize, vecLen);
    gTgtBiasMem = epsilon_vector(vocabSize);
  }
  
  void reset_adagrad() {
    gTgtVecMem = epsilon_vector(vocabSize, vecLen);
    gConVecMem = epsilon_vector(vocabSize, vecLen);
    gTgtBiasMem = epsilon_vector(vocabSize);
  }
    
  void set_noise_dist() {
    noiseDist = get_unigram_dist(vocab, indexedVocab);
    vector<double> multinomial(vocabSize, 0.0);
    mapUnsignedDouble::iterator it;
    for (it = noiseDist.begin(); it != noiseDist.end(); ++it)
      multinomial[it->first] = it->second;
    sampler.initialise(multinomial);
  }
  
  double score_word_context(unsigned word, RowVectorXf contxtVec) {          
    return tgtVecs[word].dot(contxtVec) + tgtBiases[word];
  }
  
  double score_vocab_context(RowVectorXf contxtVec, unsigned nCores) {
    unsigned i;
    double logScore = 0;
    #pragma omp parallel for private(i) num_threads(2*nCores) shared(logScore)
    for (i=0; i<filtVocabList.size(); ++i) {
      unsigned word = indexedVocab[filtVocabList[i]];
      double pairScore = tgtVecs[word].dot(contxtVec) + tgtBiases[word];
      #pragma omp critical
      {logScore = log_add(pairScore, logScore);}
    }
    return logScore;
  }
  
  double score_noise_context(RowVectorXf contxtVec, unsigned nCores, unsigned *noiseWords) {
    double score = 0;
    for (unsigned i=0; i<noiseSize; ++i) {
      unsigned word = noiseWords[i];
      // this can overflow, use logadd if required
      score += exp(tgtVecs[word].dot(contxtVec) + tgtBiases[word]);
    }
    return log(score);
  }
  
  double score_noise_context_is(RowVectorXf contxtVec, unsigned nCores, unsigned *noiseWords) {
    double score = 0;
    for (unsigned i=0; i<noiseSize; ++i) {
      unsigned word = noiseWords[i];
      // this can overflow, use logadd if required
      score += exp(tgtVecs[word].dot(contxtVec) + tgtBiases[word])/noiseDist[word];
    }
    return score;
  }
  
  double log_lh_noise(string corpus, unsigned nCores) {
    double lh = 0;
    ifstream inputFile(corpus.c_str());
    string line, normWord;
    vector<unsigned> words;
    int sentNum = 0, numWords = 0;
    if (inputFile.is_open()) {
      while (getline(inputFile, line)) {
        vector<string> tokens = split_line(line, ' ');
        words.clear();
        for (unsigned j=0; j<tokens.size(); ++j)
          if (word2norm.find(tokens[j]) != word2norm.end())
            words.push_back(indexedVocab[word2norm[tokens[j]]]);
        
        for (unsigned tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
          unsigned tgtWord = words[tgtWrdIx];
          vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
          RowVectorXf contxtVec(vecLen); contxtVec.setZero(vecLen);
          for (unsigned i=0; i<contextWords.size(); ++i)
            contxtVec += conVecs[contextWords[i]]/contextWords.size();
          double pWrdDoc = score_word_context(tgtWord, contxtVec);
          double scoreVocab = score_vocab_context(contxtVec, nCores);
          pWrdDoc = exp(pWrdDoc-scoreVocab);
          pWrdDoc /= (pWrdDoc+noiseSize*noiseDist[tgtWord]);
          lh += log(pWrdDoc);
          
          unsigned noiseWords[noiseSize];
          for (unsigned selWrds=0; selWrds<noiseSize; ++selWrds)
            noiseWords[selWrds] = sampler.Draw();
          
          for (unsigned j=0; j<noiseSize; ++j) {
            unsigned noiseWord = noiseWords[j];
            pWrdDoc = score_word_context(noiseWord, contxtVec);
            pWrdDoc = exp(pWrdDoc-scoreVocab);
            pWrdDoc /= (pWrdDoc+noiseSize*noiseDist[noiseWord]);
            lh += log(1-pWrdDoc);
          }
        }
        numWords += words.size();
        sentNum++;
      }
    inputFile.close();
    }
    else
      cout << "\nUnable to open file\n";
  return lh;
  }
  
  double log_lh(string corpus, unsigned nCores) {
    double lh = 0;
    ifstream inputFile(corpus.c_str());
    string line, normWord;
    vector<unsigned> words;
    if (inputFile.is_open()) {
      while (getline(inputFile, line)) {
        vector<string> tokens = split_line(line, ' ');
        words.clear();
        for (unsigned j=0; j<tokens.size(); ++j)
          if (word2norm.find(tokens[j]) != word2norm.end())
            words.push_back(indexedVocab[word2norm[tokens[j]]]);
        
        for (unsigned tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
          unsigned tgtWord = words[tgtWrdIx];
          vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
          RowVectorXf contxtVec(vecLen); contxtVec.setZero(vecLen);
          for (unsigned i=0; i<contextWords.size(); ++i)
            contxtVec += conVecs[contextWords[i]]/contextWords.size();
          double wordContxtScore = score_word_context(tgtWord, contxtVec);
          double vocabContxtScore = score_vocab_context(contxtVec, nCores);
          lh += wordContxtScore - vocabContxtScore;
        }
      }
    inputFile.close();
    }
    else
      cout << "\nUnable to open file\n";
  return lh;
  }
  
  void update(unsigned tgtWord, vector<unsigned> contextWords,
              RowVectorXf delTgtVec, RowVectorXf delContxtVec, 
              double delTgtBias, double rate) {
    /* Update the target adagrad memory first */
    RowVectorXf delTgtVecSquare = delTgtVec.array().square();
    gTgtVecMem[tgtWord] += delTgtVecSquare;
    gTgtBiasMem[tgtWord] += delTgtBias * delTgtBias;
    /* Update target word vector and bias */
    RowVectorXf temp = gTgtVecMem[tgtWord].array().sqrt();
    tgtVecs[tgtWord] += rate * delTgtVec.cwiseQuotient(temp);
    tgtBiases[tgtWord] += rate * delTgtBias / sqrt(gTgtBiasMem[tgtWord]);
    /* Update adagrad params and the context word vectors */
    RowVectorXf delContxtVecSquare = delContxtVec.array().square();
    for (unsigned k=0; k<contextWords.size(); ++k) {
      unsigned contxtWord = contextWords[k];
      gConVecMem[contxtWord] += delContxtVecSquare;
      RowVectorXf temp = gConVecMem[contxtWord].array().sqrt();
      conVecs[contxtWord] += rate * delContxtVec.cwiseQuotient(temp);
    }
  }
  
  void train_nce(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      /* Get the words in the window context of the target word */
      vector<unsigned> contxtWords = context(words, tgtWrdIx, windowSize);
      unsigned h = contxtWords.size();
      RowVectorXf contxtVec(vecLen); contxtVec.setZero(vecLen);
      for (unsigned i=0; i<contxtWords.size(); ++i)
        contxtVec += conVecs[contxtWords[i]]/h;
      /* Get prob that the word is sampled from the context dist */
      double pTgtWrdDoc = exp(score_word_context(tgtWord, contxtVec));
      pTgtWrdDoc /= (pTgtWrdDoc+noiseSize*noiseDist[tgtWord]);
      /* Select noise words for this target word */
      unsigned noiseWords[noiseSize];
      for (unsigned selWrds=0; selWrds<noiseSize; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* Marginalizing over the noise words */
      RowVectorXf delContxtVec = (1-pTgtWrdDoc) * tgtVecs[tgtWord]/h;
      double pWrdDocSum = 0;
      for (unsigned j=0; j<noiseSize; ++j) {
        unsigned noiseWord = noiseWords[j];
        double pWrdDoc = exp(score_word_context(noiseWord, contxtVec));
        pWrdDoc /= (pWrdDoc+noiseSize*noiseDist[noiseWord]);
        pWrdDocSum += pWrdDoc;
        delContxtVec -= pWrdDoc * tgtVecs[noiseWord]/h;
      }
      /* Calculate the update in the target word vector and bias */
      double delTgtBias = 1 - pTgtWrdDoc - pWrdDocSum;
      RowVectorXf delTgtVec = delTgtBias * contxtVec;
      /* Now apply updates */
      update(tgtWord, contxtWords, delTgtVec, delContxtVec, delTgtBias, rate);
    }
  }
  
  /* importance sampling */
  void train_is(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      unsigned h = contextWords.size();
      RowVectorXf contxtVec(vecLen); contxtVec.setZero(vecLen);
      for (unsigned i=0; i<contextWords.size(); ++i)
        contxtVec += conVecs[contextWords[i]]/h;
      /* Select noise words for this target word */
      unsigned noiseWords[noiseSize];
      for (unsigned selWrds=0; selWrds<noiseSize; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* V and v is defined in eq. 5 of Mnih and Teh 2012 */
      double V=0, v[noiseSize];
      for (unsigned i=0; i<noiseSize; i++) {
        unsigned word = noiseWords[i];
        v[i] = exp(score_word_context(word, contxtVec))/noiseDist[word];
        V += v[i];
      }
      /* Important sampling pre-calcualtions */
      RowVectorXf prodVecProb(vecLen); prodVecProb.setZero(vecLen);
      double noiseProbSum = 0;
      /* Marginalize over the noise words, not the whole vocab */
      for (unsigned i=0; i<noiseSize; ++i) {
        unsigned word = noiseWords[i];
        double probWordContxt = v[i]/V;
        prodVecProb += probWordContxt * tgtVecs[word]/h;
        noiseProbSum += probWordContxt;
      }
      /* Calculate update in target word vector and bias */
      RowVectorXf delTgtVec = (1 - noiseProbSum) * contxtVec;
      double delTgtBias = 1 - noiseProbSum;
      /* Calculate the change in context word vectors */
      RowVectorXf delContxtVec = tgtVecs[tgtWord]/h - prodVecProb;
      /* Now apply updates */
      update(tgtWord, contextWords, delTgtVec, delContxtVec, delTgtBias, rate);
    }
  }
  
  void train_mle_neg(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      unsigned h = contextWords.size();
      RowVectorXf contxtVec(vecLen); contxtVec.setZero(vecLen);
      for (unsigned i=0; i<contextWords.size(); ++i)
        contxtVec += conVecs[contextWords[i]]/h;
      /* Select noise words for this target word */
      unsigned noiseWords[noiseSize];
      for (unsigned selWrds=0; selWrds<noiseSize; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* MLE pre-calcualtions */
      RowVectorXf prodVecProb(vecLen); prodVecProb.setZero(vecLen);
      double vocabProbSum = 0;
      double scoreVocabContxt = score_noise_context(contxtVec, nCores, noiseWords);
      /* Marginalize over the noise words, not the whole vocab */
      for (unsigned i=0; i<noiseSize; ++i) {
        unsigned word = noiseWords[i];
        double scoreWordContxt = score_word_context(word, contxtVec);
        double probWordContxt = exp(scoreWordContxt-scoreVocabContxt);
        prodVecProb += probWordContxt * tgtVecs[word]/h;
        vocabProbSum += probWordContxt;
      }
      /* Calculate update in target word vector and bias */
      RowVectorXf delTgtVec = (1 - vocabProbSum) * contxtVec;
      double delTgtBias = 1 - vocabProbSum;
      /* Calculate the change in context word vectors */
      RowVectorXf delContxtVec = tgtVecs[tgtWord]/h - prodVecProb;
      /* Now apply updates */
      update(tgtWord, contextWords, delTgtVec, delContxtVec, delTgtBias, rate);
    }
  }
  
  void train_mle(vector<unsigned>& words, unsigned nCores, double rate) {
    unsigned tgtWrdIx;
    #pragma omp parallel num_threads(nCores)
    #pragma omp for nowait private(tgtWrdIx)
    for (tgtWrdIx=0; tgtWrdIx<words.size(); ++tgtWrdIx) {
      unsigned tgtWord = words[tgtWrdIx];
      vector<unsigned> contextWords = context(words, tgtWrdIx, windowSize);
      unsigned h = contextWords.size();
      RowVectorXf contxtVec(vecLen); contxtVec.setZero(vecLen);
      for (unsigned i=0; i<contextWords.size(); ++i)
        contxtVec += conVecs[contextWords[i]]/h;
      /* Select noise words for this target word */
      unsigned noiseWords[noiseSize];
      for (unsigned selWrds=0; selWrds<noiseSize; ++selWrds)
        noiseWords[selWrds] = sampler.Draw();
      /* MLE pre-calcualtions */
      RowVectorXf prodVecProb(vecLen); prodVecProb.setZero(vecLen);
      double vocabProbSum = 0;
      double scoreVocabContxt = score_vocab_context(contxtVec, nCores);
      /* Marginalize over the noise words, not the whole vocab */
      for (unsigned i=0; i<noiseSize; ++i) {
        unsigned word = noiseWords[i];
        double scoreWordContxt = score_word_context(word, contxtVec);
        double probWordContxt = exp(scoreWordContxt-scoreVocabContxt);
        prodVecProb += probWordContxt * tgtVecs[word]/h;
        vocabProbSum += probWordContxt;
      }
      /* Calculate update in target word vector and bias */
      RowVectorXf delTgtVec = (1 - vocabProbSum) * contxtVec;
      double delTgtBias = 1 - vocabProbSum;
      /* Calculate the change in context word vectors */
      RowVectorXf delContxtVec = tgtVecs[tgtWord]/h - prodVecProb;
      /* Now apply updates */
      update(tgtWord, contextWords, delTgtVec, delContxtVec, delTgtBias, rate);
    }
  }
  
  void 
  train_on_corpus(string corpus, unsigned iter, unsigned nCores, double lRate) {
    preprocess_vocab(corpus);
    init_vectors(vocabSize, vecLen);
    set_noise_dist();
    time_t start, end;
    time(&start);
    /*cerr << "\nLLH: " << log_lh(corpus, nCores);
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
      reset_adagrad();
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
          train_is(words, nCores, rate);
          numWords += words.size();
          cerr << int(numWords/1000) << "K\r";
        }
        inputFile.close();
        time(&end);
        cerr << "Time taken: " << float(difftime(end,start)/3600) << " hrs\n";
        /*time(&start);
        double llh = log_lh(corpus, nCores);
        cerr << "\nLLH: " << llh << " per word: " << llh/numWords;
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
  string corpus = "../news.2011.en.norm";
  unsigned window = 5, freqCutoff = 10, noiseWords = 25, vectorLen = 80;
  unsigned numIter = 1, numCores = 8;
  double rate = 0.05;
  
  WordVectorLearner obj(window, freqCutoff, noiseWords, vectorLen);
  obj.train_on_corpus(corpus, numIter, numCores, rate);
  print_vectors("is/news-con-r0.05-l80-n25.txt", obj.conVecs, obj.indexedVocab);
  //print_vectors("vlbl-news-con-corr.txt", obj.conVecs, obj.indexedVocab);
  return 1;
}