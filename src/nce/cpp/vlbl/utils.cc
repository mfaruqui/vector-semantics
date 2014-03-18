#include "utils.h"

using namespace std;
using namespace Eigen;

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

/* Reads lexical paraphrases from the input file in the following format:
  fromWord toWord1 toWord2 toWord3 */
mapLexParaP read_lex_parap(string filename, mapStrUnsigned& indexedVocab) {
  string line, normWord, fromWord;
  vector<string> words;
  mapLexParaP lexParaP;
  unsigned numParaP=0;
  ifstream myfile(filename.c_str());
  if (myfile.is_open()) {
    while(getline(myfile, line)) {
      words = split_line(line, ' ');
      fromWord = normalize_word(words[0]);
      /* If fromWord present in vocab */
      if (indexedVocab.find(fromWord) != indexedVocab.end()) {
        vector<unsigned> toWordsInVocab;
        /* process the toWords which are second word onwards */
        for (unsigned i=1; i<words.size(); ++i) {
          string toWordNorm = normalize_word(words[i]);
          /* see if the toWord is present in vocab */
          if (indexedVocab.find(toWordNorm) != indexedVocab.end()) {
            toWordsInVocab.push_back(indexedVocab[toWordNorm]);
            numParaP += 1;
          }
        }
        lexParaP[indexedVocab[fromWord]] = toWordsInVocab;
        toWordsInVocab.erase(toWordsInVocab.begin(), toWordsInVocab.end());
      }
    }
    myfile.close();
    cerr << "\n" << "Read the paraphrase file" << "\n";
    cerr << "Words with paraphrases: " << lexParaP.size() << "\n";
    if (lexParaP.size() > 0)
      cerr << "Avg lex pPhrases/word: " << numParaP/lexParaP.size() << "\n";
  }
  else
    cerr << "\n" << "Unable to open paraphrase corpus" << "\n";
  return lexParaP;
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

vector<string> filter_vocab(mapStrUnsigned& vocab, const unsigned freqCutoff) {
  vector<string> filtVocab;
  for (mapStrUnsigned::iterator it = vocab.begin(); it != vocab.end(); ++it)
    if (! (it->second < freqCutoff) )
      filtVocab.push_back(it->first);
  return filtVocab;
}

mapStrUnsigned reindex_vocab(vector<string> vocabList) {
  mapStrUnsigned indexedVocab;
  for (unsigned i = 0; i < vocabList.size(); ++i) {
    indexedVocab[vocabList[i]] = i;
  }
  return indexedVocab;
}

mapUnsignedDouble get_log_unigram_dist(mapStrUnsigned& vocab, mapStrUnsigned& indexedVocab) {
  double sumFreq = 0;
  mapUnsignedDouble unigramDist;
  mapStrUnsigned::iterator it;
  for (it = indexedVocab.begin(); it != indexedVocab.end(); ++it)
    sumFreq += vocab[it->first];
  for (it = indexedVocab.begin(); it != indexedVocab.end(); ++it)
    unigramDist[it->second] = log(vocab[it->first]/sumFreq);
  return unigramDist;
}

double get_corpus_size(mapStrUnsigned& vocab, mapStrUnsigned& indexedVocab) {
  double sumFreq = 0;
  for (auto it = indexedVocab.begin(); it != indexedVocab.end(); ++it)
    sumFreq += vocab[it->first];
  return sumFreq;
}

mapUnsignedDouble get_unigram_dist(mapStrUnsigned& vocab, mapStrUnsigned& indexedVocab) {
  double sumFreq = 0;
  mapUnsignedDouble unigramDist;
  mapStrUnsigned::iterator it;
  for (it = indexedVocab.begin(); it != indexedVocab.end(); ++it)
    sumFreq += vocab[it->first];
  for (it = indexedVocab.begin(); it != indexedVocab.end(); ++it)
    unigramDist[it->second] = vocab[it->first]/sumFreq;
  return unigramDist;
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
  mapStrUnsigned::iterator it;
  for (it=indexedVocab.begin(); it!= indexedVocab.end(); it++) {
    outFile << it->first << " ";
    for (unsigned i=0; i != wordVectors[it->second].size(); ++i)
      outFile << wordVectors[it->second][i] << " ";
    outFile << "\n";
  }
  outFile.close();
}

void print_biases(char* fileName, RowVectorXf wordBiases,
                   mapStrUnsigned& indexedVocab) {
  ofstream outFile(fileName);
  mapStrUnsigned::iterator it;
  for (it=indexedVocab.begin(); it!= indexedVocab.end(); it++)
    outFile << it->first << " " << wordBiases[it->second] << "\n";
  outFile.close();
}