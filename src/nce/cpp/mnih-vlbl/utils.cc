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
  mapStrUnsigned::iterator it;
  for (it=indexedVocab.begin(); it!= indexedVocab.end(); it++) {
    outFile << it->first << " ";
    for (unsigned i=0; i != wordVectors[it->second].size(); ++i)
      outFile << wordVectors[it->second][i] << " ";
    outFile << "\n";
  }
}