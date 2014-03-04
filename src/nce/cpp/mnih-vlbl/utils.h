#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <cmath>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

#define EPSILON 0.00000000000000000001;

typedef std::tr1::unordered_map<string, unsigned> mapStrUnsigned;
typedef std::tr1::unordered_map<string, string> mapStrStr;
typedef std::tr1::unordered_map<unsigned, double> mapUnsignedDouble;

string normalize_word(string& word);

vector<string> split_line(string& line, char delim);

pair<mapStrUnsigned, mapStrStr> get_vocab(string filename);
vector<string> filter_vocab(mapStrUnsigned& vocab, const unsigned freqCutoff);
mapStrUnsigned reindex_vocab(vector<string> vocabList);

mapUnsignedDouble get_unigram_dist(mapStrUnsigned& vocab, mapStrUnsigned& indexedVocab);

RowVectorXf epsilon_vector(unsigned row);
vector<RowVectorXf> epsilon_vector(unsigned row, unsigned col);
RowVectorXf random_vector(const unsigned length);
vector<RowVectorXf> random_vector(unsigned row, unsigned col);

void print_vectors(char* fileName, vector<RowVectorXf>& wordVectors,
                   mapStrUnsigned& indexedVocab);