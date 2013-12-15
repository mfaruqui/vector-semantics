#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <iostream>
#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

typedef std::tr1::unordered_map<string, unsigned int> mapStrUint;
typedef std::tr1::unordered_map<unsigned int, float> mapUintFloat;
typedef std::tr1::unordered_map<string, string> mapStrStr;

void lower_string(string& word);
string normalize_word(string& word);
vector<string> &split(string &s, char delim, vector<string> &elems);
vector<string> split_line(string& line, char delim);

pair<mapStrUint, mapStrStr> get_vocab(char* filename);
void filter_vocab(mapStrUint& vocab, const int freqCutoff);
mapStrUint reindex_vocab(mapStrUint& vocab);
mapUintFloat get_unigram_dist(mapStrUint& vocab, mapStrUint& indexedVocab);

void print_map(mapStrUint& vocab);
void print_map(mapUintFloat& vocab);
void print_map(mapStrStr& vocab);

RowVectorXf random_vector(const unsigned int length);
vector<RowVectorXf> random_vector(unsigned int row, unsigned int col);
vector<RowVectorXf> epsilon_vector(unsigned int row, unsigned int col);
RowVectorXf epsilon_vector(unsigned int row);

void print_vectors(char* fileName, vector<RowVectorXf>& wordVectors, mapStrUint& indexedVocab);

#endif