#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <iostream>
#include <vector>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

typedef std::tr1::unordered_map<string, int> mapStrInt;
typedef std::tr1::unordered_map<int, float> mapIntFloat;
typedef std::tr1::unordered_map<string, string> mapStrStr;

void lower_string(string& word);
string normalize_word(string& word);
vector<string> &split(string &s, char delim, vector<string> &elems);
vector<string> split_line(string& line, char delim);

pair<mapStrInt, mapStrStr> get_vocab(string filename);
void filter_vocab(mapStrInt& vocab, const int freqCutoff);
mapStrInt reindex_vocab(mapStrInt& vocab);
mapIntFloat get_unigram_dist(mapStrInt& vocab, mapStrInt& indexedVocab);

void print_map(mapStrInt& vocab);
void print_map(mapIntFloat& vocab);
void print_map(mapStrStr& vocab);

RowVectorXf random_vector(const int length);
vector<RowVectorXf> random_vector(int row, int col);
vector<RowVectorXf> epsilon_vector(int row, int col);
RowVectorXf epsilon_vector(int row);

void print_vectors(char* fileName, vector<RowVectorXf>& wordVectors, 
                   mapStrInt& indexedVocab);

vector<int> words_in_window(vector<int>& words, int wordIndex, int windowSize);

#endif