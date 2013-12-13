#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <iostream>
#include <vector>
#include <string>

using namespace std;
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

void normalize_vector(vector<float>& vec);
vector<float> random_vector(const unsigned int length);
vector<vector<float> > random_vector(unsigned int row, unsigned int col);
vector<vector<float> > epsilon_vector(unsigned int row, unsigned int col);
vector<float> epsilon_vector(unsigned int row);

void print_vectors(char* fileName, vector<vector<float> >& wordVectors, mapStrUint& indexedVocab);

#endif