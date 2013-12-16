#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <tr1/unordered_map>
#include <map>
#include <cmath>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

typedef std::tr1::unordered_map<string, unsigned int> mapStrUint;
typedef std::tr1::unordered_map<unsigned int, float> mapUintFloat;
typedef std::tr1::unordered_map<string, string> mapStrStr;

float EPSILON = 0.00000000000000000001;

void lower_string(string& word ){
    
    transform(word.begin(), word.end(), word.begin(), ::tolower);
    return;
}

string normalize_word(string& word) {
    
    if (std::string::npos != word.find_first_of("0123456789"))
        return "---num---";
     
    for (int i=0; i<word.length(); i++)
        if (isalnum(word[i])){
            lower_string(word);
            return word;
        }
    
    return "---punc---";
    
}

vector<string> &split(string &s, char delim, vector<string>& elems) {
    
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty())
            elems.push_back(item);
    }
    
    return elems;
}

// Try splitting over all whitespaces not just space
vector<string> split_line(string& line, char delim) {
    
    vector<string> words;
    split(line, delim, words);
    
    return words;
}

pair<mapStrUint, mapStrStr> get_vocab(char* filename) {
    
    string line, normWord;
    vector<string> words;
    mapStrUint vocab;
    mapStrStr word2norm;
    
    ifstream myfile(filename);
    
    if (myfile.is_open()) {
        while(getline(myfile, line)) {
            words = split_line(line, ' ');
            for(int i=0; i<words.size(); i++){
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
while printing its still there ! x-(
http://stackoverflow.com/questions/17036428/c-map-element-doesnt-get-erased-if-i-refer-to-it
*/
void filter_vocab(mapStrUint& vocab, const int freqCutoff) {
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end();)
        if (it->second < freqCutoff)
            vocab.erase(it++);
        else
            it++;
    
    return;
}

mapStrUint reindex_vocab(mapStrUint& vocab) {
    
    unsigned int index = 0;
    mapStrUint indexedVocab;
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        indexedVocab[it->first] = index++;
    
    return indexedVocab;
}

mapUintFloat get_unigram_dist(mapStrUint& vocab, mapStrUint& indexedVocab) {

    float sumFreq = 0;
    mapUintFloat unigramDist;
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        sumFreq += it->second;
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        unigramDist[indexedVocab[it->first]] = it->second/sumFreq;
    
    return unigramDist;

}

void print_map(mapStrUint& vocab) {
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void print_map(mapUintFloat& vocab) {
    
    for (mapUintFloat::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void print_map(mapStrStr& vocab) {
    
    for (mapStrStr::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

vector<RowVectorXf> epsilon_vector(unsigned int row, unsigned int col) {

    vector<RowVectorXf> epsilonVec;
    RowVectorXf vec(col);
    vec.setOnes(col);
    vec *= EPSILON;
    for (int i=0; i<row; i++)
        epsilonVec.push_back(vec);
    
    return epsilonVec;
}

RowVectorXf epsilon_vector(unsigned int row) {

    RowVectorXf nonZeroVec(row);
    nonZeroVec.setOnes(row);
    nonZeroVec *= EPSILON;
    return nonZeroVec;
}

RowVectorXf random_vector(const unsigned int length) {

    RowVectorXf randVec(length);
    for (int i=0; i<randVec.size(); i++)
        randVec[i] = (rand()/(double)RAND_MAX);
    randVec /= randVec.norm();
    
    return randVec;
}

vector<RowVectorXf> random_vector(unsigned int row, unsigned int col) {

    vector<RowVectorXf> randVec;
    RowVectorXf vec(col);
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++)
            vec[j] = (rand()/(double)RAND_MAX);
        vec /= vec.norm();
        randVec.push_back(vec);
    }
    
    return randVec;
}

void print_vectors(char* fileName, vector<RowVectorXf>& wordVectors, 
                   mapStrUint& indexedVocab) {

    ofstream outFile(fileName);
    for (mapStrUint::iterator it=indexedVocab.begin(); it!= indexedVocab.end(); it++){
        /* This check is wrong but I have to put, coz of the elements not getting deleted :(
         By this we will bem issing the word at index 0. */
        if (it->second != 0){
            outFile << it->first << " ";
            for (int i=0; i != wordVectors[it->second].size(); i++)
                outFile << wordVectors[it->second][i] << " ";
            outFile << "\n";
        }
    }
    
    return;
}