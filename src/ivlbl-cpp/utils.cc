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

typedef std::tr1::unordered_map<string, int> mapStrInt;
typedef std::tr1::unordered_map<int, float> mapIntFloat;
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

pair<mapStrInt, mapStrStr> get_vocab(string filename) {
    
    string line, normWord;
    vector<string> words;
    mapStrInt vocab;
    mapStrStr word2norm;
    
    ifstream myfile(filename.c_str());
    
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
void filter_vocab(mapStrInt& vocab, const int freqCutoff) {
    
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end();)
        if (it->second < freqCutoff)
            vocab.erase(it++);
        else
            it++;
    
    return;
}

mapStrInt reindex_vocab(mapStrInt& vocab) {
    
    int index = 0;
    mapStrInt indexedVocab;
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        indexedVocab[it->first] = index++;
    
    return indexedVocab;
}

mapIntFloat get_unigram_dist(mapStrInt& vocab, mapStrInt& indexedVocab) {

    float sumFreq = 0;
    mapIntFloat unigramDist;
    
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        sumFreq += it->second;
    
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        unigramDist[indexedVocab[it->first]] = it->second/sumFreq;
    
    return unigramDist;

}

void print_map(mapStrInt& vocab) {
    
    for (mapStrInt::iterator it = vocab.begin(); it != vocab.end(); ++it)
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void print_map(mapIntFloat& vocab) {
    
    for (mapIntFloat::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void print_map(mapStrStr& vocab) {
    
    for (mapStrStr::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

vector<RowVectorXf> epsilon_vector(int row, int col) {

    vector<RowVectorXf> epsilonVec;
    RowVectorXf vec(col);
    vec.setOnes(col);
    vec *= EPSILON;
    for (int i=0; i<row; i++)
        epsilonVec.push_back(vec);
    
    return epsilonVec;
}

RowVectorXf epsilon_vector(int row) {

    RowVectorXf nonZeroVec(row);
    nonZeroVec.setOnes(row);
    nonZeroVec *= EPSILON;
    return nonZeroVec;
}

RowVectorXf random_vector(const int length) {

    RowVectorXf randVec(length);
    for (int i=0; i<randVec.size(); i++)
        randVec[i] = (rand()/(double)RAND_MAX);
    randVec /= randVec.norm();
    
    return randVec;
}

vector<RowVectorXf> random_vector(int row, int col) {

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
                   mapStrInt& indexedVocab) {

    ofstream outFile(fileName);
    for (mapStrInt::iterator it=indexedVocab.begin(); it!= indexedVocab.end(); it++){
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

vector<int> words_in_window(vector<int>& words, int wordIndex, 
                            int windowSize) {
    
    vector<int> wordsInWindow;
    int sentLen = words.size();
    
    if (wordIndex < windowSize) {
        if (wordIndex + windowSize + 1 < sentLen)
            wordsInWindow.insert(wordsInWindow.begin(), 
                                 words.begin()+wordIndex+1, 
                                 words.begin()+wordIndex+1+windowSize);
        else
            wordsInWindow.insert(wordsInWindow.begin(), 
                                 words.begin()+wordIndex+1, words.end());
                                 
        wordsInWindow.insert(wordsInWindow.begin(), words.begin(), 
                             words.begin()+wordIndex);
    } else {
        if (wordIndex + windowSize + 1 < sentLen)
            wordsInWindow.insert(wordsInWindow.begin(), 
                                 words.begin()+wordIndex+1, 
                                 words.begin()+wordIndex+1+windowSize);
        else
            wordsInWindow.insert(wordsInWindow.begin(), 
                                 words.begin()+wordIndex+1, words.end());
                                 
        wordsInWindow.insert(wordsInWindow.begin(), 
                             words.begin()+wordIndex-windowSize, 
                             words.begin()+wordIndex);
    }
    
    return wordsInWindow;
}
