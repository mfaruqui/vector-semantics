#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <tr1/unordered_map>
#include <map>
#include <cmath>

using namespace std;
typedef std::tr1::unordered_map<string, unsigned int> mapStrUint;
typedef std::tr1::unordered_map<string, string> mapStrStr;
typedef std::tr1::unordered_map<unsigned int, float> mapUintFloat;

float EPSILON = 0.00000000000000000001;

void lower_string(string& word){
    
    transform(word.begin(), word.end(), word.begin(), ::tolower);
    return;
}

string normalize_word(string& word){
    
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

pair<mapStrUint, mapStrStr> get_vocab(char* filename){
    
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

void filter_vocab(mapStrUint& vocab, const int freqCutoff){
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        if (it->second < freqCutoff)
            vocab.erase(it->first);
    
    return;
}

mapStrUint reindex_vocab(mapStrUint& vocab){
    
    unsigned int index = 0;
    mapStrUint indexedVocab;
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        indexedVocab[it->first] = index++;
    
    return indexedVocab;
}

mapUintFloat get_unigram_dist(mapStrUint& vocab, mapStrUint& indexedVocab){

    float sumFreq = 0;
    mapUintFloat unigramDist;
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        sumFreq += it->second;
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it)
        unigramDist[indexedVocab[it->first]] = it->second/sumFreq;
    
    return unigramDist;

}

void print_map(mapStrUint& vocab){
    
    for (mapStrUint::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void print_map(mapUintFloat& vocab){
    
    for (mapUintFloat::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void print_map(mapStrStr& vocab){
    
    for (mapStrStr::iterator it = vocab.begin(); it != vocab.end(); ++it) 
        cout << it->first << " " << it->second << "\n";
    
    return;
}

void normalize_vector(vector<float>& vec){

    float magnitude = 0;
    for (vector<float>::iterator it = vec.begin(); it != vec.end(); ++it)
        magnitude += *it * *it;
    magnitude = sqrt(magnitude);
    
    if (magnitude == 0) magnitude = 0.000001;
    std::transform(vec.begin(), vec.end(), vec.begin(), std::bind1st(std::multiplies<float>(),1/magnitude));
    
    return;
}

vector<vector<float> > epsilon_vector(unsigned int row, unsigned int col){

    vector<vector<float> > epsilonVec;
    vector<float> vec(col, EPSILON);
    for (int i=0; i<row; i++)
        epsilonVec.push_back(vec);
    
    return epsilonVec;
}

vector<float> epsilon_vector(unsigned int row){

    vector<float> nonZeroVec(row, EPSILON);
    return nonZeroVec;
}

vector<float> random_vector(const unsigned int length){

    vector<float> randVec(length, 0.0);
    for (int i=0; i<randVec.size(); i++)
        randVec.at(i) = (rand()/(double)RAND_MAX);
    normalize_vector(randVec);
    
    return randVec;
}

vector<vector<float> > random_vector(unsigned int row, unsigned int col){

    vector<vector<float> > randVec;
    vector<float> vec(col, 0.0);
    for (int i=0; i<row; i++){
        for (int j=0; j<col; j++)
            vec.at(j) = (rand()/(double)RAND_MAX);
        normalize_vector(vec);
        randVec.push_back(vec);
    }
    
    return randVec;
}

void print_vectors(char* fileName, vector<vector<float> >& wordVectors, mapStrUint& indexedVocab){

    ofstream outFile(fileName);
    for (mapStrUint::iterator it=indexedVocab.begin(); it!= indexedVocab.end(); it++){
        outFile << it->first << " ";
        for (int i=0; i != wordVectors[it->second].size(); i++)
            outFile << wordVectors[it->second][i] << " ";
        outFile << "\n";
    }
    
    return;
}