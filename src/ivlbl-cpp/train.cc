#include <iostream>
#include <vector>
#include <functional>
#include <numeric>
#include <cmath>
#include <string>
#include <tr1/unordered_map>
#include "utils.h"
#include "vecops.h"
#include "ivlbl.h"

int main(){
    
    /*mapStrUint vocab, indexedVocab;
    unsigned int freqCutoff = 5;
    mapUintFloat unigram;
    
    vocab = get_vocab("../100");
    filter_vocab(vocab, freqCutoff);
    cerr << "Vocab computed\n";
    
    indexedVocab = reindex_vocab(vocab);
    unigram = get_unigram_dist(vocab, indexedVocab);
    print_map(vocab);
    print_map(unigram);*/
    vector<float> a(5, 2), b;
    b = vec_sqrt(a);
    
    for (int i=0; i<b.size(); i++)
        cout << b[i] <<"\n";
    
    return 1;
}