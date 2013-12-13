#include <iostream>
#include <vector>
#include "utils.h"
#include "train.h"

int main(){
    
    pair<vector<vector<float> >, mapStrUint> p;
    p = train_on_corpus("../1", 5, 0.5, 5, 5, 1, 80);
    print_vectors("out.txt", p.first, p.second);
    
    //print_map(p.first);
    print_map(p.second);
        
    /*unsigned int wordIndex = 7, windowSize = 5;
    vector<float> words(10, 5), b;
    
    for(int i=0; i<words.size();i ++)
        words[i] = i+1;
      
    vec_div_equal(words, words);  
    //b = get_noise_words(words, 20, 100);
    for(int i=0; i<words.size(); i++)
        cout << words[i] << " ";
    cout << "\n";
    
    //cout << accumulate(b.begin(), b.end(), 0);*/
    
    return 1;
}