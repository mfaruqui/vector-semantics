#include <iostream>
#include <vector>
#include <Eigen/Core>
#include "utils.h"
#include "train.h"

using namespace std;
using namespace Eigen;

int main(){
    
    pair<vector<RowVectorXf>, mapStrUint> p;
    p = train_on_corpus("../1k", 1, 0.05, 10, 5, 2, 80);
    print_vectors("out.txt", p.first, p.second);
    
    //print_map(p.second);
    
    return 1;
}