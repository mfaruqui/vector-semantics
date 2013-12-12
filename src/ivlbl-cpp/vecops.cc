#include <vector>
#include <cmath>

using namespace std;

void vec_plus_equal(vector<float>& a, const vector<float>& b){
    
    for(int i=0; i<a.size(); i++)
        a[i] = a[i] + b[i];
    
    return;
}

vector<float> vec_plus(vector<float>& a, const vector<float>& b){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = a[i] + b[i];
    
    return result;
}

void vec_sub_equal(vector<float>& a, const vector<float>& b){
    
    for(int i=0; i<a.size(); i++)
        a[i] = a[i] - b[i];
    
    return;
}

vector<float> vec_sub(vector<float>& a, const vector<float>& b){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = a[i] - b[i];
    
    return result;
}

void vec_div_equal(vector<float>& a, const vector<float>& b){

    for(int i=0; i<a.size(); i++)
        a[i] = a[i] / b[i];
    
    return;
}

void vec_div_equal(vector<float>& a, const float b){

    for(int i=0; i<a.size(); i++)
        a[i] = a[i] / b;
    
    return;
}

vector<float> vec_div(vector<float>& a, const vector<float>& b){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = a[i] / b[i];
    
    return result;
}

vector<float> vec_div(vector<float>& a, const float b){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = a[i] / b;
    
    return result;
}

void vec_prod_equal(vector<float>& a, const vector<float>& b){

    for(int i=0; i<a.size(); i++)
        a[i] = a[i] * b[i];
    
    return;
}

void vec_prod_equal(vector<float>& a, const float b){

    for(int i=0; i<a.size(); i++)
        a[i] = a[i] * b;
    
    return;
}

vector<float> vec_prod(vector<float>& a, const vector<float>& b){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = a[i] * b[i];
    
    return result;
}

vector<float> vec_prod(vector<float>& a, const float b){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = a[i] * b;
    
    return result;
}

vector<float> vec_square(vector<float>& a){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = pow(a[i], 2);
    
    return result;
}

vector<float> vec_sqrt(vector<float>& a){
    
    vector<float> result(a.size());
    for(int i=0; i<a.size(); i++)
        result[i] = sqrt(a[i]);
    
    return result;
}