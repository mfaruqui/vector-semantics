#ifndef VECOPS_H_INCLUDED
#define VECOPS_H_INCLUDED

#include <vector>

using namespace std;

void vec_plus_equal(vector<float>& a, const vector<float>& b);
vector<float> vec_plus(vector<float>& a, const vector<float>& b);

void vec_sub_equal(vector<float>& a, const vector<float>& b);
vector<float> vec_sub(vector<float>& a, const vector<float>& b);

void vec_div_equal(vector<float>& a, const vector<float>& b);
void vec_div_equal(vector<float>& a, const float b);
vector<float> vec_div(vector<float>& a, const vector<float>& b);
vector<float> vec_div(vector<float>& a, const float b);

void vec_prod_equal(vector<float>& a, const vector<float>& b);
void vec_prod_equal(vector<float>& a, const float b);
vector<float> vec_prod(vector<float>& a, const vector<float>& b);
vector<float> vec_prod(vector<float>& a, const float b);

vector<float> vec_square(vector<float>& a);
vector<float> vec_sqrt(vector<float>& a);

#endif