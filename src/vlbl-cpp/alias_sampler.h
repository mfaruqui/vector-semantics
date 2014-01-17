#ifndef _ALIAS_SAMPLER_H_
#define _ALIAS_SAMPLER_H_

#include <vector>
#include <limits>
#include <random>

using namespace std;

// R. A. Kronmal and A. V. Peterson, Jr. (1977) On the alias method for
// generating random variables from a discrete distribution. In The American
// Statistician, Vol. 33, No. 4. Pages 214--218.
//
// Intuition: a multinomial with N outcomes can be rewritten as a uniform
// mixture of N Bernoulli distributions. The ith Bernoulli returns i with
// probability F[i], otherwise it returns an "alias" value L[i]. The
// constructor computes the F's and L's given an arbitrary multionimial p in
// O(n) time and Draw returns samples in O(1) time.

/* By Manaal */
default_random_engine generator;
uniform_real_distribution<double> distribution(0.0,1.0);
    
struct AliasSampler {
  AliasSampler() {}
  explicit AliasSampler(const vector<double>& p) { Init(p); }
  /* By Manaal */
  void initialise(const vector<double>& p) { Init(p); }
  void Init(const vector<double>& p) {
    const unsigned N = p.size();
    cutoffs_.resize(p.size());
    aliases_.clear();
    aliases_.resize(p.size(), numeric_limits<unsigned>::max());
    vector<unsigned> s,g;
    for (unsigned i = 0; i < N; ++i) {
      const double cutoff = cutoffs_[i] = N * p[i];
      if (cutoff >= 1.0) g.push_back(i); else s.push_back(i);
    }
    while(!s.empty() && !g.empty()) {
      const unsigned k = g.back();
      const unsigned j = s.back();
      aliases_[j] = k;
      cutoffs_[k] -= 1.0 - cutoffs_[j];
      s.pop_back();
      if (cutoffs_[k] < 1.0) {
        g.pop_back();
        s.push_back(k);
      }
    }
  }
  
  /* By Manaal */
  unsigned Draw() {
    unsigned n = distribution(generator) * cutoffs_.size();
    if (distribution(generator) > cutoffs_[n]) return aliases_[n]; else return n;
  }
  
  vector<double> cutoffs_;    // F
  vector<unsigned> aliases_;  // L
};

#endif