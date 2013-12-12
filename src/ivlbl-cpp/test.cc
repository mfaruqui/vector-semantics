#include <iostream>     // std::cout
#include <cstddef>      // std::size_t
#include <cmath>        // std::pow
#include <vector>

using namespace std;
typedef Matrix< float, 1, Dynamic > RowVectorFloat;

int main ()
{
  //vector<float> val(5, 1);
  //vector<float> val2(5, 2), c;
  RowVectorFloat val(5);
  RowVectorFloat val2(5), c(5);
  val[3] = -1;
  
  c = val + val2;
  for (int i=0; i<c.size(); i++)
      cout << c[i] << "\n";

  return 1;
}