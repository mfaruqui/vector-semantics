#include <assert.h>
#include <math.h>

double log_add(double lx, double ly)
{
  /* REQUIRES !isnan(lx) && !isnan(ly) */
  if (isinf(lx) && !signbit(lx))
    return lx;
  if (isinf(ly) && !signbit(ly))
    return ly;
  if (isinf(lx) && signbit(lx))
    return ly;
  if (isinf(ly) && signbit(ly))
    return lx;
  /* This assertion can fail if lx or ly is NaN. */
  assert(isfinite(lx) && isfinite(ly));

  double d = lx - ly;
  if (d >= 0) {
    assert(lx >= ly);
    if (d > 745) {
      /* exp(-d) is 0 for 64 bit doubles */
      return lx;
    } else {
      assert(exp(-d) > 0);
      return lx + log1p(exp(-d));
    }
  } else {
    assert(lx < ly);
    if (d < -745) {
      /* exp(d) is 0 for 64 bit doubles */
      return ly;
    } else {
      assert(exp(d) > 0);
      return ly + log1p(exp(d));
    }
  }
}