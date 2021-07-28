#ifndef HASC_SEIDEL_PAR_H
#define HASC_SEIDEL_PAR_H
#include <cassert>
#include "span.h"
#include "util.h"

namespace hasc
{
inline __attribute__((always_inline))
double symmetric_seidel_2d_kernel(int n, int k, int i, int j,
                                  span<double> u, span<const double> coeff)
{
  const int n_coeff = 2*k+1;
  const int a_begin = MAX(i-k, 0);
  const int a_end = MIN(i+k, n-1);
  const int coeff_a_begin = MAX(a_begin-i+k, 0);

  const int b_begin = MAX(j-k, 0);
  const int b_end = MIN(j+k, n-1);
  const int coeff_b_begin = MAX(b_begin-j+k, 0);

  double tmp = 0;
  for (int a = a_begin, coeff_a = coeff_a_begin; a <= a_end; ++a, ++coeff_a)
  {
    const size_t offset = INDEX(a, b_begin, n);
    const size_t coeff_offset = INDEX(coeff_a, coeff_b_begin, n_coeff);

    for (int cb = 0; cb <= b_end-b_begin; ++cb)
      tmp += coeff[coeff_offset+cb] * u[offset+cb];
  }
  return tmp;
}

} // namespace hasc

#endif // HASC_SEIDEL_PAR_H
