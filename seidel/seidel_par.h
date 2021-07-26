#ifndef HASC_SEIDEL_PAR_H
#define HASC_SEIDEL_PAR_H
#include <cassert>
#include "span.h"
#include "util.h"

namespace hasc
{
inline __attribute__((always_inline))
double symmetric_seidel_2d_kernel(int n, int a_begin, int a_end, int b_begin, int b_end,
                                  int n_coeff, int coeff_a_begin, int coeff_b_begin,
                                  span<double> u, span<const double> coeff)
{
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
