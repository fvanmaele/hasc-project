#ifndef HASC_SEIDEL_H
#define HASC_SEIDEL_H
#include <cassert>
#include "span.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{
namespace detail
{
inline void update_neighbors(int i, int j, int n, int k, int a_begin, int a_end, int b_begin, int b_end,
                             span<double> u, span <const double> coeff)
{
  const size_t center = INDEX(i, j, n);
  size_t cnt = 0;

  // process coefficients from left to right
  for (int a = a_begin; a <= a_end; ++a)
  {
    for (int b = b_begin; b <= b_end; ++b, ++cnt)
    {
      const size_t idx = INDEX(a, b, n);
      u[center] += coeff[cnt] * u[idx];
    }
  }
}

} // namespace detail

inline void symmetric_seidel_2d(int n, int k, int iterations, span<double> u,
                                span<const double> coeff)
{
  assert(coeff.size() == (2*k+1)*(2*k+1)); // assume coefficient stencil of fixed size

  for (int it = 1; it <= iterations; ++it)
  {
    // forward sweep
    for (int i = 0; i < n; ++i)
    {
      const int a_begin = MAX(i-k, 0);
      const int a_end = MIN(i+k, n-1);

      for (int j = 0; j < n; ++j)
      {
        const int b_begin = MAX(j-k, 0);
        const int b_end = MIN(j+k, n-1);
        detail::update_neighbors(i, j, n, k, a_begin, a_end, b_begin, b_end, u, coeff);
      }
    }

    // backward sweep
    for (int i = n-1; i >= 0; --i)
    {
      const int a_begin = MAX(i-k, 0);
      const int a_end = MIN(i+k, n-1);

      for (int j = n-1; j >= 0; --j)
      {
        const int b_begin = MAX(j-k, 0);
        const int b_end = MIN(j+k, n-1);
        detail::update_neighbors(i, j, n, k, a_begin, a_end, b_begin, b_end, u, coeff);
      }
    }
    // proceed to next iteration
  }
}

} // namespace hasc

#endif // HASC_SEIDEL_H
