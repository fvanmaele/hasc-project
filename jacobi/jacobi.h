#ifndef HASC_JACOBI_H
#define HASC_JACOBI_H
#include <cassert>
#include "span.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{
inline void jacobi_2d_kernel(int i0, int i1, int j0, int j1, int n, int k, int iterations,
                             span<double> uold, span<double> unew, span<const double> coeff)
{
  assert(coeff.ssize() == (2*k+1)*(2*k+1)); // assume coefficient stencil of fixed size

  for (int it = 1; it <= iterations; ++it)
  {
    for (int i = i0; i < i1; ++i)
    {
      const int a_begin = MAX(i-k, 0);
      const int a_end = MIN(i+k, n-1);

      for (int j = j0; j < j1; ++j)
      {
        const int b_begin = MAX(j-k, 0);
        const int b_end = MIN(j+k, n-1);
        const size_t center = INDEX(i, j, n);
        size_t cnt = 0;

        // process coefficients from left to right
        unew[center] = 0;
        for (int a = a_begin; a <= a_end; ++a)
          for (int b = b_begin; b <= b_end; ++b, ++cnt)
          {
            unew[center] += coeff[cnt] * uold[INDEX(a, b, n)];
          }
      }
    }
    using std::swap;
    swap(uold, unew);
  }
}

inline void jacobi_2d(int n, int k, int iterations, span<double> uold, span<double> unew,
                      span<const double> coeff)
{
  jacobi_2d_kernel(0, n, 0, n, n, k, iterations, uold, unew, coeff);
}

template <int MI, int MJ>
inline void jacobi_2d_blocked(int n, int k, int iterations, span<double> uold, span<double> unew,
                              span<const double> coeff)
{
  for (int I = 0; I < n; I+=MI)
    for (int J = 0; J < n; J+=MJ)
      jacobi_2d_kernel(I, MIN(I+MI, n),
                       J, MIN(J+MJ, n), n, k, iterations, uold, unew, coeff);
}

template <int MI, int MJ>
inline void jacobi_2d_blocked_openmp(int n, int k, int iterations, span<double> uold, span<double> unew,
                                    span<const double> coeff)
{
#pragma omp parallel for collapse(2)
  for (int I = 0; I < n; I+=MI)
    for (int J = 0; J < n; J+=MJ)
      jacobi_2d_kernel(I, MIN(I+MI, n),
                       J, MIN(J+MJ, n), n, k, iterations, uold, unew, coeff);
}

} // namespace hasc

#endif // HASC_JACOBI_H
