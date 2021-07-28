/// @file jacobi.h
/// @brief Implementation of the Jacobi method for general stencils
/// @author Ferdinand Vanmaele

#ifndef HASC_JACOBI_H
#define HASC_JACOBI_H
#include <cassert>
#include "span.h"
#include "util.h"

namespace hasc
{
inline void model_coefficients_2d(double* coeff, int n, double factor = 50.0, int m = 100)
{
  const int k = (n-1)/2; // center at (k, k)

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
    {
      const int idx = INDEX(i, j, n);
      const int a_abs = abs(i-k);
      const int b_abs = abs(j-k);

      if (i == k && j == k) // offset by k
        coeff[idx] = factor/(factor+m);
      else
        coeff[idx] = 1./(MAX(a_abs, b_abs)*MAX(a_abs, b_abs))/(factor+m);
    }
}

inline void jacobi_2d_kernel(int i0, int i1, int j0, int j1, int n, int k,
                             span<double> uold, span<double> unew, span<const double> coeff)
{
  // assume coefficient stencil of fixed size
  const int n_coeff = 2*k+1;
  assert(coeff.ssize() == n_coeff*n_coeff);

  for (int i = i0; i < i1; ++i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);

    // coefficients are zero-indexed (offset k)
    const int coeff_a_begin = MAX(a_begin-i+k, 0);
    //const int coeff_a_end = MIN(a_end-i+k, n_coeff-1);
    assert(MIN(a_end-i+k, n_coeff-1) - coeff_a_begin == a_end - a_begin);

    for (int j = j0; j < j1; ++j)
    {
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k, n-1);

      const int coeff_b_begin = MAX(b_begin-j+k, 0);
      //const int coeff_b_end = MIN(b_end-j+k, n_coeff-1);
      assert(MIN(b_end-j+k, n_coeff-1) - coeff_b_begin == b_end - b_begin);

      const size_t center = INDEX(i, j, n);
      //size_t cnt = 0;

      // process coefficients from left to right
      // more complex to correctly retrieve coefficients for boundary points
      unew[center] = 0;
      for (int a = a_begin, coeff_a = coeff_a_begin; a <= a_end; ++a, ++coeff_a)
      {
        const size_t offset = INDEX(a, b_begin, n);
        const size_t coeff_offset = INDEX(coeff_a, coeff_b_begin, n_coeff);

        for (int cb = 0; cb <= b_end-b_begin; ++cb)
          unew[center] += coeff[coeff_offset+cb] * uold[offset+cb];
      }
    }
  }
}

inline void jacobi_2d(int n, int k, int iterations, span<double> uold, span<double> unew,
                      span<const double> coeff)
{
  for (int it = 1; it <= iterations; ++it)
  {
    jacobi_2d_kernel(0, n, 0, n,
                     n, k, uold, unew, coeff);
    using std::swap;
    swap(uold, unew);
  }
}

template <int MI, int MJ>
inline void jacobi_2d_blocked(int n, int k, int iterations, span<double> uold, span<double> unew,
                              span<const double> coeff)
{
  for (int it = 1; it <= iterations; ++it)
  {
    for (int I = 0; I < n; I+=MI)
      for (int J = 0; J < n; J+=MJ)
        jacobi_2d_kernel(I, MIN(I+MI, n),
                         J, MIN(J+MJ, n), n, k, uold, unew, coeff);
    using std::swap;
    swap(uold, unew);
  }
}

template <int MI, int MJ>
inline void jacobi_2d_blocked_openmp(int n, int k, int iterations, span<double> uold, span<double> unew,
                                     span<const double> coeff)
{
  for (int it = 1; it <= iterations; ++it)
  {
#pragma omp parallel for collapse(2)
    for (int I = 0; I < n; I+=MI)
      for (int J = 0; J < n; J+=MJ)
        jacobi_2d_kernel(I, MIN(I+MI, n),
                         J, MIN(J+MJ, n), n, k, uold, unew, coeff);
    using std::swap;
    swap(uold, unew);
  }
}

} // namespace hasc

#endif // HASC_JACOBI_H
