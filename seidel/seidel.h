#ifndef HASC_SEIDEL_H
#define HASC_SEIDEL_H
#include <cassert>
#include "span.h"
#include "util.h"

namespace hasc
{
inline __attribute__((always_inline))
void symmetric_seidel_2d_forward(int i0, int i1, int j0, int j1, int n, int k,
                                 span<double> u, span<const double> coeff)
{
  const int n_coeff = 2*k+1;

  for (int i = i0; i < i1; ++i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);
    const int coeff_a_begin = MAX(a_begin-i+k, 0);

    for (int j = j0; j < j1; ++j)
    {
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k, n-1);
      const int coeff_b_begin = MAX(b_begin-j+k, 0);
      //printf("index: i = %d, j = %d\n", i, j);

      double tmp = 0;
      for (int a = a_begin, coeff_a = coeff_a_begin; a <= a_end; ++a, ++coeff_a)
      {
        const size_t offset = INDEX(a, b_begin, n);
        const size_t coeff_offset = INDEX(coeff_a, coeff_b_begin, n_coeff);

        for (int cb = 0; cb <= b_end-b_begin; ++cb)
          tmp += coeff[coeff_offset+cb] * u[offset+cb];
      }
      const size_t center = INDEX(i, j, n);
      u[center] = tmp;
    }
  }
}

// Only the loop bounds change in this version, the rest is the same...
inline __attribute__((always_inline))
void symmetric_seidel_2d_backward(int i1, int i0, int j1, int j0, int n, int k,
                                         span<double> u, span<const double> coeff)
{
  const int n_coeff = 2*k+1;

  for (int i = i1-1; i >= i0; --i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);
    const int coeff_a_begin = MAX(a_begin-i+k, 0);

    for (int j = j1-1; j >= j0; --j)
    {
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k, n-1);
      const int coeff_b_begin = MAX(b_begin-j+k, 0);
      //printf("index: i = %d, j = %d\n", i, j);

      double tmp = 0;
      for (int a = a_begin, coeff_a = coeff_a_begin; a <= a_end; ++a, ++coeff_a)
      {
        const size_t offset = INDEX(a, b_begin, n);
        const size_t coeff_offset = INDEX(coeff_a, coeff_b_begin, n_coeff);

        for (int cb = 0; cb <= b_end-b_begin; ++cb)
          tmp += coeff[coeff_offset+cb] * u[offset+cb];
      }
      const size_t center = INDEX(i, j, n);
      u[center] = tmp;
    }
  }
}

inline void symmetric_seidel_2d(int n, int k, int iterations, span<double> u,
                                span<const double> coeff)
{
  for (int it = 1; it <= iterations; ++it)
  {
    // forward sweep
    symmetric_seidel_2d_forward(0, n, 0, n,
                                n, k, u, coeff);
    // backward sweep
    symmetric_seidel_2d_backward(n, 0, n, 0,
                                 n, k, u, coeff);
  }
}

template <int MI, int MJ>
inline void symmetric_seidel_2d_inexact(int n, int k, int iterations, span<double> u,
                                        span<const double> coeff)
{
  for (int it = 1; it <= iterations; ++it)
  {
    // forward sweep
    for (int I = 0; I < n; I+=MI)
      for (int J = 0; J < n; J+=MJ)
        symmetric_seidel_2d_forward(I, MIN(I+MI, n),
                                    J, MIN(J+MJ, n), n, k, u, coeff);
    // backward sweep
    for (int I = 0; I < n; I+=MI)
      for (int J = 0; J < n; J+=MJ)
        symmetric_seidel_2d_backward(MIN(I+MI, n), I,
                                     MIN(J+MJ, n), J, n, k, u, coeff);
  }
}

inline void symmetric_seidel_2d_openmp(int n, int k, int iterations, span<double> u,
                                       span<const double> coeff)
{
  // Assumed divisiblity constraints
  //assert(n % (k+1) == 0);

  const int max_width = k+1;
  const int max_height = n/(k+1);
  // total number of spaced diagonals
  const int s_diag_total = n + (n-(k+1))/(k+1);
  // number of spaced diagonals that can be processed in parallel
  const int s_diag_interior = s_diag_total - 2;

  for (int it = 1; it <= iterations; ++it)
  {
    //printf("step (seq)\n"); // first diagonal (bootstrap)
    symmetric_seidel_2d_forward(0, 1, 0, max_width, n, k, u, coeff);

    // process lower triangle
    int step = max_width;
    for (int s_diag = 1; s_diag < n; ++s_diag)
    {
      int s_height = MIN(s_diag+1, max_height);
      //printf("diag: %d, height: %d\n", s_diag, s_height);

      for (int t = 0; t < max_width; ++t) // time steps
      {
        //printf("step: %d\n", step);
#pragma omp parallel for
        for (int cnt = 0; cnt < s_height; ++cnt)
        {
          int i = MIN(s_diag-cnt, n-1);
          int j = cnt*max_width+t;
          //printf("index: i = %d, j = %d\n", i, j);
          symmetric_seidel_2d_forward(i, i+1, j, j+1, n, k, u, coeff);
        }
        ++step;
      }
    }

    // process upper triangle
    for (int s_diag = n; s_diag <= s_diag_interior; ++s_diag)
    {
      int s_height = MIN(s_diag_total-s_diag, max_height);
      //printf("diag: %d, height: %d\n", s_diag, s_height);

      for (int t = 0; t < max_width; ++t) // time steps
      {
#pragma omp parallel for
        for (int cnt = 0; cnt < s_height; ++cnt)
        {
          int i = n-1-cnt;
          int j = (s_diag-n+cnt+1)*max_width+t;
          //printf("index: i = %d, j = %d\n", i, j);
          symmetric_seidel_2d_forward(i, i+1, j, j+1, n, k, u, coeff);
        }
        ++step;
      }
    }

    // process last diagonal sequentially
    symmetric_seidel_2d_forward(n-1, n, n-max_width, n, n, k, u, coeff);

    // process last diagonal sequentially
    symmetric_seidel_2d_backward(n, n-1, n, n-max_width, n, k, u, coeff);

    --step;
    // process upper triangle
    for (int s_diag = s_diag_interior; s_diag >= n; --s_diag)
    {
      int s_height = MIN(s_diag_total-s_diag, max_height);
      //printf("diag: %d, height: %d\n", s_diag, s_height);

      for (int t = max_width-1; t >= 0; --t) // time steps
      {
        //printf("step: %d\n", step);
#pragma omp parallel for
        for (int cnt = s_height-1; cnt >= 0; --cnt)
        {
          int i = n-1-cnt;
          int j = (s_diag-n+cnt+1)*max_width+t;
          symmetric_seidel_2d_backward(i+1, i, j+1, j, n, k, u, coeff);
        }
        --step;
      }
    }

    // process lower triangle
    for (int s_diag = n-1; s_diag >= 1; --s_diag)
    {
      int s_height = MIN(s_diag+1, max_height);
      //printf("diag: %d, height: %d\n", s_diag, s_height);

      for (int t = max_width-1; t >= 0; --t) // time steps
      {
        //printf("step: %d\n", step);
#pragma omp parallel for
        for (int cnt = s_height-1; cnt >= 0; --cnt)
        {
          int i = s_diag-cnt;
          int j = cnt*max_width+t;
          symmetric_seidel_2d_backward(i+1, i, j+1, j, n, k, u, coeff);
        }
        --step;
      }
    }
    // process first diagonal sequentially
    symmetric_seidel_2d_backward(1, 0, max_width, 0, n, k, u, coeff);
  }
}

} // namespace hasc

#endif // HASC_SEIDEL_H
