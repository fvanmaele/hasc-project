/// @file
/// @brief Vectorized algorithms for computing local mean value
/// @author Ferdinand Vanmaele

#ifndef HASC_LMV_VCL_H
#define HASC_LMV_VCL_H
#include <cassert>
#include "span.h"
#include "simd_selector.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{

// Version which vectorizes over every point (i,j)
template <int W>
inline void lmv_2d_vectorized(int i0, int i1, int j0, int j1, int n, int k,
                              span<const double> u, span<double> mean)
{
  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (int i = i0; i < i1; ++i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);

    for (int j = j0; j < j1; ++j)
    {
      const size_t center = INDEX(i, j, n);
      mean[center] = 0;
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k, n-1);
      const double factor = 1./((a_end-a_begin+1)*(b_end-b_begin+1));

      // Row-wise vector loads
      VecWd Vmean(0);
      for (int a = a_begin; a <= a_end; ++a)
      {
        int b;
        VecWd Vrow;
        for (b = b_begin; b+W <= b_end; b+=W)
        {
          Vrow.load(&u[INDEX(a, b, n)]);
          Vmean += Vrow;
        }
        Vrow.load_partial(b_end-b+1, &u[INDEX(a, b, n)]);
        Vmean += Vrow;
      }
      mean[center] += horizontal_add(Vmean);
      mean[center] *= factor;
    }
  }
}

template <int W>
inline void lmv_2d_vectorized(int n, int k, span<const double> u, span<double> mean)
{
  lmv_2d_vectorized<W>(0, n, 0, n, n, k, u, mean);
}

// Basic blocked version which does not distinguish between outer and inner blocks.
template <int MI, int MJ, int W>
inline void lmv_2d_blocked_vectorized(int n, int k, span<const double> u, span<double> mean)
{
  for (int J = 0; J < n; J+=MJ)
    for (int I = 0; I < n; I+=MI)
      lmv_2d_vectorized<W>(I, MIN(I+MI, n), J, MIN(J+MI, n), n, k, u, mean);
}

// FIXME: code broken
template <int W>
inline void lmv_2d_vectorized_2(int n, int k, span<const double> u, span<double> mean)
{
  // Divisiblity constraints
  assert(n % W == 0);

  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (int i = 0; i < n; ++i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);

    int j;
    for (j = 0; j+W < n; j+=W)
    {
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k+W, n-1);  // neighborhood extends in j direction

      // buffer for row in neighborhood of u_{i,j}
      double rowk[W+2*k] = {0}; // TODO: alignment?

      for (int a = a_begin; a <= a_end; ++a)
      {
        int b;
        for (b = b_begin; b+W <= b_end; b+=W)
        {
          const int offset = b-b_begin;
          VecWd Vseg1, Vseg2;
          Vseg1.load(&rowk[offset]);
          Vseg2.load(&u[INDEX(a, b, n)]);

          VecWd Vseg = Vseg1 + Vseg2;
          Vseg.store(&rowk[offset]); // update buffer
        }

        const int offset = b-b_begin;
        VecWd Vseg1, Vseg2;
        Vseg1.load_partial(b_end-b+1, &rowk[offset]);
        Vseg2.load_partial(b_end-b+1, &u[INDEX(a, b, n)]);

        VecWd Vseg = Vseg1 + Vseg2;
        Vseg.store_partial(a_end-a, &rowk[offset]);
      }

      // Compute local mean values (shifted by pos+k)
      const size_t center = INDEX(i, j, n);
      for (int c = 0; c < W; ++c)
      {
        const int b_begin_local = MAX(j+c-k, 0);
        const int b_end_local = MIN(j+c+k, n-1);
        const int a_n = a_end-a_begin+1;

        const double factor = 1./((b_end_local-b_begin_local+1)*a_n);
        for (int l = 0; l < a_n; ++l)
        {
          mean[center+c] += rowk[l];
        }
        mean[center+c] *= factor;
      }
    }
  }
}

} // namespace hasc

#endif // HASC_LMV_VCL_H
