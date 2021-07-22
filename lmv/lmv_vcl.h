/// @file
/// @brief Vectorized algorithms for computing local mean value
/// @author Ferdinand Vanmaele

#ifndef HASC_LMV_VCL_H
#define HASC_LMV_VCL_H
#include <cstring>
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
inline void lmv_2d_vectorized_kernel(int i0, int i1, int j0, int j1, int n, int k,
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
  lmv_2d_vectorized_kernel<W>(0, n, 0, n, n, k, u, mean);
}

// Basic blocked version which does not distinguish between outer and inner blocks.
template <int MI, int MJ, int W>
inline void lmv_2d_vectorized_blocked(int n, int k, span<const double> u, span<double> mean)
{
  for (int I = 0; I < n; I+=MI)
    for (int J = 0; J < n; J+=MJ)
      lmv_2d_vectorized_kernel<W>(I, MIN(I+MI, n),
                                  J, MIN(J+MJ, n), n, k, u, mean);
}

template <int W, int MaxSize = 256>
inline void lmv_2d_vectorized_buffered_kernel(int i0, int i1, int j0, int j1, int n, int k,
                                              span<const double> u, span<double> mean)
{
  assert(n % W == 0);
  assert(W+2*k <= MaxSize);

  using VecWd = typename SIMDSelector<W>::SIMDType;

  for (int i = i0; i < i1; ++i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);

    for (int j = j0; j+W <= j1; j+=W)
    {
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k+W-1, n-1);  // neighborhood extends in j direction

      // buffer for row of neighborhood centered in (i,j)...(i,j+W-1)
      double buf[MaxSize];
      ptrdiff_t buf_used = W+2*k;
      memset(buf, 0, sizeof(double)*buf_used); // zero-padded

      span<double> Sbuf(buf, buf_used);
      int offset = j-k < 0 ? k-j : 0;

      for (int a = a_begin; a <= a_end; ++a)
      {
        int b;
        int count = 0;
        for (b = b_begin; b+W <= b_end; b+=W)
        {
          VecWd Vseg1, Vseg2;
          Vseg1.load(&buf[offset+count*W]);
          Vseg2.load(&u[INDEX(a, b, n)]);

          VecWd Vseg = Vseg1 + Vseg2;
          Vseg.store(&buf[offset+count*W]);  // update buffer
          count++;
        }

        VecWd Vseg1, Vseg2;
        int remainder = b_end-b+1;
        Vseg1.load_partial(remainder, &buf[offset+count*W]);
        Vseg2.load_partial(remainder, &u[INDEX(a, b, n)]);

        VecWd Vseg = Vseg1 + Vseg2;
        Vseg.store_partial(remainder, &buf[offset+count*W]);
      }

      // Compute local mean values (shifted, zero additions on boundary points)
      const size_t center = INDEX(i, j, n);
      for (int c = 0; c < W; ++c)
      {
        const int bc_begin = MAX(j+c-k, 0);
        const int bc_end = MIN(j+c+k, n-1);
        const double factor = 1./((bc_end-bc_begin+1)*(a_end-a_begin+1));

        mean[center+c] = 0;
        for (int bi = 0; bi <= 2*k; ++bi)
        {
          mean[center+c] += Sbuf[bi+c]; // horizontal add
        }
        mean[center+c] *= factor;
      }
    }
  }
}

template <int W>
inline void lmv_2d_vectorized_buffered(int n, int k, span<const double> u, span<double> mean)
{
  lmv_2d_vectorized_buffered_kernel<W>(0, n, 0, n, n, k, u, mean);
}

template <int MI, int MJ, int W>
inline void lmv_2d_vectorized_buffered_blocked(int n, int k, span<const double> u, span<double> mean)
{
  for (int I = 0; I < n; I+=MI)
    for (int J = 0; J < n; J+=MJ)
      lmv_2d_vectorized_buffered_kernel<W>(I, MIN(I+MI, n),
                                           J, MIN(J+MJ, n), n, k, u, mean);
}

} // namespace hasc

#endif // HASC_LMV_VCL_H
