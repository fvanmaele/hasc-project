/// @file
/// @brief Sequential algorithms for computing local mean value
/// @author Ferdinand Vanmaele

#ifndef HASC_LMV_SEQ_H
#define HASC_LMV_SEQ_H
#include <cassert>
#include "span.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{

// Version with dynamic bounds, suitable for boundary points
inline void lmv_2d(int i0, int i1, int j0, int j1, int n, int k,
                   span<const double> u, span<double> mean)
{
  for (int i = i0; i < i1; ++i)
  {
    const int a_begin = MAX(i-k, 0);
    const int a_end = MIN(i+k, n-1);

    for (int j = j0; j < j1; ++j)
    {
      const int b_begin = MAX(j-k, 0);
      const int b_end = MIN(j+k, n-1);
      const double factor = 1./((a_end-a_begin+1)*(b_end-b_begin+1));
      const size_t center = INDEX(i, j, n);
      mean[center] = 0;

      for (int a = a_begin; a <= a_end; ++a)
        for (int b = b_begin; b <= b_end; ++b)
        {
          mean[center] += u[INDEX(a, b, n)];
        }
      mean[center] *= factor;
    }
  }
}

inline void lmv_2d(int n, int k, span<const double> u, span<double> mean)
{
  lmv_2d(0, n, 0, n, n, k, u, mean);
}

// Basic blocked version which does not distinguish between outer and inner blocks.
template <int MI, int MJ>
inline void lmv_2d_blocked(int n, int k, span<const double> u, span<double> mean)
{
  for (int I = 0; I < n; I+=MI)
    for (int J = 0; J < n; J+=MJ)
      lmv_2d(I, MIN(I+MI, n),
             J, MIN(J+MJ, n), n, k, u, mean);
}

} // namespace hasc

#endif // HASC_LMV_SEQ_H
