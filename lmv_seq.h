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
#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define max(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{

// Vanilla version
inline void lmv_2d(int n, int k, span<const double> u, span<double> mean)
{
  // Precondition checks
  assert(k >= 1 && n > k);
  assert(u.ssize() == n*n);

  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
    {
      const size_t center = INDEX(i, j, n);
      mean[center] = 0;

      int a_begin = max(i-k, 0);
      int a_end = min(i-k, n-1);
      int b_begin = max(j-k, 0);
      int b_end = min(j+k, n-1);

      for (int b = b_begin; b <= b_end; ++b)
        for (int a = a_begin; a <= a_end; ++a)
        {
          mean[center] += u[INDEX(a, b, n)];
        }
      double factor = 1/((a_end - a_begin)*(b_end - b_begin));
      mean[center] *= factor;
    }
}

// Version which uses a padded array n. Because the factor 1/|Sigma| is constant for
// every point, this version will have different results on the boundary.
inline void lmv_2d_padded(int pn, int k, span<const double> pu, span<double> pmean)
{
  // Precondition checks
  assert(k >= 1 && pn > k);
  assert(pu.ssize() == pn*pn);

  // Constant factor for all points
  const double factor = 1./((2*k + 1)*(2*k + 1));

  for (int j = k; j < pn-k; j++)
    for (int i = k; i < pn-k; i++)
    {
      const size_t center = INDEX(i, j, pn);
      pmean[center] = 0;

      for (int k0 = -k; k0 <= 2*k; ++k0)
        for (int k1 = -k; k1 <= 2*k; ++k1)
        {
          pmean[center] += pu[INDEX(i+k0, j+k1, pn)];
        }
      pmean[center] *= factor;
    }
}

} // namespace hasc

#endif // HASC_LMV_SEQ_H
