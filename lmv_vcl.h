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

/// @brief Local mean value kernel
/// @tparam W SIMD width (2 for SSE2, 4 for AVX, 8 for AVX512)
template <size_t W>
inline void lmv_2d_vectorized(int i0, int i1, int j0, int j1, int n, int k,
                              span<const double> u, span<double> mean)
{
  using VecWd = typename SIMDSelector<W>::SIMDType;

  // Precondition checks
  assert(k >= 1);
  assert(u.ssize() == (ptrdiff_t)(n*n));

  // Constant factor 1 / |Sigma|
  size_t cardSigma = (2*k + 1) * (2*k + 1);
  double factor = 1./cardSigma;

  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++)
    {
      const size_t center = INDEX(i, j, n);
      mean[center] = 0;

      for (int k0 = -k; k0 <= 2*k; ++k0)
        for (int k1 = -k; k1 <= 2*k; ++k1)
        {
          mean[center] += u[INDEX(i+k0, j+k1, n)];
        }
      mean[center] *= factor;
    }
}

/// @brief Local mean value kernel (vectorized version)
/// @tparam W SIMD width (2 for SSE2, 4 for AVX, 8 for AVX512)
template <size_t W>
inline void lmv_2d_vectorized(int n, int k, span<const double> u, span<double> mean)
{
  const int n0 = k;
  const int n1 = n - k;

  lmv_2d_vectorized<W>(n0, n1, n0, n1, n, k, u, mean);
}

/// @brief Local mean value kernel (blocked and vectorized version)
/// @tparam MJ Block size (columns)
/// @tparam MI Block size (rows)
/// @tparam W SIMD width (2 for SSE2, 4 for AVX, 8 for AVX512)
template <size_t MJ, size_t MI, size_t W>
inline void lmv_2d_blocked_vectorized(int n, int k, span<const double> u, span<double> mean)
{
  const int n0 = k;
  const int n1 = n - k;

  for (int J = n0; J < n1; J+=MJ)
    for (int I = n0; I < n1; I+=MI)
      lmv_2d_vectorized<MJ, MI, W>(I, min(n1, I+MI),
                                   J, min(n1, J+MJ), n, k, u, mean);
}

#endif // HASC_LMV_VCL_H
