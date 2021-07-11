/// @file local_mean_value.h
/// @brief Sequential algorithm for computing local mean value
/// @author Ferdinand Vanmaele

#ifndef HASC_STENCIL_H
#define HASC_STENCIL_H
#include <cassert>
#include "simd_selector.h"
#include "span.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))
// minimum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))


namespace hasc
{
/// @brief Local mean value kernel (vanilla version)
inline void lmv_kernel_2d(int i0, int i1, int j0, int j1, int n, int k,
                          span<const double> u, span<double> mean)
{
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

/// @brief Local mean value kernel (vanilla version)
/// @details Compute \f$M_{i,j} = \frac{1}{|\Sigma_{i,j}^{k,n}|}\sum_{r,s\in\Sigma} u_{i,j}\f$
/// To handle boundary conditions, the input array u is assumed to be padded in both directions.
/// @param u 2-dimensional grid of size n*n, accessed in row-major order
/// @param n row size of u
/// @param k local mean value radius
/// @param mean array holding mean local values of size n*n
inline void lmv_kernel_2d(int n, int k, span<const double> u, span<double> mean)
{
  const int n0 = k;
  const int n1 = n - k;

  lmv_kernel_2d(n0, n1, n0, n1, n, k, u, mean);
}

/// @brief Local mean value kernel (vectorized version)
/// @tparam W SIMD width (2 for SSE2, 4 for AVX, 8 for AVX512)
template <int W>
inline void lmv_kernel_2d_vectorized(int i0, int i1, int j0, int j1, int n, int k,
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
template <int W>
inline void lmv_kernel_2d_vectorized(int n, int k, span<const double> u, span<double> mean)
{
  const int n0 = k;
  const int n1 = n - k;

  lmv_kernel_2d_vectorized<W>(n0, n1, n0, n1, n, k, u, mean);
}

/// @brief Local mean value kernel (blocked version)
/// @tparam MJ Block size (columns)
/// @tparam MI Block size (rows)
template <int MJ, int MI>
inline void lmv_kernel_2d_blocked(int n, int k, span<const double> u, span<double> mean)
{
  const int n0 = k;
  const int n1 = n - k;

  for (int J = n0; J < n1; J+=MJ)
    for (int I = n0; I < n1; I+=MI)
      lmv_kernel_2d(I, MIN(n1, I+MI),
                    J, MIN(n1, J+MJ), n, k, u, mean);
}

/// @brief Local mean value kernel (blocked and vectorized version)
/// @tparam MJ Block size (columns)
/// @tparam MI Block size (rows)
/// @tparam W SIMD width (2 for SSE2, 4 for AVX, 8 for AVX512)
template <int MJ, int MI, int W>
inline void lmv_kernel_2d_blocked_vectorized(int n, int k, span<const double> u, span<double> mean)
{
  const int n0 = k;
  const int n1 = n - k;

  for (int J = n0; J < n1; J+=MJ)
    for (int I = n0; I < n1; I+=MI)
      lmv_kernel_2d_vectorized<MJ, MI, W>(I, MIN(n1, I+MI),
                                          J, MIN(n1, J+MJ), n, k, u, mean);
}

} // namespace hasc

#endif // HASC_STENCIL_H
