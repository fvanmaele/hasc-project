#ifndef HASC_STENCIL_H
#define HASC_STENCIL_H
#include <cassert>
#include "simd_selector.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))


namespace hasc
{
namespace detail
{
bool check_bounds(int i0, int i1, int j0, int j1, int n, int k)
{
  return (INDEX(i0-k, j0-k, n) >= 0)        // bottom-left
      && (INDEX(i1-1+k, j1-1+k, n) < n*n);  // right-top
}

} // namespace detail


inline void lmv_kernel_2d(int i0, int i1, int j0, int j1, int n, int k,
                          const double *__restrict__ u, double *__restrict__ mean)
{
  // Precondition checks
  assert(k >= 1);
  assert(detail::check_bounds(i0, i1, j0, j1, n, k));

  // Constant factor 1 / |Sigma|
  size_t cardSigma = (2*k + 1) * (2*k + 1);
  double factor = 1./cardSigma;

  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++)
    {
      const size_t center = INDEX(i, j, n);
      const size_t bottom_left = INDEX(i-k, j-k, n);
      const size_t top_right = INDEX(i+k, j+k, n);

      // note: both u and mean are assumed of size n*n (and include padding)
      // this allows composability (i.e. multiple applications of the kernel)
      mean[center] = 0;
      for (size_t idx = bottom_left; idx <= top_right; idx++)
      {
        mean[center] += u[idx];
      }
      mean[center] *= factor;
    }
}

inline void lmv_kernel_2d(int n, int k, const double *__restrict__ u,
                          double *__restrict__ mean)
{
  const int n0 = k;
  const int n1 = n - k;

  lmv_kernel_2d(n0, n1, n0, n1, n, k, u, mean);
}

template <int W>
inline void lmv_kernel_2d_vectorized(int i0, int i1, int j0, int j1, int n, int k,
                                     const double *__restrict__ u, double *__restrict__ mean)
{
  using VecWd = typename SIMDSelector<W>::SIMDType;

  // Precondition checks
  assert(k >= 1);
  assert(detail::check_bounds(i0, i1, j0, j1, n, k));

  // Constant factor 1 / |Sigma|
  size_t cardSigma = (2*k + 1) * (2*k + 1);
  double factor = 1./cardSigma;

  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++)
    {
      const size_t center = INDEX(i, j, n);
      const size_t bottom_left = INDEX(i-k, j-k, n);
      const size_t top_right = INDEX(i+k, j+k, n);

      // note: both u and mean are assumed of size n*n (and include padding)
      // this allows composability (i.e. multiple applications of the kernel)
      VecWd meanWd(0);
      size_t idx;
      for (idx = bottom_left; idx+W <= top_right; idx+=W)
      {
        VecWd tmp;
        tmp.load(&u[idx]);
        meanWd += tmp;
      }
      mean[center] = horizontal_add(meanWd);

      // remaining elements
      for (size_t r = idx; r <= top_right; ++r)
      {
        mean[center] += u[idx];
      }
      mean[center] *= factor;
    }
}

template <int W>
inline void lmv_kernel_2d_vectorized(int n, int k, const double *__restrict__ u,
                                     double *__restrict__ mean)
{
  const int n0 = k;
  const int n1 = n - k;

  lmv_kernel_2d_vectorized<W>(n0, n1, n0, n1, n, k, u, mean);
}

template <int MJ, int MI>
inline void lmv_kernel_2d_blocked(int n, int k, const double *__restrict__ u,
                                  double *__restrict__ mean)
{
  const int n0 = k;
  const int n1 = n - k;

  for (int J = n0; J < n1; J+=MJ)
    for (int I = n0; I < n1; I+=MI)
      lmv_kernel_2d(I, MIN(n1, I+MI),
                    J, MIN(n1, J+MJ), n, k, u, mean);
}

template <int MJ, int MI, int W>
inline void lmv_kernel_2d_blocked_vectorized(int n, int k, const double *__restrict__ u,
                                             double *__restrict__ mean)
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
