#ifndef HASC_STENCIL_H
#define HASC_STENCIL_H
#include <cstddef>

namespace hasc
{
namespace detail
{
  template<class T>
  const T& min(const T& a, const T& b)
  {
    return (b < a) ? b : a;
  }

  size_t index_2d(int i, int j, int n)
  {
    return i*n + j;
  }
} // namespace detail


template <typename FloatType = double>
void lmv_kernel_2d(int i0, int i1, int j0, int j1, int radius,
                   const FloatType *__restrict__ u, FloatType *__restrict__ M)
{
  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++)
      for (int k = 1; k <= radius; k++)
      {
        // local mean value operation (assumes padded arrays)
        size_t idx = detail::index_2d(i, j, n);

      }
}

template <int MJ, int MI, typename FloatType = double>
void lmv_kernel_2d_blocked(int n, int radius,
                           const FloatType *__restrict__ u, FloatType *__restrict__ M)
{
  const int n0 = radius; // XXX: take grid size without padding?
  const int n1 = n - radius;

  for (int J = n0; J < n1; J+=MJ)
    for (int I = n0; I < n1; I+=MI)
      lmv_kernel_2d<FloatType>(I, detail::min(n1, I+MI),
                               J, detail::min(n1, J+MJ), radius, u, M);
}

} // namespace hasc

#endif // HASC_STENCIL_H
