/// @file util.h
/// @brief Various utility functions
/// @author Ferdinand Vanmaele

#ifndef HASC_UTIL_SH
#define HASC_UTIL_SH
#include <cstdio>
#include <cmath>

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{
inline void fill(double* x, size_t len, double a)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a;
}

inline void iota(double* x, size_t len, int a0 = 1)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a0++;
}

inline void print_array_2d(const double* x, int n)
{
  for (int i = 0; i < n; ++i)
  {
    std::printf("[");
    for (int j = 0; j < n-1; ++j)
      std::printf(" %8.4E", x[i*n+j]);
    std::printf(" %f ]\n", x[i*n+(n-1)]);
  }
}

inline bool isfinite_array(const double* x, size_t len)
{
  bool finite = true;
  for (size_t i = 0; i < len; ++i)
  {
    if (!std::isfinite(x[i]))
    {
      finite = false;
      break;
    }
  }
  return finite;
}

// Frobenius norm
inline double NormF(const double* x, size_t len)
{
  double sum = 0.0;
  for (size_t i = 0; i < len; ++i)
    sum += x[i]*x[i];
  return sum;
}

} // namespace hasc

#endif // HASC_UTIL_SH
