#ifndef HASC_UTIL_SH
#define HASC_UTIL_SH
#include <cstdio>
#include <cmath>

// row-major index mapping
// TODO: only define here? (include from other headers that need it)
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

inline void model_coefficients_2d(double* coeff, int n, double factor = 50.0, int m = 100)
{
  const int k = (n-1)/2; // center at (k, k)

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
    {
      const int idx = INDEX(i, j, n);
      const int a_abs = abs(i-k);
      const int b_abs = abs(j-k);

      if (i == k && j == k) // offset by k
        coeff[idx] = factor/(factor+m);
      else
        coeff[idx] = 1./(MAX(a_abs, b_abs)*MAX(a_abs, b_abs))/(factor+m);
    }
}

} // namespace hasc

#endif // HASC_UTIL_SH
