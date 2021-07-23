#ifndef HASC_UTIL_SH
#define HASC_UTIL_SH
#include <random>
#include <cstdlib>

// row-major index mapping
// TODO: only define here? (include from other headers that need it)
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{
template <typename T>
inline void model_coefficients_2d(T* coeff, int n, double factor = 50.0, int m = 100)
{
  const int k = (n-1)/2;

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

template <typename T>
inline T unifrnd(const int a, const int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<double> distr(a, b);
  return distr(generator);
}

template <typename T>
inline void fill(T* x, size_t len, T a)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a;
}

template <typename T>
inline void iota(T* x, size_t len, int a0 = 1)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a0++;
}

inline void print_array_2d(const double* x, int n)
{
  for (int i = 0; i < n; ++i)
  {
    printf("[");
    for (int j = 0; j < n-1; ++j)
      printf(" %8.4E", x[i*n+j]);
    printf(" %f ]\n", x[i*n+(n-1)]);
  }
}

} // namespace hasc

#endif // HASC_UTIL_SH
