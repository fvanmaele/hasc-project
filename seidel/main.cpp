#include <cstdio>
#include <cstring>
#include <random>
#include <omp.h>
#include "seidel.h"

using namespace hasc;

template <typename T>
T unifrnd(const int a, const int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<T> distr(a, b);
  return distr(generator);
}

void usage(const char* progn)
{
  std::fprintf(stderr, "%s <size> <radius> <iterations> [seq|omp]\n", progn);
  std::exit(1);
}

int main(int argc, char* argv[])
{
  int n = 1024;
  int k = 1;
  int iterations = 10;
  char mode[16];

  // Retrieve arguments
  if (argc < 5)
    usage(argv[0]);
  else
  {
    if (std::sscanf(argv[1], "%d", &n) != 1)
      usage(argv[0]);
    if (std::sscanf(argv[2], "%d", &k) != 1)
      usage(argv[0]);
    if (std::sscanf(argv[3], "%d", &iterations) != 1)
      usage(argv[0]);
    if (std::sscanf(argv[4], "%4s", mode) != 1)
      usage(argv[0]);
  }

  // Generate input/output arrays
  const size_t u_size = (size_t)n * n;
  aligned_array<double, 64> u(u_size);
  span<double> Su(u);
  for (size_t i = 0; i < Su.size(); ++i)
    Su[i] = unifrnd<double>(1, 10);

  // Generate coefficients
  aligned_array<double, 64> coeff((2*k+1)*(2*k+1));
  model_coefficients_2d(coeff.data(), 2*k+1);
  span<const double> Scoeff(coeff.data(), coeff.size());

  // Symmetric Gauss-Seidel iteration
  const char* mode_cmp = mode;
  if(std::strncmp(mode_cmp, "seq", 3) == 0)
    symmetric_seidel_2d(n, k, iterations, Su, Scoeff);
  else if (std::strncmp(mode_cmp, "omp", 3) == 0)
    symmetric_seidel_2d_openmp(n, k, iterations, Su, Scoeff);
  else
    usage(argv[0]);

  printf("Frobenius norm: %E\n", NormF(u.data(), u.size()));

#ifdef LMV_PRINT_VALUES
  print_array_2d(u1.data(), n);
#endif
}
