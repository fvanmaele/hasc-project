#include <cstdio>
#include <cstring>
#include <random>
#include <omp.h>
#include "jacobi.h"
#include "util.h"
#define BLOCK_SIZE_I 64
#define BLOCK_SIZE_J 64

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
  aligned_array<double, 64> u0(n*n);
  span<double> Su0(u0);
  for (size_t i = 0; i < Su0.size(); ++i)
    Su0[i] = unifrnd<double>(1, 10);

  aligned_array<double, 64> u1(n*n);
  span<double> Su1(u1);
  for (size_t i = 0; i < Su1.size(); ++i)
    Su1[i] = 0;

  // Generate coefficients
  aligned_array<double, 64> coeff((2*k+1)*(2*k+1));
  model_coefficients_2d(coeff.data(), 2*k+1);
  span<const double> Scoeff(coeff.data(), coeff.size());

  // Jacobi iteration
  const char* mode_cmp = mode;
  if(std::strncmp(mode_cmp, "seq", 3) == 0)
    jacobi_2d(n, k, iterations, Su0, Su1, Scoeff);
  else if (std::strncmp(mode_cmp, "omp", 3) == 0)
    jacobi_2d_blocked_openmp<BLOCK_SIZE_I, BLOCK_SIZE_J>(n, k, iterations, Su0, Su1, Scoeff);
  else
    usage(argv[0]);

  printf("Frobenius norm: %E\n", NormF(u1.data(), u1.size()));

#ifdef LMV_PRINT_VALUES
  print_array_2d(u1.data(), n);
#endif
}
