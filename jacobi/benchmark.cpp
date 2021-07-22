#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "jacobi.h"
#include "aligned_array.h"

using namespace hasc;

template <typename T>
T unifrnd(int a, int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<double> distr(a, b);
  return distr(generator);
}

template <typename T>
void unifrnd_array(T* x, size_t x_len, int a, int b)
{
  for (size_t i = 0; i < x_len; ++i)
    x[i] = unifrnd<T>(a, b);
}

template <typename T>
void model_coefficients_2d(T* coeff, int coeff_n, int k)
{
  for (int i = 0; i < coeff_n; ++i)
    for (int j = 0; j < coeff_n; ++j)
      i == k && j == k ? coeff[INDEX(i, j, coeff_n)] = 50
                       : coeff[INDEX(i, j, coeff_n)] = 1./MAX(i, j);
}

template <typename... Args>
void jacobi_2d_vanilla(int n, int k, int iterations, span<double> uold, span<double> unew,
                       span<const double> coeff)
{
  jacobi_2d(n, k, iterations, uold, unew, coeff);
}

// All benchmarks are identical apart from the called function, use macro
#define BM_GENERATE(func, test_case_name, ...)                      \
  static void BM_##func ## _ ##test_case_name (benchmark::State& state)     \
  {                                                         \
    size_t n = state.range(0);                              \
    size_t k = state.range(1);                              \
    size_t t = state.range(2);                              \
                                                            \
    aligned_array<double, 64> u0(n*n);                      \
    aligned_array<double, 64> u1(n*n);                      \
    unifrnd_array(u0.data(), u0.ssize(), -1, 1);            \
    span<double> Su0(u0);                                   \
    span<double> Su1(u1);                                   \
                                                            \
    aligned_array<double, 64> coeff((2*k+1)*(2*k+1));       \
    model_coefficients_2d(coeff.data(), 2*k+1, k);          \
    span<const double> Scoeff(coeff.data(), coeff.ssize()); \
                                                            \
    for (auto _ : state) {                                  \
      (func<__VA_ARGS__>)(n, k, t, Su0, Su1, Scoeff);       \
      benchmark::DoNotOptimize(Su0);                        \
    }                                                       \
  }

// Generate benchmark code
BM_GENERATE(jacobi_2d_vanilla,)
BM_GENERATE(jacobi_2d_blocked, sizeA, 4, 64)
BM_GENERATE(jacobi_2d_blocked, sizeB, 32, 256)
BM_GENERATE(jacobi_2d_blocked_openmp, sizeA, 4, 64)
BM_GENERATE(jacobi_2d_blocked_openmp, sizeB, 32, 256)

// Set limits for n, k and iterations
std::vector<std::vector<int64_t>> parameter_range{
  {64, 128, 256, 512, 1024, 2048}, {1, 3, 5, 7}, {20}
};

BENCHMARK(BM_jacobi_2d_vanilla_)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_jacobi_2d_blocked_sizeA)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_jacobi_2d_blocked_sizeB)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_jacobi_2d_blocked_openmp_sizeA)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_jacobi_2d_blocked_openmp_sizeB)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
