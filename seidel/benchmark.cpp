#include <benchmark/benchmark.h>
#include <vector>
#include <random>
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

template <typename T>
void unifrnd_array(T* x, size_t x_len, int a, int b)
{
  for (size_t i = 0; i < x_len; ++i)
    x[i] = unifrnd<T>(a, b);
}

#define BM_GENERATE(func, ...)                              \
  static void BM_##func(benchmark::State& state)            \
  {                                                         \
    int64_t n = state.range(0);                              \
    int64_t k = state.range(1);                              \
    int64_t t = state.range(2);                              \
                                                            \
    aligned_array<double, 64> u(n*n);                       \
    unifrnd_array(u.data(), u.size(), -1, 1);               \
    span<double> Su(u);                                     \
                                                            \
    aligned_array<double, 64> coeff((2*k+1)*(2*k+1));       \
    model_coefficients_2d(coeff.data(), 2*k+1);             \
    span<const double> Scoeff(coeff.data(), coeff.size());  \
                                                            \
    for (auto _ : state) {                                  \
      (func)(n, k, t, Su, Scoeff);                          \
      benchmark::DoNotOptimize(Su);                         \
    }                                                       \
  }

BM_GENERATE(symmetric_seidel_2d)
BM_GENERATE(symmetric_seidel_2d_openmp)

std::vector<std::vector<int64_t>> parameter_range{
  {48, 96, 192, 384, 768, 1536, 3072}, {1, 3, 5, 7}, {20}
};

BENCHMARK(BM_symmetric_seidel_2d)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_symmetric_seidel_2d_openmp)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
