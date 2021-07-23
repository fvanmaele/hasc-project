#include <benchmark/benchmark.h>
#include <vector>
#include <random>
#include "jacobi.h"
#include "util.h"
#include "aligned_array.h"

using namespace hasc;

template <typename T>
void unifrnd_array(T* x, size_t x_len, int a, int b)
{
  for (size_t i = 0; i < x_len; ++i)
    x[i] = unifrnd<T>(a, b);
}

template <typename... Args>
void jacobi_2d_vanilla(int n, int k, int iterations, span<double> uold, span<double> unew,
                       span<const double> coeff)
{
  jacobi_2d(n, k, iterations, uold, unew, coeff);
}

// All benchmarks are identical apart from the called function, use macro
#define BM_GENERATE(func, ...)                              \
  static void BM_##func(benchmark::State& state)            \
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
    model_coefficients_2d(coeff.data(), 2*k+1);             \
    span<const double> Scoeff(coeff.data(), coeff.ssize()); \
                                                            \
    for (auto _ : state) {                                  \
      (func<__VA_ARGS__>)(n, k, t, Su0, Su1, Scoeff);       \
      benchmark::DoNotOptimize(Su0);                        \
    }                                                       \
  }

// Generate benchmark code
BM_GENERATE(jacobi_2d_vanilla)
BM_GENERATE(jacobi_2d_blocked, 32, 128)
BM_GENERATE(jacobi_2d_blocked_openmp, 32, 128)

// Set limits for n, k and iterations
// Max stencil size (box): (2*7+1)^2 = 15^2
std::vector<std::vector<int64_t>> parameter_range{
  {64, 128, 256, 512, 1024, 2048}, {1, 3, 5, 7}, {20}
};

BENCHMARK(BM_jacobi_2d_vanilla)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_jacobi_2d_blocked)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_jacobi_2d_blocked_openmp)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
