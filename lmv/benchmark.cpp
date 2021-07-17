#include <benchmark/benchmark.h>
#include <vector>
#include "lmv_seq.h"
#include "lmv_vcl.h"
#include "aligned_array.h"

using namespace hasc;

// Helper functions
void fill(double* x, size_t len, int a)
{
  memset(x, a, sizeof(double)*len);
}
void iota(double* x, size_t len, int a0)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a0++;
}
template <typename... Args>
void lmv_2d_vanilla(int n, int k, span<const double> u, span<double> mean)
{
  lmv_2d(n, k, u, mean);
}

// All benchmarks are identical apart from the called function, use macro
#define BM_GENERATE(func, ...) \
  static void BM_##func(benchmark::State& state)  \
  {                                               \
    size_t n = state.range(0);                    \
    size_t k = state.range(1);                    \
    aligned_array<double, 64> u(n*n);             \
    iota(u.data(), u.size(), 1);                  \
                                                  \
    span<const double> Su(u.data(), u.size());    \
    aligned_array<double, 64> lmv(n*n);           \
    fill(lmv.data(), lmv.size(), 0);              \
    span<double> Smean(lmv.data(), lmv.size());   \
                                                  \
    for (auto _ : state) {                        \
      (func<__VA_ARGS__>)(n, k, Su, Smean);       \
      benchmark::DoNotOptimize(Smean);            \
    }                                             \
  }

// Generate benchmark code
BM_GENERATE(lmv_2d_vanilla)
BM_GENERATE(lmv_2d_blocked, 4, 64)
BM_GENERATE(lmv_2d_blocked_openmp, 4, 64)
BM_GENERATE(lmv_2d_vectorized, 4)
BM_GENERATE(lmv_2d_vectorized_buffered, 4)
BM_GENERATE(lmv_2d_vectorized_blocked, 4, 64, 4)
BM_GENERATE(lmv_2d_vectorized_blocked_openmp, 4, 64, 4)
BM_GENERATE(lmv_2d_vectorized_buffered_blocked, 4, 64, 4)
BM_GENERATE(lmv_2d_vectorized_buffered_blocked_openmp, 4, 64, 4)

// Set limits of n and k
std::vector<std::vector<int64_t>> parameter_range{
  {64, 128, 256, 512, 1024, 2048, 4096, 8192}, {1, 3, 5, 7}
};

// Register benchmarks
BENCHMARK(BM_lmv_2d_vanilla)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_blocked)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_blocked_openmp)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_vectorized)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_vectorized_blocked)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_vectorized_blocked_openmp)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_vectorized_buffered)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_vectorized_buffered_blocked)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_lmv_2d_vectorized_buffered_blocked_openmp)
  ->ArgsProduct(parameter_range)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
