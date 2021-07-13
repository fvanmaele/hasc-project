#include <benchmark/benchmark.h>
#define HASC_SPAN_CHECKED
#include "lmv_seq.h"
#include "lmv_vcl.h"
#include "aligned_array.h"

using namespace hasc;

void fill(double* x, size_t len, int a)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a;
}

void iota(double* x, size_t len, int a0)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a0++;
}

static void BM_lmv_2d(benchmark::State& state)
{
  // Perform setup here
  size_t n = state.range(0);
  size_t k = state.range(1);
  aligned_array<double, 64> u(n*n);
  iota(u.data(), u.ssize(), 1);
  span<const double> Su(u.data(), u.ssize());

  aligned_array<double, 64> lmv(n*n);
  fill(lmv.data(), lmv.ssize(), 0);
  span<double> Smean(lmv.data(), lmv.ssize());

  for (auto _ : state) {
    // This code gets timed
    lmv_2d(n, k, Su, Smean); // TODO: make k a bench parameter
  }
}

// TODO: templated benchmark
template <int MI, int MJ>
static void BM_lmv_2d_blocked(benchmark::State& state)
{
  // Perform setup here
  size_t n = state.range(0);
  size_t k = state.range(1);
  aligned_array<double, 64> u(n*n);
  iota(u.data(), u.ssize(), 1);
  span<const double> Su(u.data(), u.ssize());

  aligned_array<double, 64> lmv(n*n);
  fill(lmv.data(), lmv.ssize(), 0);
  span<double> Smean(lmv.data(), lmv.ssize());

  for (auto _ : state) {
    // This code gets timed
    lmv_2d_blocked<MI, MJ>(n, k, Su, Smean); // TODO: make k a bench parameter
  }
}

template <int W>
static void BM_lmv_2d_vectorized(benchmark::State& state)
{
  // Perform setup here
  size_t n = state.range(0);
  size_t k = state.range(1);
  aligned_array<double, 64> u(n*n);
  iota(u.data(), u.ssize(), 1);
  span<const double> Su(u.data(), u.ssize());

  aligned_array<double, 64> lmv(n*n);
  fill(lmv.data(), lmv.ssize(), 0);
  span<double> Smean(lmv.data(), lmv.ssize());

  for (auto _ : state) {
    // This code gets timed
    lmv_2d_vectorized<W>(n, k, Su, Smean); // TODO: make k a bench parameter
  }
}

template <int MI, int MJ, int W>
static void BM_lmv_2d_vectorized_blocked(benchmark::State& state)
{
  // Perform setup here
  size_t n = state.range(0);
  size_t k = state.range(1);
  aligned_array<double, 64> u(n*n);
  iota(u.data(), u.ssize(), 1);
  span<const double> Su(u.data(), u.ssize());

  aligned_array<double, 64> lmv(n*n);
  fill(lmv.data(), lmv.ssize(), 0);
  span<double> Smean(lmv.data(), lmv.ssize());

  for (auto _ : state) {
    // This code gets timed
    lmv_2d_blocked_vectorized<MI, MJ, W>(n, k, Su, Smean); // TODO: make k a bench parameter
  }
}

BENCHMARK(BM_lmv_2d)
  ->ArgsProduct({{64, 128, 256, 512, 1024, 2048, 4096, 8192}, {1, 3, 5}})
  ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_lmv_2d_blocked, 4, 32)
  ->ArgsProduct({{64, 128, 256, 512, 1024, 2048, 4096, 8192}, {1, 3, 5}})
  ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_lmv_2d_vectorized, 4)
  ->ArgsProduct({{64, 128, 256, 512, 1024, 2048, 4096, 8192}, {1, 3, 5}})
  ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_lmv_2d_vectorized_blocked, 4, 32, 4)
  ->ArgsProduct({{64, 128, 256, 512, 1024, 2048, 4096, 8192}, {1, 3, 5}})
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
