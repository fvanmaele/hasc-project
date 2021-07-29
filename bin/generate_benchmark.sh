#!/bin/bash
MI_start=16
MI_end=512
MJ_start=16
MJ_end=512
MI_step=16
MJ_step=16
filename=benchmark_blocked.cpp

{ cat <<EOF
// created by generate_benchmark.sh
#include <benchmark/benchmark.h>
#include <vector>
#include "lmv_seq.h"
#include "lmv_vcl.h"

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

template <int MI, int MJ>
static void BM_lmv_2d_blocked(benchmark::State& state)
{
	size_t n = state.range(0);
	size_t k = state.range(1);
	aligned_array<double, 64> u(n*n);
	iota(u.data(), u.size(), 1);

	span<const double> Su(u.data(), u.size());
	aligned_array<double, 64> lmv(n*n);
	fill(lmv.data(), lmv.size(), 0);
	span<double> Smean(lmv.data(), lmv.size());

	for (auto _ : state) {
		lmv_2d_blocked<MI, MJ>(n, k, Su, Smean);
		benchmark::DoNotOptimize(Smean);
	}
}
EOF

for (( MI = MI_start; MI <= MI_end; MI+=MI_step )); do
    for (( MJ = MJ_start; MJ <= MJ_end; MJ+=MJ_step )); do
        cat <<EOF
BENCHMARK_TEMPLATE(BM_lmv_2d_blocked, $MI, $MJ)
    ->Args({2048, 5})->Unit(benchmark::kMillisecond);
EOF
	done
done
cat <<EOF
BENCHMARK_MAIN();
EOF
} >"$filename"
