# Hardware Aware Scientific Computing (SS 2021) - Exam Sheet
* Author: Ferdinand Vanmaele
* Submission Date: 2021-07-30, 16:00

## Building the code

All programs can be compiled with cmake:
```
  $ mkdir build; cd build
  $ cmake -DCMAKE_BUILD_TYPE=Release ..
  $ make -j$(nproc)
```
Requirements:
- cmake >=3.12
- C++17 compiler (tested with GCC 10.2 and MSVC 2019)

The `benchmark` and `vectorclass` libraries are included as submodules. A
system version of `benchmark` can be used to reduce compile times by setting
the `OPT_USE_BUNDLED_BENCHMARK` option:
```
  $ cmake -DOPT_USE_BUNDLED_BENCHMARK=OFF ..
```

## Running the code

The local mean value, Jacobi and symmetric Gauss-Seidel implementations are in
the `lmv`, `jacobi`, and `seidel` subdirectories. Each subdirectory contains a
benchmark, tests, and simple command-line utility.

Tests can be run with `ctest`:
```
  $ cd build
  $ ctest
```
**WARNING:** `OPT_FAST_MATH` must be set to `OFF` to run the tests!

Benchmarks can be run directly to stdout, or with their results saved to a `.csv` file:
```
   $ ./jacobi_benchmark --benchmark_out_format=csv --benchmark_out=jacobi.csv
```
Note that the resulting `.csv` has an extraneous header which should be removed before processing.

The command-line utilities have an interface similar to those in the hasc
repository, and print the (Frobenius) norm where applicable.

