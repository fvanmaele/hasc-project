#include <cstdio>
#include <cstring>
#include "lmv_seq.h"
#include "lmv_vcl.h"
#include "util.h"
#define SIMD_WIDTH 4

using namespace hasc;

void usage(const char* progn)
{
  std::fprintf(stderr, "%s <size> <radius> [seq|vcl1|vcl2]\n", progn);
  std::exit(1);
}

int main(int argc, char* argv[])
{
  int n = 1024;
  int k = 1;
  char mode[16];

  // Retrieve arguments
  if (argc < 4)
    usage(argv[0]);
  else
  {
    if (std::sscanf(argv[1], "%d", &n) != 1)
      usage(argv[0]);
    if (std::sscanf(argv[2], "%d", &k) != 1)
      usage(argv[0]);
    if (std::sscanf(argv[3], "%4s", mode) != 1)
      usage(argv[0]);
  }

  // Generate input/output arrays
  aligned_array<double, 64> u(n*n);
  iota(u.data(), u.size(), 1);
  span<const double> Su(u.data(), u.size());

  aligned_array<double, 64> mean(n*n);
  fill(mean.data(), mean.size(), 0);
  span<double> Smean(mean);

  // Compute local mean value
  const char* mode_cmp = mode;
  if(std::strncmp(mode_cmp, "seq", 3) == 0)
    lmv_2d(n, k, Su, Smean);
  else if (std::strncmp(mode_cmp, "vcl1", 4) == 0)
    lmv_2d_vectorized<SIMD_WIDTH>(n, k, Su, Smean);
  else if (std::strncmp(mode_cmp, "vcl2", 4) == 0)
    lmv_2d_vectorized_buffered<SIMD_WIDTH>(n, k, Su, Smean);
  else
    usage(argv[0]);

#ifdef LMV_PRINT_VALUES
  print_array_2d(u.data(), n);
#endif
}
