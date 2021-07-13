#include <cstdio>
#define HASC_SPAN_CHECKED // for debugging purposes
#include "lmv_seq.h"
#include "lmv_vcl.h"
#include "aligned_array.h"

using namespace hasc;

void print_2d_array(const double* x, int n)
{
  for (int i = 0; i < n; ++i)
  {
    printf("[");
    for (int j = 0; j < n-1; ++j)
      printf(" %8.3f", x[i*n+j]);
    printf(" %f ]\n", x[i*n+(n-1)]);
  }
}

void fill(double* x, size_t len, int a)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a;
}

void iota(double* x, size_t len, int a0 = 1)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a0++;
}

int main() {
  const int n = 16;
  const size_t nsq = n*n;
  const int k = 3;
  double u[nsq];
  iota(u, nsq, 1);

  span<const double> Su(u, nsq);
  print_2d_array(u, n);
  printf("\n");

  double mean[nsq];
  span<double> Smean(mean, nsq);
  fill(mean, nsq, 0);

  // Vanilla version
  lmv_2d(n, k, Su, Smean);
  print_2d_array(mean, n);
  printf("\n");

  // Blocked version
  fill(mean, nsq, 0);
  lmv_2d_blocked<4, 64>(n, k, Su, Smean);
  print_2d_array(mean, n);
  printf("\n");

  // Vectorized version
  fill(mean, nsq, 0);
  lmv_2d_vectorized<4>(n, k, Su, Smean);
  print_2d_array(mean, n);
  printf("\n");

  // Block vectorized version
  fill(mean, nsq, 0);
  lmv_2d_blocked_vectorized<4, 64, 4>(n, k, Su, Smean);
  print_2d_array(mean, n);
}
