#include <cstdio>
#include <random>
#define HASC_SPAN_CHECKED // for debugging purposes
#include "jacobi.h"
#include "lmv/lmv_seq.h"

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

template <typename T>
void fill(T* x, size_t len, T a)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a;
}

template <typename T>
void iota(T* x, size_t len, int a0 = 1)
{
  for (size_t i = 0; i < len; ++i)
    x[i] = a0++;
}

template <typename T>
T unifrnd(const int a, const int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<double> distr(a, b);
  return distr(generator);
}

void jacobi_vanilla_kernel (int n, int iterations, double* uold, double* unew)
{
  // do iterations
  for (int i=0; i<iterations; i++)
    {
      for (int i1=1; i1<n-1; i1++)
        for (int i0=1; i0<n-1; i0++)
          unew[i1*n+i0] = 0.25*(uold[i1*n+i0-n]+uold[i1*n+i0-1]
                                +uold[i1*n+i0+1]+uold[i1*n+i0+n]);
      std::swap(uold,unew);
    }
}

int main()
{
  const int n = 8;
  const size_t nsq = n*n;
  const int k = 1;

  // Initial guess u_0
  double u0[nsq];
  iota(u0, nsq, 1);
//  for (size_t i = 0; i < nsq; ++i)
//    u0[i] = unifrnd<double>(-1, 1);
  span<double> Su0(u0, nsq);
  print_2d_array(u0, n);
  printf("\n");

  double u1[nsq];
  fill(u1, nsq, 0.0);
  span<double> Su1(u1, nsq);

  // Stencil coefficients (constant)
  const size_t coeff_n = (2*k+1)*(2*k+1);
  double coeff[coeff_n];
  fill(coeff, coeff_n, 0.0);

  coeff[INDEX(0, 1, 3)] = 0.25;
  coeff[INDEX(1, 0, 3)] = 0.25;
  coeff[INDEX(1, 2, 3)] = 0.25;
  coeff[INDEX(2, 1, 3)] = 0.25;
  span<const double> Scoeff(coeff, coeff_n);

  // Jacobi iteration
  int iterations = 50;
  jacobi_2d_kernel(1, n-1, 1, n-1, n, 1, iterations, Su0, Su1, Scoeff);
  print_2d_array(u1, n);
  printf("\n");


  for (int i = 0; i < Scoeff.ssize(); ++i)
    printf("%f\n", coeff[i]);

  iota(u0, nsq, 1);
  fill(u1, nsq, 0.0);
  jacobi_vanilla_kernel(n, iterations, u0, u1);
  print_2d_array(u1, n);
}
