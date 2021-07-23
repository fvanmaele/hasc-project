#include <cstdio>
#include <random>
#define HASC_SPAN_CHECKED // for debugging purposes
#include "jacobi.h"
#include "util.h"
#include "lmv/lmv_seq.h"

using namespace hasc;

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
  const int n = 12;
  const size_t nsq = n*n;
  const int k = 1;

  // Initial guess u_0
  double u_unif[nsq];
  //iota(u0, nsq, 1);
  for (size_t i = 0; i < nsq; ++i)
    u_unif[i] = unifrnd<double>(1, 10);
  double u0[nsq];
  for (size_t i = 0; i < nsq; ++i)
    u0[i] = u_unif[i];
  span<double> Su0(u0, nsq);
//  print_array_2d(u0, n);
//  printf("\n");

  double u1[nsq];
  fill(u1, nsq, 0.0);
  span<double> Su1(u1, nsq);

  // Stencil coefficients
    const size_t coeff_n = (2*k+1)*(2*k+1);
    double coeff[coeff_n];
    model_coefficients_2d<double>(coeff, 2*k+1);
    print_array_2d(coeff, 2*k+1);
    printf("\n");
//  fill(coeff, coeff_n, 0.0);

//  coeff[INDEX(0, 1, 3)] = 0.25;
//  coeff[INDEX(1, 0, 3)] = 0.25;
//  coeff[INDEX(1, 2, 3)] = 0.25;
//  coeff[INDEX(2, 1, 3)] = 0.25;
    span<const double> Scoeff(coeff, coeff_n);

  // Jacobi iteration
    int iterations = 100;
    jacobi_2d(n, k, iterations, Su0, Su1, Scoeff);
    print_array_2d(u1, n);
    printf("\n");

    for (size_t i = 0; i < nsq; ++i)
      u0[i] = u_unif[i];
    fill(u1, nsq, 0.0);
    jacobi_2d_blocked<4, 4>(n, k, iterations, Su0, Su1, Scoeff);
    print_array_2d(u1, n);
    printf("\n");

    for (size_t i = 0; i < nsq; ++i)
      u0[i] = u_unif[i];
    fill(u1, nsq, 0.0);
    jacobi_2d_blocked_openmp<4, 4>(n, k, iterations, Su0, Su1, Scoeff);
    print_array_2d(u1, n);

//  jacobi_2d_kernel(1, n-1, 1, n-1, n, 1, iterations, Su0, Su1, Scoeff);
//  print_array_2d(u1, n);
//  printf("\n");


//  for (int i = 0; i < Scoeff.ssize(); ++i)
//    printf("%f\n", coeff[i]);

//  iota(u0, nsq, 1);
//  fill(u1, nsq, 0.0);
//  jacobi_vanilla_kernel(n, iterations, u0, u1);
//  print_array_2d(u1, n);
}
