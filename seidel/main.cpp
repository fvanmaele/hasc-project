#include <cstdio>
#include <random>
#define HASC_SPAN_CHECKED // for debugging purposes
#include "seidel.h"
#include "seidel_par.h"
#include "util.h"

using namespace hasc;

template <typename T>
T unifrnd(const int a, const int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<T> distr(a, b);
  return distr(generator);
}

// Frobenius norm
double NormF(span<double> A)
{
  double sum = 0;
  for (auto c : A)
    sum += c*c;
  return sqrt(sum);
}

int main()
{
  const int n = 7;
  const size_t nsq = n*n;
  const int k = 2;
  const int iterations = 10;

  // Initial guess u_0
  double u_unif[nsq];
  //iota(u0, nsq, 1);
  for (size_t i = 0; i < nsq; ++i)
    u_unif[i] = unifrnd<double>(1, 10);

  double u[nsq];
  for (size_t i = 0; i < nsq; ++i)
    u[i] = u_unif[i];
  span<double> Su(u, nsq);

  // Stencil coefficients
  const size_t coeff_n = (2*k+1)*(2*k+1);
  double coeff[coeff_n];
  model_coefficients_2d(coeff, 2*k+1);
  print_array_2d(coeff, 2*k+1);
  printf("\n");

  span<const double> Scoeff(coeff, coeff_n);

  // Symmetric Gauss-Seidel iteration
  symmetric_seidel_2d(n, k, iterations, Su, Scoeff);
  //print_array_2d(u, n);
  double norm_vanilla = NormF(Su);

  for (size_t i = 0; i < nsq; ++i)
    u[i] = u_unif[i]; // reinitialize
  //symmetric_seidel_2d_blocked_inexact<4, 4>(n, k, iterations, Su, Scoeff);
  symmetric_seidel_2d_openmp(n, k, iterations, Su, Scoeff);
  print_array_2d(u, n);
  printf("Norm: %E\n", norm_vanilla);
  printf("Norm: %E\n", NormF(Su));
  //printf("\n");
}
