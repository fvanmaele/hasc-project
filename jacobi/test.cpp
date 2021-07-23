#include <cstdio>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <random>

#define HASC_SPAN_CHECKED
#include "span.h"
#include "jacobi.h"
#include "util.h"

#define REQUIRE(expr) Require(expr, __LINE__)

using namespace hasc;

static size_t test_n;

void Require(bool expr, int line)
{
    if (!expr) {
        std::fprintf(stderr, "test failure at line number %d\n", line);
        std::abort();
    }
    test_n++;
}

bool ApproxEq(span<double> a, span<double> b)
{
  if (a.ssize() != b.ssize())
    return false;
  bool equal = true;

  for (ptrdiff_t i = 0; i < a.ssize(); ++i)
    if (std::abs(a[i]-b[i]) > DBL_EPSILON)
    {
      equal = false;
      std::fprintf(stderr, "%ld (i), %f (a), %f (b)\n", i, a[i], b[i]);
      break;
    }
  return equal;
}

// Frobenius norm
double NormF(span<double> A)
{
  double sum = 0;
  for (auto c : A)
    sum += c*c;
  return sqrt(sum);
}

void jacobi_4point(int n, int iterations, span<double> uold, span<double> unew)
{
  // do iterations
  for (int it=0; it<iterations; it++)
    {
      for (int i=1; i<n-1; i++)
        for (int j=1; j<n-1; j++)
        {
          const size_t center = INDEX(i, j, n);
          unew[center] = 0.25*(
                uold[center-n] + uold[center-1] + uold[center+1] + uold[center+n]);
        }
      using std::swap;
      swap(uold, unew);
    }
}

void jacobi_4point_init(int n, span<double> Su)
{
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
    {
      if (i > 0 && i < n-1 && j > 0 && j < n-1)
        Su[INDEX(i, j, n)] = (double)(i+j)/n;
      else
        Su[INDEX(i, j, n)] = 0.0;
    }
}

template <typename T>
void unifrnd(const int a, const int b, span<double>& u)
{
  for (ptrdiff_t i = 0; i < u.ssize(); ++i)
    u[i] = unifrnd<T>(a, b);
}

int main()
{
  const int n_lim = 256;
  const int k_lim = 5;
  const int iterations = 100;

  // Test sequential version against 4-point stencil of lecture
  std::array<const double, 9> coeff_4point = {
    0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0
  };
  span<const double> Scoeff_4point(coeff_4point);

  for (int n = 64; n <= n_lim; n*=2)
  {
    std::fprintf(stderr, "4-point, n = %d\t", n);
    aligned_array<double, 64> u0_4point(n*n);
    span<double> Su0_4point(u0_4point);
    aligned_array<double, 64> u1_4point(n*n);
    span<double> Su1_4point(u1_4point);

    jacobi_4point_init(n, Su0_4point);
    jacobi_4point(n, iterations, Su0_4point, Su1_4point);

    aligned_array<double, 64> u0(n*n);
    span<double> Su0(u0);
    aligned_array<double, 64> u1(n*n);
    span<double> Su1(u1);

    jacobi_4point_init(n, Su0);
    for (int it = 1; it <= iterations; ++it)
    {
      jacobi_2d_kernel(1, n-1, 1, n-1,
                       n, 1, Su0, Su1, Scoeff_4point);
      using std::swap;
      swap(Su0, Su1);
    }

    REQUIRE(ApproxEq(Su0_4point, Su0));
    REQUIRE(ApproxEq(Su1_4point, Su1));
    std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
  }

  // Test parallel implementation against sequential version
  for (int n = 64; n <= n_lim; n*=2)
  {
    aligned_array<double> u_unif(n*n);
    span<double> Su_unif(u_unif); // TODO
    unifrnd<double>(1, 10, Su_unif);

    for (int k = 1; k <= k_lim; ++k)
    {
      aligned_array<double> coeff((2*k+1)*(2*k+1));
      model_coefficients_2d<double>(coeff.data(), 2*k+1);
      span<const double> Scoeff(coeff.data(), coeff.ssize());

      aligned_array<double> u0_seq(n*n);
      span<double> Su0_seq(u0_seq);
      for (ptrdiff_t i = 0; i < Su0_seq.ssize(); ++i)
        Su0_seq[i] = Su_unif[i];

      aligned_array<double> u1_seq(n*n);
      span<double> Su1_seq(u1_seq);      
      jacobi_2d(n, k, iterations, Su0_seq, Su1_seq, Scoeff);

      // Test convergence of sequential version to zero
      REQUIRE(NormF(Su1_seq) < 1e-6);

      { // Blocked version
        std::fprintf(stderr, "Blocked, n = %-4d, k = %-1d\t", n, k);

        aligned_array<double> u0(n*n);
        span<double> Su0(u0);
        for (ptrdiff_t i = 0; i < Su0.ssize(); ++i)
          Su0[i] = Su_unif[i];

        aligned_array<double> u1(n*n);
        span<double> Su1(u1);

        jacobi_2d_blocked<32, 128>(n, k, iterations, Su0, Su1, Scoeff);
        REQUIRE(ApproxEq(Su1, Su1_seq));
        REQUIRE(ApproxEq(Su0, Su0_seq));
        REQUIRE(NormF(Su1_seq) < 1e-6);
        std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
      }

      { // OpenMP version
        std::fprintf(stderr, "OpenMP, n = %-4d, k = %-1d\t", n, k);

        aligned_array<double> u0(n*n);
        span<double> Su0(u0);
        for (ptrdiff_t i = 0; i < Su0.ssize(); ++i)
          Su0[i] = Su_unif[i];

        aligned_array<double> u1(n*n);
        span<double> Su1(u1);

        jacobi_2d_blocked_openmp<32, 128>(n, k, iterations, Su0, Su1, Scoeff);
        REQUIRE(ApproxEq(Su1, Su1_seq));
        REQUIRE(ApproxEq(Su0, Su0_seq));
        REQUIRE(NormF(Su1_seq) < 1e-6);
        std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
      }

      { // Vectorized version

      }

      { // OpenMP, vectorized version

      }
    }
  }

  return 0;
}
