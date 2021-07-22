#include <cstdio>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <random>

#define HASC_SPAN_CHECKED
#include "span.h"
#include "jacobi.h"

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

bool ApproxEq (span<double> a, span<double> b)
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

template <typename T>
T unifrnd(const int a, const int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<double> distr(a, b);
  return distr(generator);
}

template <typename T>
void unifrnd(const int a, const int b, span<double>& u)
{
  for (ptrdiff_t i = 0; i < u.ssize(); ++i)
    u[i] = unifrnd<T>(a, b);
}

// TODO: move to jacobi_util.h
template <typename T>
void model_coefficients_2d(T* coeff, int coeff_n, int k)
{
  for (int i = 0; i < coeff_n; ++i)
    for (int j = 0; j < coeff_n; ++j)
      i == k && j == k ? coeff[INDEX(i, j, coeff_n)] = 50
                       : coeff[INDEX(i, j, coeff_n)] = 1./MAX(i, j);
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

void fill(span<double> Su, double a)
{
  for (ptrdiff_t k = 0; k < Su.ssize(); ++k)
    Su[k] = a;
}

int main()
{
  const int n_lim = 1024;
  const int k_lim = 5;
  const int iterations = 20;

  // Test sequential version against 4-point stencil of lecture
  std::array<const double, 9> coeff_4point = {
    0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0
  };
  span<const double> Scoeff_4point(coeff_4point);

  for (int n = 64; n <= n_lim; n*=2)
  {
    std::fprintf(stderr, "4-point stencil, n = %d\t", n);
    aligned_array<double, 64> u0(n*n);
    span<double> Su0(u0);
    aligned_array<double, 64> u1(n*n);
    span<double> Su1(u1);

    jacobi_4point_init(n, Su0);
    jacobi_4point(n, iterations, Su0, Su1);

    aligned_array<double, 64> u2(n*n);
    span<double> Su2(u2);
    aligned_array<double, 64> u3(n*n);
    span<double> Su3(u3);

    jacobi_4point_init(n, Su2);
    jacobi_2d_kernel(1, n-1, 1, n-1, n, 1, iterations, Su2, Su3, Scoeff_4point);

    REQUIRE(ApproxEq(Su1, Su3));
    fprintf(stderr, "\033[32;1m OK \033[0m\n");
  }

  // Test parallel implementation against sequential version
  for (int n = 64; n <= n_lim; n*=2)
  {
    std::fprintf(stderr, "n = %-4d\t", n);
    aligned_array<double> u_unif(n*n);
    span<double> Su_unif(u_unif);
    unifrnd<double>(-1, 1, Su_unif);

    for (int k = 1; k <= k_lim; ++k)
    {
      aligned_array<double> coeff((2*k+1)*(2*k+1));
      model_coefficients_2d(coeff.data(), 2*k+1, k);
      span<const double> Scoeff(coeff.data(), coeff.ssize());

      aligned_array<double> u0_seq(n*n); // TODO
      span<double> Su0_seq(u0_seq);

      aligned_array<double> u1_seq(n*n);
      span<double> Su1_seq(u1_seq);

      jacobi_2d(n, k, iterations, Su0_seq, Su1_seq, Scoeff);

      { // Blocked version
        aligned_array<double> u0(n*n);
        span<double> Su0(u0);
        // TODO
      }

      { // OpenMP version

      }
    }
  }

  return 0;
}
