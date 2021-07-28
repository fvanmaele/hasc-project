#include <cstdio>
#include <cfloat>
#include <cstdlib>
#include <random>

#define HASC_SPAN_CHECKED
#include "span.h"
#include "seidel.h"
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
  if (a.size() != b.size())
    return false;
  bool equal = true;

  for (size_t i = 0; i < a.size(); ++i)
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
  std::uniform_real_distribution<T> distr(a, b);
  return distr(generator);
}

template <typename T>
void unifrnd(const int a, const int b, span<double>& u)
{
  for (size_t i = 0; i < u.size(); ++i)
    u[i] = unifrnd<T>(a, b);
}

int main()
{
  const int n_lim = 768;
  const int k_lim = 5;
  const int k_step = 2;
  const int iterations = 20;

  // Compare parallel against sequential version
  for (int n = 48; n <= n_lim; n*=2)
  {
    aligned_array<double> u_unif(n*n);
    span<double> Su_unif(u_unif);
    unifrnd<double>(1, 10, Su_unif);

    for (int k = 1; k <= k_lim; k+=k_step)
    {
      std::fprintf(stderr, "Vanilla, n = %-4d, k = %-1d\t", n, k);
      aligned_array<double> coeff((2*k+1)*(2*k+1));
      model_coefficients_2d(coeff.data(), 2*k+1);
      span<const double> Scoeff(coeff.data(), coeff.size());

      // Check if coefficients are NaN or inf
      REQUIRE(isfinite_array(coeff.data(), coeff.size()));

      // Prepare input data
      aligned_array<double> u_seq(n*n);
      span<double> Su_seq(u_seq);
      for (size_t i = 0; i < Su_seq.size(); ++i)
        Su_seq[i] = Su_unif[i];

      // Test convergence of sequential version to zero
      symmetric_seidel_2d(n, k, iterations, Su_seq, Scoeff);
      REQUIRE(isfinite_array(u_seq.data(), u_seq.size()));
      REQUIRE(NormF(u_seq.data(), u_seq.size()) < 1e-6);
      std::fprintf(stderr, "\033[32;1m OK \033[0m\n");

      { // Parallel version
        std::fprintf(stderr, "OpenMP,  n = %-4d, k = %-1d\t", n, k);

        aligned_array<double> u(n*n);
        span<double> Su(u);
        for (size_t i = 0; i < Su.size(); ++i)
          Su[i] = Su_unif[i];

        symmetric_seidel_2d_openmp(n, k, iterations, Su, Scoeff);
        REQUIRE(ApproxEq(Su, Su_seq));
        REQUIRE(NormF(u.data(), u.size()) < 1e-6);
        std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
      }
    }
  }
  return 0;
}
