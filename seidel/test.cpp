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
      std::fprintf(stderr, "%zu (i), %f (a), %f (b)\n", i, a[i], b[i]);
      break;
    }
  return equal;
}

void seidel_4point(int n, int iterations, span<double> Su)
{
  // do iterations
  for (int it=0; it<iterations; it++)
    {
      for (int i=1; i<n-1; i++)
        for (int j=1; j<n-1; j++)
        {
          const size_t center = INDEX(i, j, n);
          Su[center] = 0.25*(Su[center-n] + Su[center-1] + Su[center+1] + Su[center+n]);
        }
    }
}

void seidel_4point_init(int n, span<double> Su)
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
  const int n_lim = 192;
  const int k_lim = 5;
  const int k_step = 2;
  const int iterations = 100;

  // Test sequential version against 4-point stencil of lecture
  std::array<const double, 9> coeff_4point = {
    0.0, 0.25, 0.0, 
    0.25, 0.0, 0.25, 
    0.0, 0.25, 0.0
  };
  span<const double> Scoeff_4point(coeff_4point);

  for (int n = 48; n <= n_lim; n*=2)
  {
    std::fprintf(stderr, "4-point, n = %d\t", n);
    const size_t u_size = (size_t)n * n;
    aligned_array<double> u_4point(u_size);
    span<double> Su_4point(u_4point);

    seidel_4point_init(n, Su_4point);
    seidel_4point(n, iterations, Su_4point);

    aligned_array<double> u(u_size);
    span<double> Su(u);

    seidel_4point_init(n, Su);
    for (int it = 1; it <= iterations; ++it)
      symmetric_seidel_2d_forward(1, n-1, 1, n-1,
                                  n, 1, Su, Scoeff_4point);

    REQUIRE(ApproxEq(Su_4point, Su));
    REQUIRE(isfinite_array(Su_4point.data(), Su_4point.size()));
    REQUIRE(isfinite_array(Su.data(), Su.size()));
    std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
  }

  // Test coefficients for boundary points
  for (int n = 48; n <= n_lim; n*=2)
  {
    const int k = 1;
    const int iterations = 5;
    std::fprintf(stderr, "Boundary values, n = %d\t", n);
    const size_t u_size = (size_t)n * n;

    aligned_array<double> u(u_size);
    span<double> Su(u);
    iota(Su.data(), Su.size(), 1);

    std::array<double, 9> coeff {
      0.0, 0.0, 0.0,
      0.0, 1.0, 1.0,
      0.0, 1.0, 1.0
    };
    span<const double> Scoeff(coeff.data(), coeff.size());

    symmetric_seidel_2d(n, k, iterations, Su, Scoeff);
    bool zero_element = false;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (i == 0 || j == 0 || i == n-1 || j == n-1)
          zero_element = Su[INDEX(i, j, n)] == 0;

    REQUIRE(zero_element == false);
    REQUIRE(isfinite_array(Su.data(), Su.size()));
    std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
  }

  // Compare parallel against sequential version
  for (int n = 48; n <= n_lim; n*=2)
  {
    const size_t u_size = (size_t)n * n;
    aligned_array<double> u_unif(u_size);
    span<double> Su_unif(u_unif);
    unifrnd<double>(1, 10, Su_unif);

    for (int k = 1; k <= k_lim; k+=k_step)
    {
      std::fprintf(stderr, "Vanilla, n = %-4d, k = %-1d\t", n, k);
      const size_t coeff_size = (2*k+1)*(2*k+1);
      aligned_array<double> coeff(coeff_size);
      
      model_coefficients_2d(coeff.data(), 2*k+1);
      span<const double> Scoeff(coeff.data(), coeff.size());

      // Check if coefficients are NaN or inf
      REQUIRE(isfinite_array(coeff.data(), coeff.size()));

      // Prepare input data
      aligned_array<double> u_seq(u_size);
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

        aligned_array<double> u(u_size);
        span<double> Su(u);
        for (size_t i = 0; i < u.size(); ++i)
          Su[i] = Su_unif[i];

        symmetric_seidel_2d_openmp(n, k, iterations, Su, Scoeff);
        REQUIRE(ApproxEq(Su, Su_seq));
        REQUIRE(NormF(u.data(), u.size()) < 1e-6);
        std::fprintf(stderr, "\033[32;1m OK \033[0m\n");
      }
    }
  }
  std::printf("Ran %zu tests successfully\n", test_n);
  return 0;
}
