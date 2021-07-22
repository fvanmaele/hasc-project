#include <cstdio>
#include <cfloat>
#include <memory>
#include "lmv_seq.h"
#include "lmv_vcl.h"
#include "aligned_array.h"

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

int main()
{
  const int lim_n = 4096;
  const int lim_k = 5;
  std::fprintf(stderr, "Ranges: n = 64 .. %d, k = 1 .. %d\n", lim_n, lim_k);

  for (int n = 64; n <= lim_n; n*=2)
  {
    const size_t u_size = n*n;
    std::unique_ptr<double> u(new double[u_size]);
    for (size_t i = 0; i < u_size; ++i) {
      u.get()[i] = i+1;
    }
    span<const double> Su(u.get(), u_size);

    std::fprintf(stderr, "n = %-4d\t", n);
    for (int k = 1; k <= lim_k; ++k)
    {
      // Sequential version as baseline comparison
      std::unique_ptr<double> mean_seq(new double[u_size]);
      span<double> Smean_seq(mean_seq.get(), u_size);
      lmv_2d(n, k, Su, Smean_seq);

      { // Blocked version
        std::unique_ptr<double> mean(new double[u_size]);
        span<double> Smean(mean.get(), u_size);

        lmv_2d_blocked<4, 64>(n, k, Su, Smean);
        REQUIRE(ApproxEq(Smean, Smean_seq));
      }

      { // Vectorized version
        std::unique_ptr<double> mean(new double[u_size]);
        span<double> Smean(mean.get(), u_size);

        lmv_2d_vectorized<4>(n, k, Su, Smean);
        REQUIRE(ApproxEq(Smean, Smean_seq));
      }

      { // Buffered vectorized version
        std::unique_ptr<double> mean(new double[u_size]);
        span<double> Smean(mean.get(), u_size);

        lmv_2d_vectorized_buffered<4>(n, k, Su, Smean);
        REQUIRE(ApproxEq(Smean, Smean_seq));
      }

      { // Blocked vectorized version
        std::unique_ptr<double> mean(new double[u_size]);
        span<double> Smean(mean.get(), u_size);

        lmv_2d_vectorized_blocked<4, 64, 4>(n, k, Su, Smean);
        REQUIRE(ApproxEq(Smean, Smean_seq));
      }

      { // Blocked buffered vectorized version
        std::unique_ptr<double> mean(new double[u_size]);
        span<double> Smean(mean.get(), u_size);

        lmv_2d_vectorized_buffered_blocked<4, 64, 4>(n, k, Su, Smean);
        REQUIRE(ApproxEq(Smean, Smean_seq));
      }
    }
    fprintf(stderr, "\033[32;1m OK \033[0m\n");
  }
  std::printf("Ran %lu tests successfully\n", test_n);
}
