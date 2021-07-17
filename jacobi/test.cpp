#include <cstdio>
#include <cfloat>
#include <cstdlib>
#include <memory>

#include "jacobi.h"
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
  if (a.size() != b.size())
    return false;
  bool equal = true;

  for (ptrdiff_t i = 0; i < a.size(); ++i)
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
    return 0;
}
