#define HASC_SPAN_CHECKED
#include "span.h"
#include "local_mean_value.h"
#include "aligned_ptr.h"

int main() {
  using namespace hasc;
  double a[5];
  a[0] = a[1] = a[2] = a[3] = a[4] = 1;
  span<double> S(a, 5);
  S[1] = 444;
  printf("%f\n", S[1]);

  return 0;
}
