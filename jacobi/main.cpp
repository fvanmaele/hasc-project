#include <cstdio>
#include <random>
#define HASC_SPAN_CHECKED // for debugging purposes
#include "jacobi.h"

template <typename T>
T unifrnd(const int a, const int b)
{
  std::random_device rand_dev;
  std::mt19937_64 generator(rand_dev());
  std::uniform_real_distribution<double> distr(a, b);
  return distr(generator);
}

int main()
{
  return 0;
}
