/// @file span.h
/// @brief Simple span implementation with optional bounds checking
/// @author Ferdinand Vanmaele

#ifndef HASC_SPAN_H
#define HASC_SPAN_H
#include <cstddef>

#ifdef HASC_SPAN_CHECKED
#include <cstdlib>
#include <cstdio>
#endif

namespace hasc
{
template <typename T>
class span
{
private:
  T *__restrict__ ptr;
  ptrdiff_t n;

public:
  constexpr span(T *ptr_, ptrdiff_t n_)
    : ptr(ptr_), n(n_) {}

  constexpr T& operator[](ptrdiff_t idx) const
  {
#ifdef HASC_SPAN_CHECKED
    if (idx < 0 || idx >= n) {
      fprintf(stderr, "invalid index access (idx = %ld, n = %ld)\n", idx, n);
      abort();
    }
#endif
    return ptr[idx];
  }
  constexpr ptrdiff_t size() const {
    return n;
  }
  constexpr bool empty() const {
    return n == 0;
  }
};

} // namespace hasc

#endif // HASC_SPAN_H