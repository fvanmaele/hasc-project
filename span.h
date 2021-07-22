/// @file span.h
/// @brief Simple span implementation with optional bounds checking
/// @author Ferdinand Vanmaele

#ifndef HASC_SPAN_H
#define HASC_SPAN_H
#include <cstddef>
#include <cmath>
#include <array>
#include "aligned_array.h"

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

  template <size_t alignment>
  constexpr span(const aligned_array<T, alignment>& A)
    : ptr(A.data()), n(A.ssize()) {}

  template <size_t alignment>
  constexpr span(aligned_array<T, alignment>& A)
    : ptr(A.data()), n(A.ssize()) {}

  template <size_t N>
  constexpr span(const std::array<T, N>& A)
    : ptr(A.data()), n(static_cast<ptrdiff_t>(A.size())) {}

  template <size_t N>
  constexpr span(std::array<T, N>& A)
    : ptr(A.data()), n(static_cast<ptrdiff_t>(A.size())) {}

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
  constexpr ptrdiff_t ssize() const
  {
    return n;
  }
  constexpr T* data() const
  {
    return ptr;
  }
  constexpr T* begin() const
  {
    return ptr;
  }
  constexpr T* end() const
  {
    return ptr+n;
  }
};


} // namespace hasc


#endif // HASC_SPAN_H
