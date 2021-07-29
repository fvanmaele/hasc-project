/// @file aligned_array.h
/// @brief Dynamically allocated array with variable alignment
/// @author Ferdinand Vanmaele

#ifndef HASC_ALIGNED_ARRAY_H
#define HASC_ALIGNED_ARRAY_H
#include <new>

namespace hasc
{
template <typename T, size_t alignment = 64>
struct aligned_array
{
private:
  T* __restrict ptr;
  size_t n;

public:
  explicit aligned_array(size_t n_)
    : ptr(new (std::align_val_t(alignment)) T[n_]), n(n_)
  {}

  // move-only type
  aligned_array(const aligned_array& other) = delete;
  aligned_array& operator=(const aligned_array& other) = delete;
  aligned_array(aligned_array&& other) = default;
  aligned_array& operator=(aligned_array&& other) = default;

  ~aligned_array() {
    delete[] ptr;
  }

  // accessor methods
  constexpr T* data() {
    return ptr;
  }
  constexpr const T* data() const {
    return ptr;
  }
  constexpr size_t size() const {
    return n;
  }
};

} // namespace hasc

#endif // HASC_ALIGNED_ARRAY_H
