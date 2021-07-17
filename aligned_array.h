/// @file aligned_array.h
/// @brief
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
  T* __restrict__ ptr;
  ptrdiff_t n;

public:
  explicit aligned_array(ptrdiff_t n_)
    : ptr(new (std::align_val_t(alignment)) T[n_]), n(n_)
  {}

  // move-only type
  aligned_array(const aligned_array& other) = delete;
  aligned_array& operator=(const aligned_array& other) = delete;
  aligned_array(aligned_array&& other) = default;
  aligned_array& operator=(aligned_array&& other) = default;

  ~aligned_array()
  {
    delete[] ptr;
  }

  // accessor methods
  T* data() noexcept {
    return ptr;
  }
  const T* data() const noexcept {
    return ptr;
  }
  ptrdiff_t size() const noexcept {
    return n;
  }
};

} // namespace hasc

#endif // HASC_ALIGNED_ARRAY_H
