#ifndef HASC_SIMDSELECTOR_H
#define HASC_SIMDSELECTOR_H
#include <cstddef>
#include <vectorclass.h>

namespace hasc
{
template<size_t simd_width>
struct SIMDSelector
{};

template<>
struct SIMDSelector<2>
{
  static const size_t simd_width = 2;
  static const size_t simd_registers = 16;
  typedef Vec2d SIMDType;
};

template<>
struct SIMDSelector<4>
{
  static const size_t simd_width = 4;
  static const size_t simd_registers = 16;
  typedef Vec4d SIMDType;
};

template<>
struct SIMDSelector<8>
{
  static const size_t simd_width = 8;
  static const size_t simd_registers = 32;
  typedef Vec8d SIMDType;
};

} // namespace hasc

#endif // HASC_SIMDSELECTOR_H
