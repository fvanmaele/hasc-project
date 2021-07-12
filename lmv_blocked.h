/// @file
/// @brief Blocked algorithms for computing local mean value
/// @author Ferdinand Vanmaele

#ifndef HASC_LMV_BLOCKED_H
#define HASC_LMV_BLOCKED_H
#include <cassert>
#include "span.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))
// minimum and maximum value
#define min(X, Y) ((X) < (Y) ? (X) : (Y))
#define max(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{

/// @brief Local mean value (blocked version)
/// @tparam MJ Block size (columns)
/// @tparam MI Block size (rows)
template <size_t MJ, size_t MI>
inline void lmv_2d_blocked(int n, int k, span<const double> u, span<double> mean)
{
  for (int J = 0; J < n; J+=MJ)
    for (int I = 0; I < n; I+=MI)
    {

    }
}

// Basic blocked version which does not distinguish between outer and inner blocks.
template <size_t MJ, size_t MI>
inline void lmv_2d_blocked_2(int n, int k, span<const double> u, span<double> mean)
{
  for (int J = 0; J < n; J+=MJ)
    for (int I = 0; I < n; I+=MI)
    {

    }
}

} // namespace hasc




#endif // HASC_LMV_BLOCKED_H
