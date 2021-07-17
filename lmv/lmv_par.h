/// @file
/// @brief Parallel algorithms for computing local mean value
/// @author Ferdinand Vanmaele

#ifndef HASC_LMV_PAR_H
#define HASC_LMV_PAR_H

namespace hasc
{
template <int MI, int MJ>
inline void lmv_2d_blocked_openmp(int n, int k, span<const double> u, span<double> mean)
{
  for (int I = 0; I < n; I+=MI)
    for (int J = 0; J < n; J+=MJ)
      lmv_2d(I, MIN(I+MI, n),
             J, MIN(J+MJ, n), n, k, u, mean);
}

}


#endif // HASC_LMV_PAR_H
