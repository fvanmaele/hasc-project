#ifndef HASC_JACOBI_VCL_H
#define HASC_JACOBI_VCL_H
#include <cassert>
#include <vectorclass/vectorclass.h>
#include "span.h"

// row-major index mapping
#define INDEX(i, j, n) ((i)*n+(j))

// minimum and maximum value
#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

namespace hasc
{

} // namespace hasc

#endif // HASC_JACOBI_VCL_H
