#ifndef _DATA_CONVERT_H_
#define _DATA_CONVERT_H_

#include <cuda.h>
#include "common.h"

__global__ void device_convert(double *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = x[tid];
    }
}

void host_convert_forward(double *x, float *y, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i];
    }
}

void host_convert_backward(float *x, double *y, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        y[i] = x[i];
    }
}

#endif