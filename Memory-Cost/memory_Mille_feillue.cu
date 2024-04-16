#include <stdio.h>
#include <cuda_fp16.h>
#include <sys/time.h>
#include "csr2block.h"
#include "blockspmv_cpu.h"
#include "utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "./biio2.0/src/biio.h"
#include "common.h"


#define NUM_THREADS 128
#define NUM_BLOCKS 16


#define THREAD_ID threadIdx.x + blockIdx.x *blockDim.x
#define THREAD_COUNT gridDim.x *blockDim.x


#define epsilon 1e-6

#define IMAX 1000

double utime()
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return (tv.tv_sec + double(tv.tv_usec) * 1e-6);
}
__global__ void device_convert(double *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = x[tid];
    }
}
__global__ void device_convert_half(double *x, half *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = __double2half(x[tid]);
    }
}
__global__ void add_mix(double *y, float *y_float, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] += (double)(y_float[tid]);
    }
}
__global__ void add_mix_half(double *y, half *y_half, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] += (double)(y_half[tid]);
    }
}

__global__ void veczero(int n, double *vec)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        vec[i] = 0;
}

__global__ void scalardiv(double *num, double *den, double *result)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *result = (*num) / (*den);
}

__global__ void axpy(int n, double *a, double *x, double *y, double *r)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        r[i] = y[i] + (*a) * x[i];
}

// Computes y= y-a*x for n-length vectors x and y, and scalar a.
__global__ void ymax(int n, double *a, double *x, double *y)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        y[i] = y[i] - (*a) * x[i];
}


// Sets dest=src for scalars on the GPU.
void scalarassign(double *dest, double *src)
{
    cudaMemcpy(dest, src, sizeof(double), cudaMemcpyDeviceToDevice);
}

// Sets dest=src for n-length vectors on the GPU.
void vecassign(double *dest, double *src, int n)
{
    cudaMemcpy(dest, src, sizeof(double) * n, cudaMemcpyDeviceToDevice);
}


__global__ void stir_spmv_cuda_kernel_newcsr(int tilem, int tilen, int rowA, int colA, int nnzA,
                                                     int *d_tile_ptr,
                                                     int *d_tile_columnidx,
                                                     char *d_Format,
                                                     int *d_blknnz,
                                                     unsigned char *d_blknnznnz,
                                                     unsigned char *d_csr_compressedIdx,
                                                     double *d_Blockcsr_Val_d,
                                                     float *d_Blockcsr_Val_f,
                                                     half *d_Blockcsr_Val_h,
                                                     unsigned char *d_Blockcsr_Ptr,
                                                     int *d_ptroffset1,
                                                     int *d_ptroffset2,
                                                     int rowblkblock,
                                                     unsigned int *d_blkcoostylerowidx,
                                                     int *d_blkcoostylerowidx_colstart,
                                                     int *d_blkcoostylerowidx_colstop,
                                                     double *d_x_d,
                                                     double *d_y_d,
                                                     unsigned char *d_blockrowid_new,
                                                     unsigned char *d_blockcsr_ptr_new,
                                                     int *d_nonzero_row_new,
                                                     unsigned char *d_Tile_csr_Col)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_x_d[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_x_warp_d = &s_x_d[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    if (blki_blc < rowblkblock)
    {
        double sum_d = 0.0;
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;
        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_tile_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_tile_ptr[blki + 1];
        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
        }
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {
            int colid = s_columnid_local[blkj - rowblkjstart];
            int x_offset = colid * BLOCK_SIZE;
            int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
            int ri = lane_id >> 1;
            int virtual_lane_id = lane_id & 0x1;
            int s1 = d_nonzero_row_new[blkj];
            int s2 = d_nonzero_row_new[blkj + 1];
            sum_d = 0.0;
            if (lane_id < BLOCK_SIZE)
            {
                s_x_warp_d[lane_id] = d_x_d[x_offset + lane_id];
            }
            if (ri < s2 - s1)
            {
                int ro = d_blockrowid_new[s1 + ri + 1];
                for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    int csrcol = d_Tile_csr_Col[csroffset + rj];
                    sum_d += s_x_warp_d[csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                }
                atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            }
        }
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr_balance_mix(int tilem, int tilen, int rowA, int colA, int nnzA,
                                                         int *d_tile_ptr,
                                                         int *d_tile_columnidx,
                                                         char *d_Format,
                                                         int *d_blknnz,
                                                         unsigned char *d_blknnznnz,
                                                         unsigned char *d_csr_compressedIdx,
                                                         double *d_Blockcsr_Val_d,
                                                         float *d_Blockcsr_Val_f,
                                                         half *d_Blockcsr_Val_h,
                                                         unsigned char *d_Blockcsr_Ptr,
                                                         int *d_ptroffset1,
                                                         int *d_ptroffset2,
                                                         int rowblkblock,
                                                         unsigned int *d_blkcoostylerowidx,
                                                         int *d_blkcoostylerowidx_colstart,
                                                         int *d_blkcoostylerowidx_colstop,
                                                         double *d_x_d,
                                                         float *d_x_f,
                                                         half *d_x_h,
                                                         double *d_y_d,
                                                         float *d_y_f,
                                                         half *d_y_h,
                                                         unsigned char *d_blockrowid_new,
                                                         unsigned char *d_blockcsr_ptr_new,
                                                         int *d_nonzero_row_new,
                                                         unsigned char *d_Tile_csr_Col,
                                                         int *d_vis_mix,
                                                         unsigned int *d_vis_mix_16,
                                                         unsigned int *d_vis_mix_32)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_x_d[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_x_warp_d = &s_x_d[local_warp_id * BLOCK_SIZE];
    __shared__ float s_x_f[WARP_PER_BLOCK * BLOCK_SIZE];
    float *s_x_warp_f = &s_x_f[local_warp_id * BLOCK_SIZE];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    if (blki_blc < rowblkblock)
    {
        double sum_d = 0.0;
        float sum_f = 0.0;
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;
        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_tile_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_tile_ptr[blki + 1];
        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
        }
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {
            int colid = s_columnid_local[blkj - rowblkjstart];
            int x_offset = colid * BLOCK_SIZE;
            int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
            int ri = lane_id >> 1;
            int virtual_lane_id = lane_id & 0x1;
            int s1 = d_nonzero_row_new[blkj];
            int s2 = d_nonzero_row_new[blkj + 1];
            // #ifdef USE_D
            //  double
            //if (!d_vis_mix[colid])
            // if(!((d_vis_mix_16[colid / 16] >> (colid % 16)) & 1))
            if(!((d_vis_mix_32[colid / 32] >> (colid % 32)) & 1))
            {
                sum_d = 0;
                if (lane_id < BLOCK_SIZE)
                {
                    s_x_warp_d[lane_id] = d_x_d[x_offset + lane_id];
                }
                if (ri < s2 - s1)
                {
                    int ro = d_blockrowid_new[s1 + ri + 1];
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_d += s_x_warp_d[csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
            }
            else
            {
                sum_f = 0.0;
                if (lane_id < BLOCK_SIZE)
                {
                    s_x_warp_f[lane_id] = d_x_f[x_offset + lane_id];
                }
                if (ri < s2 - s1)
                {
                    int ro = d_blockrowid_new[s1 + ri + 1];
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_f += s_x_warp_f[csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                    }
                    atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                }
            }
        }
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr_balance_inc(int tilem, int tilen, int rowA, int colA, int nnzA,
                                                         MAT_PTR_TYPE *d_tile_ptr,
                                                         int *d_tile_columnidx,
                                                         char *d_Format,
                                                         int *d_blknnz,
                                                         unsigned char *d_blknnznnz,
                                                         unsigned char *d_csr_compressedIdx,
                                                         MAT_VAL_TYPE *d_Blockcsr_Val,
                                                         unsigned char *d_Blockcsr_Ptr,
                                                         int *d_ptroffset1,
                                                         int *d_ptroffset2,
                                                         int rowblkblock,
                                                         unsigned int *d_blkcoostylerowidx,
                                                         int *d_blkcoostylerowidx_colstart,
                                                         int *d_blkcoostylerowidx_colstop,
                                                         MAT_VAL_TYPE *d_x,
                                                         MAT_VAL_TYPE *d_y,
                                                         unsigned char *d_blockrowid_new,
                                                         unsigned char *d_blockcsr_ptr_new,
                                                         int *d_nonzero_row_new,
                                                         unsigned char *d_Tile_csr_Col,
                                                         int *d_vis)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    __shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];
    __shared__ MAT_VAL_TYPE s_y[WARP_PER_BLOCK * BLOCK_SIZE];
    MAT_VAL_TYPE *s_y_warp = &s_y[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    if (blki_blc < rowblkblock)
    {
        MAT_VAL_TYPE sum = 0;
        // MAT_VAL_TYPE sumsum = 0;
        if (lane_id < BLOCK_SIZE)
            s_y_warp[lane_id] = 0;
        int coostyleblkrowidx = d_blkcoostylerowidx[blki_blc];
        int signbit = (coostyleblkrowidx >> 31) & 0x1;
        int blki = signbit == 1 ? coostyleblkrowidx & 0x7FFFFFFF : coostyleblkrowidx;
        int rowblkjstart = signbit == 1 ? d_blkcoostylerowidx_colstart[blki_blc] : d_tile_ptr[blki];
        int rowblkjstop = signbit == 1 ? d_blkcoostylerowidx_colstop[blki_blc] : d_tile_ptr[blki + 1];
        if (lane_id < rowblkjstop - rowblkjstart)
        {
            s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
        }
        for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        {
            int colid = s_columnid_local[blkj - rowblkjstart];
            if (d_vis[colid])
            {
                int x_offset = colid * BLOCK_SIZE;
                sum = 0;
                int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
                if (lane_id < BLOCK_SIZE)
                    s_x_warp[lane_id] = d_x[x_offset + lane_id];
                int ri = lane_id >> 1;
                int virtual_lane_id = lane_id & 0x1;
                int s1 = d_nonzero_row_new[blkj];
                int s2 = d_nonzero_row_new[blkj + 1];
                if (ri < s2 - s1)
                {
                    int ro = d_blockrowid_new[s1 + ri + 1];
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum += s_x_warp[csrcol] * d_Blockcsr_Val[csroffset + rj];
                    }
                    atomicAdd(&d_y[blki * BLOCK_SIZE + ro], sum);
                }
            }
        }
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr_balance_inc_balance(
    MAT_PTR_TYPE *d_tile_ptr,
    int *d_tile_columnidx,
    MAT_VAL_TYPE *d_Blockcsr_Val,
    int *d_ptroffset1,
    int rowblkblock,
    MAT_VAL_TYPE *d_x,
    MAT_VAL_TYPE *d_y,
    unsigned char *d_blockrowid_new,
    unsigned char *d_blockcsr_ptr_new,
    int *d_nonzero_row_new,
    unsigned char *d_Tile_csr_Col,
    int *d_vis,
    int *d_b_start,
    int *d_tile_rowidx,
    int *d_b_map)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // const int blki = global_id >> 5;
    const int blki_blc = global_id >> 5;
    // const int local_warp_id = threadIdx.x >> 5;
    //__shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    // MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];
    //  __shared__ MAT_VAL_TYPE s_y[WARP_PER_BLOCK * BLOCK_SIZE];
    //  MAT_VAL_TYPE *s_y_warp = &s_y[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    // __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    if (blki_blc < rowblkblock)
    {
        MAT_VAL_TYPE sum = 0;
        int rowblkjstart = d_b_start[blki_blc];
        int rowblkjstop = d_b_start[blki_blc + 1];
        // if (lane_id < rowblkjstop - rowblkjstart)
        // {
        //     s_columnid_local[lane_id] = d_tile_columnidx[d_b_map[rowblkjstart+ lane_id]];
        //     //s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
        // }
        for (int blkj_n = rowblkjstart; blkj_n < rowblkjstop; blkj_n++)
        {
            int blkj = d_b_map[blkj_n];
            int colid = d_tile_columnidx[blkj];
            // int colid = s_columnid_local[blkj_n - rowblkjstart];
            //  if (d_vis[colid])
            //  {
            int x_offset = colid * BLOCK_SIZE;
            sum = 0;
            int csroffset = d_ptroffset1[blkj];
            // int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
            //  if (lane_id < BLOCK_SIZE)
            //      s_x_warp[lane_id] = d_x[x_offset + lane_id];
            int ri = lane_id >> 1;
            int virtual_lane_id = lane_id & 0x1;
            int s1 = d_nonzero_row_new[blkj];
            int s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                int blki = d_tile_rowidx[blkj];
                int ro = d_blockrowid_new[s1 + ri + 1];
                for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    int csrcol = d_Tile_csr_Col[csroffset + rj];
                    // sum += s_x_warp[csrcol] * d_Blockcsr_Val[csroffset + rj];
                    sum += d_x[x_offset + csrcol] * d_Blockcsr_Val[csroffset + rj];
                }
                atomicAdd(&d_y[blki * BLOCK_SIZE + ro], sum);
            }
            //}
        }
    }
}


__global__ void stir_spmv_cuda_kernel_newcsr_balance_inc_balance_mix(
    MAT_PTR_TYPE *d_tile_ptr,
    int *d_tile_columnidx,
    double *d_Blockcsr_Val_d,
    float *d_Blockcsr_Val_f,
    int *d_ptroffset1,
    int rowblkblock,
    double *d_x_d,
    float *d_x_f,
    double *d_y_d,
    float *d_y_f,
    unsigned char *d_blockrowid_new,
    unsigned char *d_blockcsr_ptr_new,
    int *d_nonzero_row_new,
    unsigned char *d_Tile_csr_Col,
    int *d_vis,
    int *d_b_start,
    int *d_tile_rowidx,
    int *d_b_map,
    unsigned int *d_vis_mix_32)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // const int blki = global_id >> 5;
    const int blki_blc = global_id >> 5;
    // const int local_warp_id = threadIdx.x >> 5;
    //__shared__ MAT_VAL_TYPE s_x[WARP_PER_BLOCK * BLOCK_SIZE];
    // MAT_VAL_TYPE *s_x_warp = &s_x[local_warp_id * BLOCK_SIZE];
    //  __shared__ MAT_VAL_TYPE s_y[WARP_PER_BLOCK * BLOCK_SIZE];
    //  MAT_VAL_TYPE *s_y_warp = &s_y[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    // __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];
    if (blki_blc < rowblkblock)
    {
        double sum_d = 0.0;
        float sum_f=0.0;
        int rowblkjstart = d_b_start[blki_blc];
        int rowblkjstop = d_b_start[blki_blc + 1];
        for (int blkj_n = rowblkjstart; blkj_n < rowblkjstop; blkj_n++)
        {
            int blkj = d_b_map[blkj_n];
            int colid = d_tile_columnidx[blkj];
            int x_offset = colid * BLOCK_SIZE;
            int csroffset = d_ptroffset1[blkj];
            int ri = lane_id >> 1;
            int virtual_lane_id = lane_id & 0x1;
            int s1 = d_nonzero_row_new[blkj];
            int s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                int blki = d_tile_rowidx[blkj];
                int ro = d_blockrowid_new[s1 + ri + 1];
                // if(!((d_vis_mix_32[colid / 32] >> (colid % 32)) & 1))
                // {
                    // sum_d=0;
                    // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    // {
                    //     int csrcol = d_Tile_csr_Col[csroffset + rj];
                    //     sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                    // }
                    // atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                // }
                // else
                // {
                    sum_f = 0.0;
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        //sum_f += d_x_f[x_offset + csrcol] * float(d_Blockcsr_Val_d[csroffset + rj]);
                        sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                    }
                    atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                //}
            }
        }
    }
}



__global__ void te1(double *p, double *last_val, int *vis_new,unsigned int *vis_mix_32)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // double res=fabs(p[global_id] - last_val[global_id]);
    if (fabs(p[global_id] - last_val[global_id]) >= 0.00000001)
    {
        vis_new[blockIdx.x] = 1;
    }
    // use low precision
    if (fabs(p[global_id] - last_val[global_id]) <= 1e-1)
    {
        //vis_mix[blockIdx.x] = 1;
        //vis_mix_16[blockIdx.x/16] |= (1 << (blockIdx.x % 16));
        //atomicOr(&(vis_mix_16[blockIdx.x/16]),(1 << (blockIdx.x % 16)));
        atomicOr(&(vis_mix_32[blockIdx.x/32]),(1 << (blockIdx.x % 32)));
    }
}
extern "C" void cg_solve_inc(int *RowPtr, int *ColIdx, MAT_VAL_TYPE *Val, MAT_VAL_LOW_TYPE *Val_Low, double *x, double *b, int n, int *iter, int maxiter, double threshold, char *filename, int nnzR, int ori)
{
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    struct timeval t1, t2, t3, t4;
    //gettimeofday(&t1, NULL);
    int rowA = n;
    int colA = ori;
    rowA = (rowA / BLOCK_SIZE) * BLOCK_SIZE;
    Tile_matrix *matrix = (Tile_matrix *)malloc(sizeof(Tile_matrix));
    Tile_create(matrix,
                rowA, colA, nnzR,
                RowPtr,
                ColIdx,
                Val,
                Val_Low);
    int num_seg = ceil((double)rowA / BLOCK_SIZE);
    int memory_all=0;
    double cnt_memory=0;
    // num_seg += 1;
    //printf("rowA=%d colA=%d\n", rowA, colA);
    int tilenum = matrix->tilenum;
    int *ptroffset1 = (int *)malloc(sizeof(int) * tilenum);
    int *ptroffset2 = (int *)malloc(sizeof(int) * tilenum);
    memset(ptroffset1, 0, sizeof(int) * tilenum);
    memset(ptroffset2, 0, sizeof(int) * tilenum);
    MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * n);
    memset(x, 0, sizeof(double) * n);
    memset(y, 0, sizeof(MAT_VAL_TYPE) * n);
    int rowblkblock = 0;
    unsigned int *blkcoostylerowidx;
    int *blkcoostylerowidx_colstart;
    int *blkcoostylerowidx_colstop;
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    blockspmv_cpu(matrix,
                 ptroffset1,
                 ptroffset2,
                 &rowblkblock,
                 &blkcoostylerowidx,
                 &blkcoostylerowidx_colstart,
                 &blkcoostylerowidx_colstop,
                 rowA, colA, nnzR,
                 RowPtr,
                 ColIdx,
                 Val,
                 x,
                 y,
                 y_golden);
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    //printf("rowA=%d,rowblkblock=%d\n", tilem, rowblkblock);
    int *tile_nnz = matrix->tile_nnz;
    char *Format = matrix->Format;
    int *blknnz = matrix->blknnz;
    unsigned char *blknnznnz = matrix->blknnznnz;
    char *tilewidth = matrix->tilewidth;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    MAT_VAL_LOW_TYPE *Blockcsr_Val_Low = matrix->Blockcsr_Val_Low;
    unsigned char *Tile_csr_Col = matrix->Tile_csr_Col;
    unsigned char *csr_compressedIdx = matrix->csr_compressedIdx;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    int csrsize = matrix->csrsize;
    int csrptrlen = matrix->csrptrlen;
    int csr_csize = csrsize % 2 == 0 ? csrsize / 2 : csrsize / 2 + 1;


    int *vis_new = (int *)malloc(sizeof(int) * num_seg);
    memset(vis_new, 0, sizeof(int) * num_seg);
    int *d_vis_new;
    cudaMalloc((void **)&d_vis_new, num_seg * sizeof(int));
    int vis_new_size_32 = (num_seg / 32) + 1;
    int *d_vis;
    cudaMalloc((void **)&d_vis, num_seg * sizeof(int));
    unsigned int *d_vis_mix_32;
    cudaMalloc((void **)&d_vis_mix_32, vis_new_size_32 * sizeof(unsigned int));



    MAT_PTR_TYPE *d_tile_ptr;
    int *d_tile_columnidx;
    char *d_Format;
    int *d_blknnz;
    unsigned char *d_blknnznnz;
    int *tile_rowidx = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_rowidx, 0, sizeof(int) * tilenum);
    int *d_tile_rowidx;
    cudaMalloc((void **)&d_tile_rowidx, tilenum * sizeof(int));
    //printf("tilenum=%d tilem=%d num_seg=%d rowA=%d\n", tilenum, tilem, num_seg, rowA);
    cudaMalloc((void **)&d_tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_tile_columnidx, tilenum * sizeof(int));
    cudaMalloc((void **)&d_Format, tilenum * sizeof(char));
    cudaMalloc((void **)&d_blknnz, (tilenum + 1) * sizeof(int));
    cudaMalloc((void **)&d_blknnznnz, (tilenum + 1) * sizeof(unsigned char));

    cudaMemcpy(d_tile_ptr, tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_columnidx, tile_columnidx, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Format, Format, tilenum * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blknnz, blknnz, (tilenum + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blknnznnz, blknnznnz, (tilenum + 1) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    memory_all+=((tilem + 1) * sizeof(MAT_PTR_TYPE));//tile_ptr
    memory_all+=(tilenum * sizeof(int));//tile_columnidx
    memory_all+=((tilenum + 1) * sizeof(unsigned char));//blknnz
    //memory_all+=((tilenum + 1) * sizeof(unsigned char));//blkmask
    memory_all+=((tilenum + 1) * sizeof(int));//vis
    memory_all+=((tilenum + 1) * sizeof(int)); //rowindex
    // memory_all+=((tilenum + 1) * sizeof(unsigned char));//vis_mask8
    // memory_all+=((tilenum + 1) * sizeof(unsigned char));//vis_mask16
    // memory_all+=((tilenum + 1) * sizeof(unsigned char));//vis_mask32
    memory_all+=((tilenum + 3) * sizeof(int)); //d_block //d_dot //d_iter

    cnt_memory+=(double)memory_all/1048576;
    memory_all=0;
    // CSR
    unsigned char *d_csr_compressedIdx = (unsigned char *)malloc((csr_csize) * sizeof(unsigned char));
    MAT_VAL_TYPE *d_Blockcsr_Val;
    unsigned char *d_Blockcsr_Ptr;

    cudaMalloc((void **)&d_csr_compressedIdx, (csr_csize) * sizeof(unsigned char));
    cudaMalloc((void **)&d_Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char));

    cudaMemcpy(d_csr_compressedIdx, csr_compressedIdx, (csr_csize) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Val, Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Ptr, Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char), cudaMemcpyHostToDevice);

    float *d_Blockcsr_Val_float;
    half *d_Blockcsr_Val_half;
    half *Blockcsr_Val_half = (half *)malloc(sizeof(half) * csrsize);
    cudaMalloc((void **)&d_Blockcsr_Val_float, (csrsize) * sizeof(float));
    cudaMalloc((void **)&d_Blockcsr_Val_half, (csrsize) * sizeof(half));
    cudaMemcpy(d_Blockcsr_Val_float, Blockcsr_Val_Low, (csrsize) * sizeof(float), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < csrsize; i++)
    {
        Blockcsr_Val_half[i] = (half)(Blockcsr_Val_Low[i]);
    }
    cudaMemcpy(d_Blockcsr_Val_half, Blockcsr_Val_half, (csrsize) * sizeof(half), cudaMemcpyHostToDevice);
    free(Blockcsr_Val_half);

    unsigned int *d_blkcoostylerowidx;
    int *d_blkcoostylerowidx_colstart;
    int *d_blkcoostylerowidx_colstop;

    cudaMalloc((void **)&d_blkcoostylerowidx, rowblkblock * sizeof(unsigned int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstart, rowblkblock * sizeof(int));
    cudaMalloc((void **)&d_blkcoostylerowidx_colstop, rowblkblock * sizeof(int));

    cudaMemcpy(d_blkcoostylerowidx, blkcoostylerowidx, rowblkblock * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstart, blkcoostylerowidx_colstart, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcoostylerowidx_colstop, blkcoostylerowidx_colstop, rowblkblock * sizeof(int), cudaMemcpyHostToDevice);

    int *d_ptroffset1;
    int *d_ptroffset2;

    cudaMalloc((void **)&d_ptroffset1, tilenum * sizeof(int));
    cudaMalloc((void **)&d_ptroffset2, tilenum * sizeof(int));
    cudaMemcpy(d_ptroffset1, ptroffset1, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ptroffset2, ptroffset2, tilenum * sizeof(int), cudaMemcpyHostToDevice);

    // x and y
    MAT_VAL_TYPE *d_x;
    MAT_VAL_TYPE *d_y;
    float *d_x_float;
    float *d_y_float;

    cudaMalloc((void **)&d_x, rowA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_y, rowA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_x_float, rowA * sizeof(float));
    cudaMalloc((void **)&d_y_float, rowA * sizeof(float));
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));

    double *k_b, *k_x, *k_r, *k_d, *k_q, *k_s;
    float *k_q_float, *k_d_float;
    half *k_d_half, *k_q_half;
    double *k_d_last;
    double *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;
    double t, s0, snew;
    double *test = (double *)malloc(sizeof(double) * n);
    double *k_val;
    int iterations = 0;

    cudaMalloc((void **)&k_b, sizeof(double) * (n));
    cudaMemcpy(k_b, b, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_val, sizeof(double) * (nnzR));
    cudaMemcpy(k_val, Val, sizeof(double) * (nnzR), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&k_x, sizeof(double) * (n));
    cudaMalloc((void **)&k_r, sizeof(double) * (n + 1));
    cudaMalloc((void **)&k_d, sizeof(double) * (n + 1));
    cudaMalloc((void **)&k_d_last, sizeof(double) * (n + 1));
    cudaMalloc((void **)&k_d_float, sizeof(float) * (n));
    cudaMalloc((void **)&k_d_half, sizeof(half) * (n));
    cudaMalloc((void **)&k_q, sizeof(double) * (n));
    cudaMalloc((void **)&k_q_float, sizeof(float) * (n));
    cudaMalloc((void **)&k_q_half, sizeof(half) * (n));
    cudaMalloc((void **)&k_s, sizeof(double) * (n));
    cudaMalloc((void **)&k_alpha, sizeof(double));
    cudaMalloc((void **)&k_snew, sizeof(double) * NUM_BLOCKS);
    cudaMalloc((void **)&k_sold, sizeof(double));
    cudaMalloc((void **)&k_beta, sizeof(double));
    cudaMalloc((void **)&k_s0, sizeof(double));
    double *r = (double *)malloc(sizeof(double) * (n + 1));
    memset(r, 0, sizeof(double) * (n + 1));
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);

    //cudaBindTexture(NULL, tex_val, k_val, sizeof(double) * (nnzR));
    veczero<<<1, BlockDim>>>(n, k_x);
    // r=b-Ax (r=b since x=0), and d=M^(-1)r
    cudaMemcpy(k_r, k_b, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
    cublasDdot(cublasHandle, n, k_r, 1, k_r, 1, k_s0);
    // cudaMemcpy(r, k_r, sizeof(double) * (n), cudaMemcpyDeviceToHost);
    // r[n] = 1.1;
    // cudaMemcpy(k_r, r, sizeof(double) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d, k_r, sizeof(double) * (n + 1), cudaMemcpyDeviceToDevice);
    // r[n] = 1.2;
    // cudaMemcpy(k_d_last, r, sizeof(double) * (n + 1), cudaMemcpyHostToDevice);
    //  snew = s0
    scalarassign(k_snew, k_s0);
    // Copy snew and s0 back to host so that host can evaluate stopping condition
    cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s0, k_s0, sizeof(double), cudaMemcpyDeviceToHost);
    //printf("begin GPU CG\n");
    // printf("s0=%lf\n",s0);
    cudaMemset(d_vis_new, 0, num_seg * sizeof(int));
    //cudaMemset(d_vis_mix, 0, num_seg * sizeof(int));
    double time_spmv = 0;

    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            csrcount = ptroffset2[blkj];
            tile_rowidx[blkj] = blki;
            for (int ri = 0; ri < rowlength; ri++)
            {
                int stop = ri == rowlength - 1 ? (matrix->blknnz[blkj + 1] - matrix->blknnz[blkj]) : matrix->Blockcsr_Ptr[ri + 1 + csrcount];
                if (stop != matrix->Blockcsr_Ptr[csrcount + ri])
                {
                    nonzero_row_new[blkj] += 1;
                }
            }
            nonzero_row_new[blkj] += 1;
        }
    }
    exclusive_scan(nonzero_row_new, tilenum + 1);
    int cnt_non_new = nonzero_row_new[tilenum];
    unsigned char *blockrowid_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockrowid_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    unsigned char *blockcsr_ptr_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockcsr_ptr_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    int csrcount_new1 = 0;

    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            csrcount = ptroffset2[blkj];
            csrcount_new1 = nonzero_row_new[blkj];
            int fl = 0;
            for (int ri = 0; ri < rowlength; ri++)
            {
                int stop = ri == rowlength - 1 ? (matrix->blknnz[blkj + 1] - matrix->blknnz[blkj]) : matrix->Blockcsr_Ptr[ri + 1 + csrcount];
                if (ri == 0)
                {
                    blockrowid_new[csrcount_new1 + fl] = ri;
                    blockcsr_ptr_new[csrcount_new1 + fl] = 0;
                    fl++;
                }
                if (stop != matrix->Blockcsr_Ptr[csrcount + ri])
                {
                    blockrowid_new[csrcount_new1 + fl] = ri;
                    blockcsr_ptr_new[csrcount_new1 + fl] = stop;
                    fl++;
                }
            }
        }
    }
    double pro_cnt=0.0;
    unsigned char *d_blockrowid_new;
    unsigned char *d_blockcsr_ptr_new;
    int *d_nonzero_row_new;
    unsigned char *d_Tile_csr_Col;
    cudaMalloc((void **)&d_blockrowid_new, sizeof(unsigned char) * (cnt_non_new + 1));
    cudaMalloc((void **)&d_blockcsr_ptr_new, sizeof(unsigned char) * (cnt_non_new + 1));
    cudaMalloc((void **)&d_nonzero_row_new, sizeof(int) * (tilenum + 1));
    cudaMalloc((void **)&d_Tile_csr_Col, sizeof(unsigned char) * (matrix->csrsize));
    cudaMemcpy(d_blockrowid_new, blockrowid_new, sizeof(unsigned char) * (cnt_non_new + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blockcsr_ptr_new, blockcsr_ptr_new, sizeof(unsigned char) * (cnt_non_new + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nonzero_row_new, nonzero_row_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Tile_csr_Col, Tile_csr_Col, sizeof(unsigned char) * (matrix->csrsize), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_rowidx, tile_rowidx, sizeof(int) * (tilenum), cudaMemcpyHostToDevice);
    memory_all+=(sizeof(unsigned char) * (cnt_non_new + 1));
    memory_all+=(sizeof(unsigned char) * (cnt_non_new + 1));
    memory_all+=(sizeof(unsigned char) *tilenum);
    memory_all+=(sizeof(unsigned char) * (matrix->csrsize));
    memory_all+=(sizeof(int) * (tilenum));//rowindex
    cnt_memory+=(double)memory_all/1048576;
    memory_all=0;
    int fp8=0;
    int fp16=0;
    int fp32=0;
    int fp64=0;
    int cnt_row=0;
    for (int blki = 0; blki < tilem; blki++)
    {
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            //int csrcolidx = tile_columnidx[blkj];
            //int x_offset = csrcolidx * BLOCK_SIZE;
            csroffset = matrix->csr_offset[blkj];
            for (int ri = nonzero_row_new[blkj]; ri < nonzero_row_new[blkj + 1]; ri++)
            {
                //double sum_new = 0;
                cnt_row++;
                //int ro = blockrowid_new[ri + 1];
                for (int rj = blockcsr_ptr_new[ri]; rj < blockcsr_ptr_new[ri + 1]; rj++)
                {
                    //fp64++;
                    double val_b=matrix->Blockcsr_Val[csroffset + rj];
                    if(val_b>=(-3.9375)&&val_b<=3.9375)
                    {
                        fp8++;
                        //fp16++;
                        //printf("%e\n",Val[i]);
                    }
                    else if(val_b>=(-6*1e1)&&val_b<=(6*1e1))
                    {
                        fp16++;
                        //fp32++;
                        //printf("%e\n",Val[i]);
                    }
                    else if(val_b>=(-6*1e3)&&val_b<=(6*1e3))
                    {
                        fp32++;
                        //fp64++;
                    }
                    else{
                        fp64++;
                    }
                }
            }
        }
    }
    // memory_all+=(sizeof(double)*fp64+sizeof(float)*fp64+sizeof(half)*fp64+1*fp64);
    // memory_all+=(sizeof(float)*fp32+sizeof(half)*fp32+1*fp32);
    // memory_all+=(sizeof(half)*fp16+1*fp16);
    // memory_all+=(1*fp16);
    memory_all+=sizeof(double)*fp64+sizeof(half)*fp16+sizeof(float)*fp32+sizeof(int8_t)*fp8;
    //memory_all+=(sizeof(double)*(nnzR));
    //memory_all+=(sizeof(double)*nnzR+sizeof(float)*nnzR+sizeof(half)*nnzR+1*nnzR);
    cnt_memory+=(double)memory_all/1048576;
    printf("memory_MB=%lf\n",cnt_memory);
    memory_all=0;
    memory_all+=sizeof(double)*cnt_row;
    double memory_CP=(double)memory_all/1048576;
    char *s = (char *)malloc(sizeof(char) * 150);
    sprintf(s, "memory_MB=%lf,memory_CP=%lf,nnzR=%d\n",cnt_memory,memory_CP,nnzR);
    FILE *file1 = fopen("memory_mille_feillue_new.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
    // cudaFree(k_val);
    // cudaFree(k_b);
    // cudaFree(k_x);
    // cudaFree(k_r);
    // cudaFree(k_d);
    // cudaFree(k_d_float);
    // cudaFree(k_d_half);
    // cudaFree(k_q);
    // cudaFree(k_alpha);
    // cudaFree(k_snew);
    // cudaFree(k_sold);
    // cudaFree(k_beta);
    // cudaFree(k_s0);
    // cudaFree(d_tile_ptr);
    // cudaFree(d_tile_columnidx);
    // cudaFree(d_Format);
    // cudaFree(d_blknnz);
    // cudaFree(d_blknnznnz);
    // cudaFree(d_csr_compressedIdx);
    // cudaFree(d_Blockcsr_Val);
    // cudaFree(d_Blockcsr_Ptr);
    // cudaFree(d_Blockcsr_Val_float);
    // cudaFree(d_Blockcsr_Val_half);
    // cudaFree(d_blkcoostylerowidx);
    // cudaFree(d_blkcoostylerowidx_colstart);
    // cudaFree(d_blkcoostylerowidx_colstop);
    // cudaFree(d_ptroffset1);
    // cudaFree(d_ptroffset2);
    // cudaFree(d_x);
    // cudaFree(d_y);
    // cudaFree(d_x_float);
    // cudaFree(d_y_float);
    // free(matrix);
    // free(ptroffset1);
    // free(ptroffset2);
    // free(y_golden);
    // free(y);
    // free(blkcoostylerowidx);
    // free(blkcoostylerowidx_colstart);
    // free(blkcoostylerowidx_colstop);
    // free(tile_ptr);
    // free(tile_columnidx);
    // free(tile_nnz);
    // free(Format);
    // free(blknnz);
    // free(blknnznnz);
    // free(tilewidth);
    // free(csr_offset);
    // free(csrptr_offset);
    // free(Blockcsr_Val);
    // free(Blockcsr_Val_Low);
    // free(csr_compressedIdx);
    // free(Blockcsr_Ptr);
    // //free(vis);
    // free(vis_new);
    //free(last);
}
int main(int argc, char **argv)
{
  char *filename = argv[1];
  int m, n, nnzR, isSymmetric;
  int *RowPtr;
  int *ColIdx;
  MAT_VAL_TYPE *Val;
  read_Dmatrix_32(&m, &n, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
  MAT_VAL_LOW_TYPE *Val_Low = (MAT_VAL_LOW_TYPE *)malloc(sizeof(MAT_VAL_LOW_TYPE) * nnzR);
  for (int i = 0; i < nnzR; i++)
  {
    Val_Low[i] = Val[i];
  }
  int ori = n;
  n = (n / BLOCK_SIZE) * BLOCK_SIZE;
  m = (m / BLOCK_SIZE) * BLOCK_SIZE;
  MAT_VAL_TYPE *X = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (n));
  MAT_VAL_TYPE *Y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (m));
  MAT_VAL_TYPE *Y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (m));
  memset(X, 0, sizeof(MAT_VAL_TYPE) * (n));
  memset(Y, 0, sizeof(MAT_VAL_TYPE) * (n));
  memset(Y_golden, 0, sizeof(MAT_VAL_TYPE) * (n));
  // int tt;
  // fscanf(file_rhs,"%d",&tt);
  // FILE *fp1 = fopen(file_rhs, "a+");
  // int tt;
  // fscanf(fp1, "%d", &tt);
  for (int i = 0; i < n; i++)
  {
    X[i] = 1;
    //fscanf(fp1, "%lf", &X[i]);
  }
  int iter = 0;
  for (int i = 0; i < n; i++)
    for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
      Y_golden[i] += Val[j] * X[ColIdx[j]];
  // for (int i = 0; i < n; i++)
  // {
  //   fscanf(fp1, "%lf", &X[i]);
  //   
  // }

  cg_solve_inc(RowPtr, ColIdx, Val, Val_Low, X, Y_golden, n, &iter, 10, 1e-5, filename, nnzR, ori);
  //cg_cusparse(filename,RowPtr,ColIdx,Val,Y_golden,n,X,nnzR);
}
