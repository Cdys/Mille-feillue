#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <helper_functions.h> // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>
#include "csr2block.h"
#include "blockspmv_cpu.h"
#include "utils.h"
#include "common.h"
#include "./biio2.0/src/biio.h"
#define min(a, b) ((a < b) ? (a) : (b))
#define NUM_THREADS 128
#define NUM_BLOCKS 16
#define THREAD_ID threadIdx.x + blockIdx.x *blockDim.x
#define THREAD_COUNT gridDim.x *blockDim.x
#define epsilon 1e-6
#define IMAX 1000
double itsol_norm(double *x, int n, int nthread)
{
    int i;
    double t = 0.;
    for (i = 0; i < n; i++)
        t += x[i] * x[i];

    return sqrt(t);
}
double itsol_dot(double *x, double *y, int n, int nthread)
{
    int i;
    double t = 0.;
    for (i = 0; i < n; i++)
        t += x[i] * y[i];

    return t;
}
void mv(int n, int *Rowptr, int *ColIndex, double *Value, double *x, double *y)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = 0.0;
        for (int j = Rowptr[i]; j < Rowptr[i + 1]; j++)
        {
            int k = ColIndex[j];
            y[i] += Value[j] * x[k];
        }
    }
}

void scalarassign(double *dest, double *src)
{
    cudaMemcpy(dest, src, sizeof(double), cudaMemcpyDeviceToDevice);
}
__global__ void scalardiv(double *dest, double *src)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dest) = (*src) / (*dest);
    }
}
__global__ void scalardiv_new(double *dest, double *src1, double *src2)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*dest) = (*src1) / (*src2);
    }
}
__global__ void scalardiv_five(double *prb, double *r1, double *pra, double *r0, double *prc)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*prb) = ((*r1) * (*pra)) / ((*r0) * (*prc));
    }
}
__global__ void yminus_mult(int n, double *sg, double *rg, double *vg, double *pra)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        sg[i] = rg[i] - (*pra) * vg[i];
}

__global__ void yminus_mult_new(int n, double *x, double *pg, double *sg, double *rg, double *tg, double *pra, double *prc)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
    {
        x[i] = x[i] + (*pra) * pg[i] + (*prc) * sg[i];
        rg[i] = sg[i] - (*prc) * tg[i];
    }
}
__global__ void yminus_final(int n, double *pg, double *rg, double *prb, double *prc, double *vg)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        pg[i] = rg[i] + (*prb) * (pg[i] - (*prc) * vg[i]);
}

__global__ void stir_spmv_cuda_kernel_newcsr(int tilem, int tilen, int rowA, int colA, int nnzA,
                                             int *d_tile_ptr,
                                             int *d_tile_columnidx,
                                             unsigned char *d_csr_compressedIdx,
                                             double *d_Blockcsr_Val_d,
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

__global__ void stir_spmv_cuda_kernel_newcsr_balance_inc(int tilem, int tilen, int rowA, int colA, int nnzA,
                                                         MAT_PTR_TYPE *d_tile_ptr,
                                                         int *d_tile_columnidx,
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
    const int blki_blc = global_id >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    if (blki_blc < rowblkblock)
    {
        MAT_VAL_TYPE sum = 0;
        int rowblkjstart = d_b_start[blki_blc];
        int rowblkjstop = d_b_start[blki_blc + 1];
        for (int blkj_n = rowblkjstart; blkj_n < rowblkjstop; blkj_n++)
        {
            int blkj = d_b_map[blkj_n];
            int colid = d_tile_columnidx[blkj];
            int x_offset = colid * BLOCK_SIZE;
            sum = 0;
            int csroffset = d_ptroffset1[blkj];
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
                    sum += d_x[x_offset + csrcol] * d_Blockcsr_Val[csroffset + rj];
                }
                atomicAdd(&d_y[blki * BLOCK_SIZE + ro], sum);
            }
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
                if(!((d_vis_mix_32[colid / 32] >> (colid % 32)) & 1))
                {
                    sum_d=0;
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);

                }
                else
                {
                    sum_f = 0.0;
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                    }
                    atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);

                }
            }
        }
    }
}


__global__ void te1(double *p, double *last_val, int *vis_new, unsigned int *vis_mix_32)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // double res=fabs(p[global_id] - last_val[global_id]);
    // vis_new[blockIdx.x] = 1;
    if (fabs(p[global_id] - last_val[global_id]) >= 0.00000001)
    {
        vis_new[blockIdx.x] = 1;
    }
    // use low precision
    if (fabs(p[global_id] - last_val[global_id]) <= 1e-1)
    {
        //vis_mix[blockIdx.x] = 1;
        // vis_mix_16[blockIdx.x/16] |= (1 << (blockIdx.x % 16));
        //atomicOr(&(vis_mix_16[blockIdx.x / 16]), (1 << (blockIdx.x % 16)));
        atomicOr(&(vis_mix_32[blockIdx.x / 32]), (1 << (blockIdx.x % 32)));
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

__global__ void device_convert(double *x, float *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = x[tid];
    }
}


__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                         int *d_tile_ptr,
                                                         int *d_tile_columnidx,
                                                         unsigned char *d_csr_compressedIdx,
                                                         double *d_Blockcsr_Val_d,
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
                                                         unsigned char *d_Tile_csr_Col,
                                                         int *d_block_signal,
                                                         int *signal_dot,
                                                         int *signal_final,
                                                         int *signal_dot1,
                                                         int *d_ori_block_signal,
                                                         double *k_rh,
                                                         double *k_pra,
                                                         double *k_r1,
                                                         double *k_sg,
                                                         double *k_rg,
                                                         double *k_tg,
                                                         double *k_tmp1,
                                                         double *k_tmp2,
                                                         double *k_x,
                                                         double *k_residual,
                                                         double *k_r_new,
                                                         double *k_r0,
                                                         int *d_balance_tile_ptr,
                                                         int *d_row_each_block,
                                                         int *d_index_each_block,
                                                         int balance_row,
                                                         int *d_non_each_block_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_pra[WARP_PER_BLOCK];
    __shared__ double s_prc[WARP_PER_BLOCK];
    __shared__ double s_prb[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        int blkj_blc;
        int blkj;
        int blki;
        int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        for (int iter = 0; (iter < 50); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_r_new[0];
                s_pra[threadIdx.x] = 0;
                s_prc[threadIdx.x] = 0;
                s_prb[threadIdx.x] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
                k_tg[global_id] = 0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
            }
            __threadfence();
            if (global_id == 0)
            {
                k_r0[0]=k_r_new[0];
                signal_dot[0] = tilem; // 问题在signal_dot
                signal_dot1[0] = 0;
                k_pra[0] = 0;
                k_tmp1[0] = 0;
                k_tmp2[0] = 0;
                signal_final[0] = 0;
                k_residual[0] = 0;
                k_r_new[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicSub(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != 0);
                

                if ((lane_id < BLOCK_SIZE))
                {
                    atomicAdd(k_pra, (d_y_d[blki_blc * BLOCK_SIZE + lane_id] * k_rh[blki_blc * BLOCK_SIZE + lane_id]));
                }
                
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_pra[local_warp_id] = s_snew[local_warp_id] / k_pra[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_sg[blki_blc * BLOCK_SIZE + lane_id] = k_rg[blki_blc * BLOCK_SIZE + lane_id] - s_pra[local_warp_id] * d_y_d[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (k_sg[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&k_tg[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != d_ori_block_signal[blki_blc]);
                if ((lane_id < BLOCK_SIZE))
                {
                    atomicAdd(k_tmp1, (k_tg[blki_blc * BLOCK_SIZE + lane_id] * k_sg[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                    atomicAdd(k_tmp2, (k_tg[blki_blc * BLOCK_SIZE + lane_id] * k_tg[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                }
                if ((lane_id == 0))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);
                if (lane_id == 0)
                {
                    s_prc[local_warp_id] = k_tmp1[0] / k_tmp2[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_x[blki_blc * BLOCK_SIZE + lane_id] = k_x[blki_blc * BLOCK_SIZE + lane_id] + s_pra[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]+s_prc[local_warp_id]*k_sg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                    k_rg[blki_blc * BLOCK_SIZE + lane_id] = k_sg[blki_blc * BLOCK_SIZE + lane_id] - s_prc[local_warp_id]*k_tg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if ((lane_id < BLOCK_SIZE))
                {
                    atomicAdd(k_residual, (k_rg[blki_blc * BLOCK_SIZE + lane_id] * k_rg[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                    atomicAdd(k_r_new, (k_rg[blki_blc * BLOCK_SIZE + lane_id] * k_rh[blki_blc * BLOCK_SIZE + lane_id]));
                    __threadfence();
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot1, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot1[0] != tilem);
                if (lane_id == 0)
                {
                    s_prb[local_warp_id] = (k_r_new[0]*s_pra[local_warp_id]) / (k_r0[0]*s_prc[local_warp_id]);
                }
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                {
                    d_x_d[blki_blc * BLOCK_SIZE + lane_id]=k_rg[blki_blc * BLOCK_SIZE + lane_id]+(s_prb[local_warp_id])*(d_x_d[blki_blc * BLOCK_SIZE + lane_id]-(s_prc[local_warp_id]*d_y_d[blki_blc * BLOCK_SIZE + lane_id]));
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != 0);
        }
    }
}



__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                         int *d_tile_ptr,
                                                         int *d_tile_columnidx,
                                                         unsigned char *d_csr_compressedIdx,
                                                         double *d_Blockcsr_Val_d,
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
                                                         unsigned char *d_Tile_csr_Col,
                                                         int *d_block_signal,
                                                         int *signal_dot,
                                                         int *signal_final,
                                                         int *signal_dot1,
                                                         int *d_ori_block_signal,
                                                         double *k_rh,
                                                         double *k_pra,
                                                         double *k_sg,
                                                         double *k_rg,
                                                         double *k_tg,
                                                         double *k_tmp1,
                                                         double *k_tmp2,
                                                         double *k_x,
                                                         double *k_residual,
                                                         double *k_r_new,
                                                         double *k_r0,
                                                         int *d_balance_tile_ptr,
                                                         int *d_row_each_block,
                                                         int *d_index_each_block,
                                                         int balance_row,
                                                         int *d_non_each_block_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot3[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot3_val = &s_dot3[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot4[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot4_val = &s_dot4[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot5[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot5_val = &s_dot5[local_warp_id * BLOCK_SIZE];
    
    
    
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_pra[WARP_PER_BLOCK];
    __shared__ double s_prc[WARP_PER_BLOCK];
    __shared__ double s_prb[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        int blkj_blc;
        int blkj;
        int blki;
        int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        int index_dot;
        int offset=blki_blc * BLOCK_SIZE;
        for (int iter = 0; (iter < 50); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_r_new[0];
                s_pra[threadIdx.x] = 0;
                s_prc[threadIdx.x] = 0;
                s_prb[threadIdx.x] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
                k_tg[global_id] = 0;
            }
            __threadfence();
            if (lane_id < BLOCK_SIZE)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
                s_dot3_val[lane_id] = 0.0;
                s_dot4_val[lane_id] = 0.0;
                s_dot5_val[lane_id] = 0.0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
            }
            __threadfence();
            if (global_id == 0)
            {
                k_r0[0]=k_r_new[0];
                signal_dot[0] = tilem; // 问题在signal_dot
                signal_dot1[0] = 0;
                k_pra[0] = 0;
                k_tmp1[0] = 0;
                k_tmp2[0] = 0;
                signal_final[0] = 0;
                k_residual[0] = 0;
                k_r_new[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicSub(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != 0);
                
                index_dot=offset + lane_id;
                // if ((lane_id < BLOCK_SIZE))
                // {
                //     atomicAdd(k_pra, (d_y_d[blki_blc * BLOCK_SIZE + lane_id] * k_rh[blki_blc * BLOCK_SIZE + lane_id]));
                // }
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id]+=(d_y_d[index_dot] * k_rh[index_dot]);
                }
                __syncthreads();
                int i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot1[threadIdx.x] += s_dot1[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_pra, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_pra[local_warp_id] = s_snew[local_warp_id] / k_pra[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_sg[blki_blc * BLOCK_SIZE + lane_id] = k_rg[blki_blc * BLOCK_SIZE + lane_id] - s_pra[local_warp_id] * d_y_d[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (k_sg[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&k_tg[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < tilem)
            {
                do
                {
                    __threadfence();
                } while (d_block_signal[blki_blc] != d_ori_block_signal[blki_blc]);
                // if ((lane_id < BLOCK_SIZE))
                // {
                //     atomicAdd(k_tmp1, (k_tg[blki_blc * BLOCK_SIZE + lane_id] * k_sg[blki_blc * BLOCK_SIZE + lane_id]));
                //     __threadfence();
                //     atomicAdd(k_tmp2, (k_tg[blki_blc * BLOCK_SIZE + lane_id] * k_tg[blki_blc * BLOCK_SIZE + lane_id]));
                //     __threadfence();
                // }
                index_dot=offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot2_val[lane_id]+=(k_tg[index_dot] * k_sg[index_dot]);
                    s_dot3_val[lane_id]+=(k_tg[index_dot] * k_tg[index_dot]);
                }
                __syncthreads();
                int i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                        s_dot3[threadIdx.x] += s_dot3[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_tmp1, s_dot2[0]);
                    atomicAdd(k_tmp2, s_dot3[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);
                if (lane_id == 0)
                {
                    s_prc[local_warp_id] = k_tmp1[0] / k_tmp2[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_x[blki_blc * BLOCK_SIZE + lane_id] = k_x[blki_blc * BLOCK_SIZE + lane_id] + s_pra[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]+s_prc[local_warp_id]*k_sg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                    k_rg[blki_blc * BLOCK_SIZE + lane_id] = k_sg[blki_blc * BLOCK_SIZE + lane_id] - s_prc[local_warp_id]*k_tg[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);

                // if ((lane_id < BLOCK_SIZE))
                // {
                //     atomicAdd(k_residual, (k_rg[blki_blc * BLOCK_SIZE + lane_id] * k_rg[blki_blc * BLOCK_SIZE + lane_id]));
                //     __threadfence();
                //     atomicAdd(k_r_new, (k_rg[blki_blc * BLOCK_SIZE + lane_id] * k_rh[blki_blc * BLOCK_SIZE + lane_id]));
                //     __threadfence();
                // }
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot4_val[lane_id]+=(k_rg[index_dot]*k_rg[index_dot]);
                    s_dot5_val[lane_id]+=(k_rg[index_dot]*k_rh[index_dot]);
                }
                __syncthreads();
                i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot4[threadIdx.x] += s_dot4[threadIdx.x + i];
                        s_dot5[threadIdx.x] += s_dot5[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_residual, s_dot4[0]);
                    atomicAdd(k_r_new, s_dot5[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot1, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot1[0] != tilem);
                if (lane_id == 0)
                {
                    s_prb[local_warp_id] = (k_r_new[0]*s_pra[local_warp_id]) / (k_r0[0]*s_prc[local_warp_id]);
                }
                __threadfence();
                if ((lane_id < BLOCK_SIZE))
                {
                    d_x_d[blki_blc * BLOCK_SIZE + lane_id]=k_rg[blki_blc * BLOCK_SIZE + lane_id]+(s_prb[local_warp_id])*(d_x_d[blki_blc * BLOCK_SIZE + lane_id]-(s_prc[local_warp_id]*d_y_d[blki_blc * BLOCK_SIZE + lane_id]));
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != 0);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce(int tilem, int tilenum, int rowA, int colA, int nnzA,
                                                         int *d_tile_ptr,
                                                         int *d_tile_columnidx,
                                                         unsigned char *d_csr_compressedIdx,
                                                         double *d_Blockcsr_Val_d,
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
                                                         unsigned char *d_Tile_csr_Col,
                                                         int *d_block_signal,
                                                         int *signal_dot,
                                                         int *signal_final,
                                                         int *signal_dot1,
                                                         int *d_ori_block_signal,
                                                         double *k_rh,
                                                         double *k_pra,
                                                         double *k_r1,
                                                         double *k_sg,
                                                         double *k_rg,
                                                         double *k_tg,
                                                         double *k_tmp1,
                                                         double *k_tmp2,
                                                         double *k_x,
                                                         double *k_residual,
                                                         double *k_r_new,
                                                         double *k_r0,
                                                         int *d_balance_tile_ptr,
                                                         int *d_row_each_block,
                                                         int *d_index_each_block,
                                                         int balance_row,
                                                         int *d_non_each_block_offset,
                                                         int vector_each_warp,
                                                         int vector_total
                                                         )
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_pra[WARP_PER_BLOCK];
    __shared__ double s_prc[WARP_PER_BLOCK];
    __shared__ double s_prb[WARP_PER_BLOCK];

    //点积reduce数组
    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    __shared__ double s_dot3[WARP_PER_BLOCK * 32];
    double *s_dot3_val = &s_dot3[local_warp_id * 32];
    __shared__ double s_dot4[WARP_PER_BLOCK * 32];
    double *s_dot4_val = &s_dot4[local_warp_id * 32];
    __shared__ double s_dot5[WARP_PER_BLOCK * 32];
    double *s_dot5_val = &s_dot5[local_warp_id * 32];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        int blkj_blc;
        int blkj;
        int blki;
        int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;
        int virtual_lane_id = lane_id & 0x1;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int index_dot;
        int offset=blki_blc * vector_each_warp;
        int csrcol;
        int u;
        for (int iter = 0; (iter < 50); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_r_new[0];
                s_pra[threadIdx.x] = 0;
                s_prc[threadIdx.x] = 0;
                s_prb[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
                s_dot3_val[lane_id] = 0.0;
                s_dot4_val[lane_id] = 0.0;
                s_dot5_val[lane_id] = 0.0;
            }
            __syncthreads();
            __threadfence();
            // if (global_id < rowA)
            // {
            //     d_y_d[global_id] = 0;
            //     k_tg[global_id] = 0;
            // }
            // __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
            }
            __threadfence();
            if (global_id == 0)
            {
                k_r0[0]=k_r_new[0];
                signal_dot[0] = vector_total; // 问题在signal_dot
                signal_dot1[0] = 0;
                k_pra[0] = 0;
                k_tmp1[0] = 0;
                k_tmp2[0] = 0;
                signal_final[0] = 0;
                k_residual[0] = 0;
                k_r_new[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicSub(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < vector_total)
            {
                //下面这个等待保证准确性
                // for(u = 0; u < vector_each_warp*2; u++)
                // {
                //     int off=blki_blc * vector_each_warp*2;
                //     //index_dot=iter*d_ori_block_signal[(off + u)];
                //     do
                //     {
                //         __threadfence_system();
                //         //__threadfence();
                //     } while (d_block_signal[(off + u)] != 0);
                //     //while (d_block_signal[(off + u)] != index_dot);
                // }

                //下面这个等待保证性能
                for(u = 0; u < vector_each_warp; u++)
                {
                    int off=blki_blc * vector_each_warp*2;
                    //index_dot=iter*d_ori_block_signal[(offset + u)];
                    do
                    {
                        __threadfence_system();
                        //__threadfence();
                    }  while (d_block_signal[(off + u)] != 0);
                    //while (d_block_signal[(offset + u)] != index_dot);//同步写的有问题 不能保证bypass的准确性
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    // do
                    // {
                    //     __threadfence();
                    // } while (d_block_signal[(offset + u)] != 0);
                    index_dot=(offset + u) * 32 + lane_id;
                    //index_dot=index_dot_val[u]+lane_id;
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * k_rh[index_dot]);
                }
                __syncthreads(); // 一个block内的线程都执行到这里
                int i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot1[threadIdx.x] += s_dot1[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }

                if (threadIdx.x == 0)
                {
                    atomicAdd(k_pra, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_pra[local_warp_id] = s_snew[local_warp_id] / k_pra[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    k_sg[index_dot] = k_rg[index_dot] - s_pra[local_warp_id] * d_y_d[index_dot];
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        index_s = rj + shared_offset;
                       
                        sum_d = sum_d + (k_sg[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&k_tg[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < vector_total)
            {
                //下面这个等待保证准确性
                // for(u = 0; u < vector_each_warp*2; u++)
                // {
                //     int off=blki_blc * vector_each_warp*2;
                //     //index_dot=iter*d_ori_block_signal[(off + u)];
                //     do
                //     {
                //         __threadfence_system();
                //         //__threadfence();
                //     } while (d_block_signal[(off + u)] != 0);
                //     //while (d_block_signal[(off + u)] != index_dot);
                // }

                //下面这个等待保证性能
                for(u = 0; u < vector_each_warp; u++)
                {
                    int off=blki_blc * vector_each_warp*2;
                    //index_dot=iter*d_ori_block_signal[(offset + u)];
                    do
                    {
                        __threadfence_system();
                        //__threadfence();
                    }  while (d_block_signal[(off + u)] != d_ori_block_signal[(off + u)]);
                    //while (d_block_signal[(offset + u)] != index_dot);//同步写的有问题 不能保证bypass的准确性
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    s_dot2_val[lane_id]+=k_tg[index_dot]*k_sg[index_dot];
                    s_dot3_val[lane_id]+=k_tg[index_dot]*k_tg[index_dot];
                }
                __syncthreads();
                int i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot1[threadIdx.x + i];
                        s_dot3[threadIdx.x] += s_dot3[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_tmp1, s_dot2[0]);
                    atomicAdd(k_tmp2, s_dot3[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);
                if (lane_id == 0)
                {
                    s_prc[local_warp_id] = k_tmp1[0] / k_tmp2[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_pra[local_warp_id] * d_x_d[index_dot]+s_prc[local_warp_id]*k_sg[index_dot];
                    __threadfence();
                    k_rg[index_dot] = k_sg[index_dot] - s_prc[local_warp_id]*k_tg[index_dot];
                    __threadfence();
                }
                if (lane_id == 0)
                {
                    atomicSub(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    s_dot4_val[lane_id]+=(k_rg[index_dot]*k_rg[index_dot]);
                    s_dot5_val[lane_id]+=(k_rg[index_dot]*k_rh[index_dot]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot4[threadIdx.x] += s_dot4[threadIdx.x + i];
                        s_dot5[threadIdx.x] += s_dot5[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }

                if (threadIdx.x == 0)
                {
                    atomicAdd(k_residual, s_dot4[0]);
                    atomicAdd(k_r_new, s_dot5[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot1, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot1[0] != vector_total);
                 if (lane_id == 0)
                {
                    s_prb[local_warp_id] = (k_r_new[0]*s_pra[local_warp_id]) / (k_r0[0]*s_prc[local_warp_id]);
                }
                __threadfence();
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    d_x_d[index_dot]=k_rg[index_dot]+(s_prb[local_warp_id])*(d_x_d[index_dot]-(s_prc[local_warp_id]*d_y_d[index_dot]));
                    __threadfence();
                    d_y_d[index_dot]=0.0;
                    k_tg[index_dot]=0.0;
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicSub(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != 0);
        }
    }
}



int bicgstab(int *RowPtr, int *ColIdx, double *Val, double *rhs, double *x,
             int nnzR, char *filename, int n,int block_nnz) // bicg
{
    int nnz = nnzR;
    float *Val_Low = (float *)malloc(sizeof(float) * nnz);
    for (int i = 0; i < nnz; i++)
    {
        Val_Low[i] = (float)Val[i];
    }
    int colA = n;
    n = (n / BLOCK_SIZE) * BLOCK_SIZE;
    int rowA = n;
    rowA = (rowA / BLOCK_SIZE) * BLOCK_SIZE;
    Tile_matrix *matrix = (Tile_matrix *)malloc(sizeof(Tile_matrix));
    Tile_create(matrix,
                rowA, colA, nnzR,
                RowPtr,
                ColIdx,
                Val,
                Val_Low);
    int num_seg = ceil((double)rowA / BLOCK_SIZE);
    int tilenum = matrix->tilenum;
    int *ptroffset1 = (int *)malloc(sizeof(int) * tilenum);
    int *ptroffset2 = (int *)malloc(sizeof(int) * tilenum);
    memset(ptroffset1, 0, sizeof(int) * tilenum);
    memset(ptroffset2, 0, sizeof(int) * tilenum);
    MAT_VAL_TYPE *y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * rowA);
    MAT_VAL_TYPE *y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * n);
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
                 rowA, colA, nnz,
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
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    MAT_VAL_LOW_TYPE *Blockcsr_Val_Low = matrix->Blockcsr_Val_Low;
    unsigned char *Tile_csr_Col = matrix->Tile_csr_Col;
    unsigned char *csr_compressedIdx = matrix->csr_compressedIdx;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    int csrsize = matrix->csrsize;
    int csrptrlen = matrix->csrptrlen;

    int csr_csize = csrsize % 2 == 0 ? csrsize / 2 : csrsize / 2 + 1;
    // record last val
    int *vis_new = (int *)malloc(sizeof(int) * num_seg);
    memset(vis_new, 0, sizeof(int) * num_seg);
    int *d_vis_new;
    cudaMalloc((void **)&d_vis_new, num_seg * sizeof(int));

    int vis_new_size_32 = (num_seg / 32) + 1;
    unsigned int *d_vis_mix_32;
    cudaMalloc((void **)&d_vis_mix_32, vis_new_size_32 * sizeof(unsigned int));

    MAT_PTR_TYPE *d_tile_ptr;
    int *d_tile_columnidx;
    int *tile_rowidx = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_rowidx, 0, sizeof(int) * tilenum);
    int *d_tile_rowidx;
    cudaMalloc((void **)&d_tile_rowidx, tilenum * sizeof(int));
    //printf("tilenum=%d tilem=%d num_seg=%d rowA=%d\n", tilenum, tilem, num_seg, rowA);
    cudaMalloc((void **)&d_tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_tile_columnidx, tilenum * sizeof(int));


    cudaMemcpy(d_tile_ptr, tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_columnidx, tile_columnidx, tilenum * sizeof(int), cudaMemcpyHostToDevice);

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
    cudaMalloc((void **)&d_Blockcsr_Val_float, (csrsize) * sizeof(float));
    cudaMemcpy(d_Blockcsr_Val_float, Blockcsr_Val_Low, (csrsize) * sizeof(float), cudaMemcpyHostToDevice);
  

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
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
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
    int *block_signal = (int *)malloc(sizeof(int) * (tilem + 1));
    memset(block_signal, 0, sizeof(int) * (tilem + 1)); // 记录块数
    for (int blki = 0; blki < tilem; blki++)
    {
        int rowlength = blki == tilem - 1 ? rowA - (tilem - 1) * BLOCK_SIZE : BLOCK_SIZE;
        block_signal[blki] = matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki];
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


    // 用于无同步的负载均衡部分 重新分配每个warp需要计算的非零元
    int *non_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        // 记录每个块的非零元个数
    int *non_each_block_offset = (int *)malloc(sizeof(int) * (tilenum + 1)); // 用于shared memory的索引
    int *row_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));        // 记录每个块的行号
    int *index_each_block = (int *)malloc(sizeof(int) * (tilenum + 1));      // 排序前每个块的索引
    memset(non_each_block, 0, sizeof(int) * (tilenum + 1));
    memset(non_each_block_offset, 0, sizeof(int) * (tilenum + 1));
    memset(row_each_block, 0, sizeof(int) * (tilenum + 1));
    memset(index_each_block, 0, sizeof(int) * (tilenum + 1));
    int nnz_total = 0;
    for (int blki = 0; blki < tilem; blki++)
    {
        for (int blkj = tile_ptr[blki]; blkj < tile_ptr[blki + 1]; blkj++)
        {
            non_each_block[blkj] = matrix->blknnz[blkj + 1] - matrix->blknnz[blkj];
            nnz_total += non_each_block[blkj];
            row_each_block[blkj] = blki;
            index_each_block[blkj] = blkj;
            //printf("%d ",tile_columnidx[blkj]);
        }
        //printf("\n");
    }
    // 快排根据每个块非零元素的个数从小到大
    //quickSort(non_each_block, row_each_block, index_each_block, 0, tilenum - 1);
    //quickSort_col(non_each_block, row_each_block, index_each_block,tile_columnidx, 0, tilenum - 1);
    //printf("sort end\n");
    int *row_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));   // 记录每个块的行号
    int *index_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1)); // 排序前每个块的索引
    int *non_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(row_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(index_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(non_each_block_new, 0, sizeof(int) * (tilenum + 1));
    int each_block_nnz = block_nnz;

    printf("nnz_total=%d each_block_nnz=%d\n", nnz_total, each_block_nnz);
    int cnt = 0;
    int balance_row = 0;
    int index = 1;
    
    int block_per_warp=180;
    int i = 0;
    int j = tilenum - 1;
    cnt = 0;
    index = 1;
    int step = 0;
    int cnt_block1=0;
    // int nnz_list[12]={16,32,64,96,128,256,512,1024,2048,4096,nnzR/6912};
    // while(1)
    // {
    // for(int k=0;k<12;k++)
    // {
    // each_block_nnz=nnz_list[k];
    // i = 0;
    // j = tilenum - 1;
    // cnt = 0;
    // index = 1;
    // step = 0;
    // cnt_block1=0;
    while (i < j)
    {
        if (((non_each_block[i] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[i];
            i++;
            cnt_block1++;
        }
        else if (((non_each_block[i] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            i++;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
        if (((non_each_block[j] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[j];
            j--;
            cnt_block1++;
        }
        else if (((non_each_block[j] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            j--;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
    }
    // if(index<6912)
    // break;
    // }
    // printf("index=%d\n");
    // if(index<6912)
    // break;
    // block_per_warp=block_per_warp*2;
    // }
    printf("index=%d each_block_nnz=%d\n", index, each_block_nnz);
    int vector_each_warp_16;
    int vector_total_16;
    int vector_each_warp_32;
    int vector_total_32;
    if (index < tilem)
    {
        printf("index=%d tilem=%d\n", index, tilem);
        vector_each_warp_16 = ceil((double)(tilem) / (double)(index));
        vector_total_16 = tilem / vector_each_warp_16;
        printf("index=%d tilem=%d vector_each_warp=%d vector_total=%d\n", index, tilem, vector_each_warp_16, vector_total_16);
        int tilem_32 = ceil((double)tilem / 2);
        //vector_each_warp_32=ceil((double)(tilem_32)/(double)(index));
        vector_each_warp_32 = vector_each_warp_16*2;
        vector_total_32 = tilem_32 / vector_each_warp_32;
        vector_total_32 = (vector_total_32/WARP_PER_BLOCK)*WARP_PER_BLOCK;
        //if(vector_total_32%2!=0)
        //vector_total_32-=1;
        printf("index=%d tilem=%d vector_each_warp_32=%d vector_total_32=%d\n", index, tilem, vector_each_warp_32, vector_total_32);
    }
    // if(index<tilem||index>6912) return;
    if (index > 6912||index==0||tilem==0)
        return;
    int *balance_tile_ptr_new = (int *)malloc(sizeof(int) * (index + 1));
    memset(balance_tile_ptr_new, 0, sizeof(int) * (index + 1));
    int *balance_tile_ptr_shared_end = (int *)malloc(sizeof(int) * (index + 1));
    memset(balance_tile_ptr_shared_end, 0, sizeof(int) * (index + 1));
    i = 0;
    j = tilenum - 1;
    cnt = 0;
    index = 1;
    step = 0;
    cnt_block1=0;
    while (i < j)
    {
        if (((non_each_block[i] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[i];
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[i] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
         if (((non_each_block[j] + cnt) < each_block_nnz)&&((cnt_block1+1)<block_per_warp))
        {
            cnt += non_each_block[j];
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[j] + cnt) >= each_block_nnz)||((cnt_block1+1)>=block_per_warp))
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1=0;
        }
        if (i == j)
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            step++;
            balance_tile_ptr_new[index] = step;
        }
        if (i > j)
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            balance_tile_ptr_new[index] = step;
        }
        // printf("i=%d j=%d step=%d\n",i,j,step);
    }
    int *d_balance_tile_ptr_new;
    cudaMalloc((void **)&d_balance_tile_ptr_new, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_new, balance_tile_ptr_new, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_row_each_block;
    int *d_index_each_block;
    cudaMalloc((void **)&d_row_each_block, sizeof(int) * (tilenum + 1));
    cudaMalloc((void **)&d_index_each_block, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_row_each_block, row_each_block_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_each_block, index_each_block_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    // 双指针的划分方式
    int cnt_block = 0;
    int cnt_nnz = 0;
    for (int i = 0; i <= index; i++)
    {
        balance_tile_ptr_shared_end[i] = balance_tile_ptr_new[i];
    }
    int shared_nnz_each_block=256;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            int blkj=index_each_block_new[j];
            if (j == balance_tile_ptr_new[i])
                non_each_block_offset[j] = 0;
            cnt_nnz += non_each_block_new[j];
            cnt_block++;
            if (j != balance_tile_ptr_new[i] && cnt_nnz <=shared_nnz_each_block)
            {
                non_each_block_offset[j] = non_each_block_new[j - 1];
                non_each_block_offset[j] += non_each_block_offset[j - 1];
            }
            if (cnt_nnz > shared_nnz_each_block)
            {
                balance_tile_ptr_shared_end[i + 1] = j;
                break;
            }
        }
        //printf("\n");
        //printf("i=%d nnz_each_block=%d block_num=%d\n",i,cnt_nnz,balance_tile_ptr_new[i+1]-balance_tile_ptr_new[i]);
    }
    // 验证balance_tile_ptr_shared_end
    int cnt_nnz_shared = 0;
    int cnt_nnz_total = 0;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        cnt_nnz_shared = 0;
        cnt_nnz_total = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz_total += non_each_block_new[j];
        }
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_shared_end[i + 1]; j++)
        {
            cnt_nnz_shared += non_each_block_new[j];
        }
        for (int j = balance_tile_ptr_shared_end[i + 1]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz += non_each_block_new[j];
        }
        // printf("cnt_nnz_shared=%d cnt_nnz=%d cnt_nnz_total=%d\n",cnt_nnz_shared,cnt_nnz,cnt_nnz_total);
    }

    printf("cnt_block=%d tilenum=%d tilem=%d\n", cnt_block, tilenum,tilem);
    int *d_non_each_block_offset;
    cudaMalloc((void **)&d_non_each_block_offset, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_non_each_block_offset, non_each_block_offset, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);

    int *d_balance_tile_ptr_shared_end;
    cudaMalloc((void **)&d_balance_tile_ptr_shared_end, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_shared_end, balance_tile_ptr_shared_end, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_block_signal;
    cudaMalloc((void **)&d_block_signal, sizeof(int) * (tilem + 1));
    int *signal_dot;
    cudaMalloc((void **)&signal_dot, sizeof(int));
    int *signal_final;
    cudaMalloc((void **)&signal_final, sizeof(int));
    int *signal_final1;
    cudaMalloc((void **)&signal_final1, sizeof(int));
    double *k_threshold;
    cudaMalloc((void **)&k_threshold, sizeof(double));
    int *d_ori_block_signal;
    cudaMalloc((void **)&d_ori_block_signal, sizeof(int) * (tilem + 1));
    cudaMemcpy(d_block_signal, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ori_block_signal, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
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
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    checkCudaErrors(cublasStatus);
    double time_cg = 0;
    double time_spmv = 0;
    double time_sptrsv = 0;
    struct timeval t1, t2, t3, t4, t5, t6;
    double *rg, *rh, *pg, *ph, *sg, *sh, *tg, *vg, *tp;
    double *k_rg, *k_rh, *k_pg, *k_ph, *k_sg, *k_sh, *k_tg, *k_vg, *k_tp;
    float  *k_vg_float;
    float  *k_tg_float;
    double *k_pg_last;
    double *k_sg_last;
    float *k_pg_float,*k_sg_float;
    double r0 = 0, r1 = 0, pra = 0, prb = 0, prc = 0;
    double residual, err_rel = 0;
    double *k_r0, *k_r1, *k_pra, *k_prb, *k_prc;
    double *k_residual, *k_err_rel;
    double *k_x;
    double *k_tmp1, *k_tmp2;
    //int i; 
    int retval = 0;
    int itr = 0.;
    double tol = 1e-5;
    int maxits = 50;
    double *x_last = (double *)malloc(sizeof(double) * (n + 1));
    int nthreads = 8;
    for (int i = 0; i < n; i++)
    {
        rhs[i] = 0;
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            rhs[i] += Val[j];
        }
    }
    rg = (double *)malloc(n * sizeof(double));
    rh = (double *)malloc(n * sizeof(double));
    pg = (double *)malloc(n * sizeof(double));
    ph = (double *)malloc(n * sizeof(double));
    sg = (double *)malloc(n * sizeof(double));
    sh = (double *)malloc(n * sizeof(double));
    tg = (double *)malloc(n * sizeof(double));
    vg = (double *)malloc(n * sizeof(double));
    tp = (double *)malloc(n * sizeof(double));
    cudaMalloc((void **)&k_rg, sizeof(double) * n);
    cudaMalloc((void **)&k_rh, sizeof(double) * n);
    cudaMalloc((void **)&k_pg, sizeof(double) * n);
    cudaMalloc((void **)&k_pg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_sg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_pg_last, sizeof(double) * n);
    cudaMalloc((void **)&k_ph, sizeof(double) * n);
    cudaMalloc((void **)&k_sg, sizeof(double) * n);
    cudaMalloc((void **)&k_sg_last, sizeof(double) * n);
    cudaMalloc((void **)&k_sh, sizeof(double) * n);
    cudaMalloc((void **)&k_tg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_tg_float, sizeof(float) * n);
    cudaMalloc((void **)&k_tp, sizeof(double) * n);
    cudaMalloc((void **)&k_x, sizeof(double) * n);

    cudaMalloc((void **)&k_r0, sizeof(double));
    cudaMalloc((void **)&k_r1, sizeof(double));
    double *k_r_new;
    cudaMalloc((void **)&k_r_new, sizeof(double));
    cudaMemset(k_r_new,0,sizeof(double));
    cudaMalloc((void **)&k_pra, sizeof(double));
    cudaMalloc((void **)&k_prb, sizeof(double));
    cudaMalloc((void **)&k_prc, sizeof(double));
    cudaMalloc((void **)&k_residual, sizeof(double));
    cudaMalloc((void **)&k_err_rel, sizeof(double));
    cudaMalloc((void **)&k_tmp1, sizeof(double));
    cudaMalloc((void **)&k_tmp2, sizeof(double));
    int *k_findrm, *k_colm;
    double *k_val;
    cudaMalloc((void **)&k_findrm, sizeof(int) * (n + 1));
    cudaMemcpy(k_findrm, RowPtr, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_colm, sizeof(int) * (nnzR));
    cudaMemcpy(k_colm, ColIdx, sizeof(int) * (nnzR), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&k_val, sizeof(double) * (nnzR));
    cudaMemcpy(k_val, Val, sizeof(double) * (nnzR), cudaMemcpyHostToDevice);
    mv(n, RowPtr, ColIdx, Val, x, tp);
    for (i = 0; i < n; i++)
        rg[i] = rhs[i] - tp[i];
    for (i = 0; i < n; i++)
    {
        rh[i] = rg[i];
        sh[i] = ph[i] = 0.;
    }
    int *vis_pg = (int *)malloc(sizeof(int) * n);
    int *vis_sg = (int *)malloc(sizeof(int) * n);
    residual = err_rel = itsol_norm(rg, n, nthreads);
    tol = residual * fabs(tol);
    // int cnt_pg=0;
    for (i = 0; i < n; i++)
        pg[i] = rg[i];
    r1 = itsol_dot(rg, rh, n, nthreads);
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);
    cudaMemcpy(k_pg, pg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rg, rg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_rh, rh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sh, sh, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_ph, ph, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_sg, sg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tg, tg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_vg, vg, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_tp, tp, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_x, x, sizeof(double) * (n), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r0, &r0, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r1, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_r_new, &r1, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_pra, &pra, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prb, &prb, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prc, &prc, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_residual, &residual, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_err_rel, &err_rel, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    //for (itr = 0; itr < maxits; itr++)
    {
        //scalarassign(k_r0, k_r1);
        // cudaMemset(k_vg, 0, n * sizeof(double));
        // cudaMemset(k_tg, 0, n * sizeof(double));
        int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
        if(index>=tilem)
        {
            int tilem_new=(tilem/WARP_PER_BLOCK+2)*WARP_PER_BLOCK;
            int re_size=(tilem_new)*BLOCK_SIZE;
            printf("tilem=%d tilem_new=%d tilem/WARP_PER_BLOCK=%d tilem_new/WARP_PER_BLOCK=%d\n",tilem,tilem_new,tilem/WARP_PER_BLOCK,tilem_new/WARP_PER_BLOCK);
            int *d_block_signal_new;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            double *k_vg_new;
            cudaMalloc((void **)&k_vg_new, sizeof(double) * re_size);
            double *k_pg_new;
            cudaMalloc((void **)&k_pg_new, sizeof(double) * re_size);
            cudaMemset(k_pg_new, 0,  re_size* sizeof(double));
            cudaMemcpy(k_pg_new, k_pg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_rh_new;
            cudaMalloc((void **)&k_rh_new, sizeof(double) * re_size);
            cudaMemcpy(k_rh_new, k_rh, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_sg_new;
            cudaMalloc((void **)&k_sg_new, sizeof(double) * re_size);
            cudaMemcpy(k_sg_new, k_sg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_rg_new;
            cudaMalloc((void **)&k_rg_new, sizeof(double) * re_size);
            cudaMemcpy(k_rg_new, k_rg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_tg_new;
            cudaMalloc((void **)&k_tg_new, sizeof(double) * re_size);
            cudaMemcpy(k_tg_new, k_tg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_x_new;
            cudaMalloc((void **)&k_x_new, sizeof(double) * re_size);
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);

            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            //printf("index>tilem\n");
            // if(index==tilem)
            // index=tilem+1;
            // stir_spmv_cuda_kernel_newcsr_nnz_balance<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
            //                                                                                   d_tile_ptr, d_tile_columnidx,
            //                                                                                   d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                   d_ptroffset1, d_ptroffset2,
            //                                                                                   rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                   k_pg, k_vg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal,
            //                                                                                   signal_dot, signal_final, signal_final1, d_ori_block_signal,
            //                                                                                   k_rh,k_pra,k_r1,k_sg,k_rg,k_tg,k_tmp1,k_tmp2,k_x,k_residual,k_r_new,k_r0,
            //                                                                                   d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset);
            stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_pg_new, k_vg_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                              k_rh_new,k_pra,k_sg_new,k_rg_new,k_tg_new,k_tmp1,k_tmp2,k_x_new,k_residual,k_r_new,k_r0,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset);
            cudaDeviceSynchronize();
            gettimeofday(&t4, NULL);
            time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
            cudaFree(d_ori_block_signal_new);
            cudaFree(d_ori_block_signal_new);
            cudaFree(k_vg_new);
            cudaFree(k_pg_new);
            cudaFree(k_rh_new);
            cudaFree(k_sg_new);
            cudaFree(k_rg_new);
            cudaFree(k_tg_new);
            cudaFree(k_x_new);
        }
        else
        {
            printf("index<tilem\n");
            if(vector_each_warp_32*vector_total_32*32>rowA)
            {
                printf("%d %d %d\n",rowA,(vector_each_warp_32*vector_total_32*32),rowA/BLOCK_SIZE);
                rowA=vector_each_warp_32*vector_total_32*32;
                printf("%d %d %d %d\n",rowA,(vector_each_warp_32*vector_total_32*32),rowA/BLOCK_SIZE,tilem);
                printf("above\n");
            }
            int tilem_new=rowA/BLOCK_SIZE;
            int *d_block_signal_new;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            double *k_vg_new;
            cudaMalloc((void **)&k_vg_new, sizeof(double) * rowA);
            double *k_pg_new;
            cudaMalloc((void **)&k_pg_new, sizeof(double) * rowA);
            cudaMemset(k_pg_new, 0,  rowA* sizeof(double));
            cudaMemcpy(k_pg_new, k_pg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_rh_new;
            cudaMalloc((void **)&k_rh_new, sizeof(double) * rowA);
            cudaMemcpy(k_rh_new, k_rh, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_sg_new;
            cudaMalloc((void **)&k_sg_new, sizeof(double) * rowA);
            cudaMemcpy(k_sg_new, k_sg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_rg_new;
            cudaMalloc((void **)&k_rg_new, sizeof(double) * rowA);
            cudaMemcpy(k_rg_new, k_rg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_tg_new;
            cudaMalloc((void **)&k_tg_new, sizeof(double) * rowA);
            cudaMemcpy(k_tg_new, k_tg, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_x_new;
            cudaMalloc((void **)&k_x_new, sizeof(double) * rowA);
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            
            
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_pg_new, k_vg_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                              k_rh_new,k_pra,k_r1,k_sg_new,k_rg_new,k_tg_new,k_tmp1,k_tmp2,k_x_new,k_residual,k_r_new,k_r0,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,vector_each_warp_32,vector_total_32);
            cudaDeviceSynchronize();
            gettimeofday(&t4, NULL);
            time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        }
        //cublasDdot(cublasHandle, n, k_rh, 1, k_vg, 1, k_pra);
        //k_pra=k_r1/k_pra
        //scalardiv<<<1, 1>>>(k_pra, k_r1);//后续都写完这句放在无同步里面
        //yminus_mult<<<GridDim, BlockDim>>>(n, k_sg, k_rg, k_vg, k_pra);
        // cudaDeviceSynchronize();
        // gettimeofday(&t5, NULL);
        // stir_spmv_cuda_kernel_newcsr<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnz,
        //                                                               d_tile_ptr, d_tile_columnidx,
        //                                                               d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
        //                                                               d_ptroffset1, d_ptroffset2,
        //                                                               rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
        //                                                               k_sg, k_tg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col);

        // cudaDeviceSynchronize();
        // gettimeofday(&t6, NULL);
        // time_spmv += (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
        // cublasDdot(cublasHandle, n, k_tg, 1, k_sg, 1, k_tmp1);
        // cublasDdot(cublasHandle, n, k_tg, 1, k_tg, 1, k_tmp2);
        //scalardiv_new<<<1, 1>>>(k_prc, k_tmp1, k_tmp2);
        //yminus_mult_new<<<GridDim, BlockDim>>>(n, k_x, k_pg, k_sg, k_rg, k_tg, k_pra, k_prc);
        //cublasDdot(cublasHandle, n, k_rg, 1, k_rg, 1, k_residual);
        cudaMemcpy(&residual,k_residual,sizeof(double),cudaMemcpyDeviceToHost);
        //cublasDdot(cublasHandle, n, k_rg, 1, k_rh, 1, k_r1);//这里换个变量 放到无同步 不能用r1了
        //scalarassign(k_r1, k_r_new);
        //prb=r1*pra/r0*prc
        //scalardiv_five<<<1, 1>>>(k_prb, k_r1, k_pra, k_r0, k_prc);
        //yminus_final<<<GridDim, BlockDim>>>(n, k_pg, k_rg, k_prb, k_prc, k_vg);
        //printf("%d %e %e\n", itr, sqrt(residual) / err_rel,re1/ err_rel);
        printf("%d %e\n",itr,sqrt(residual));
        // if (sqrt(residual) <= tol)
        //     break;
    }
    cudaThreadSynchronize();
    gettimeofday(&t2, NULL);
    time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // double time_overall_serial = time_cg / itr;
    // double GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
    double Gflops_spmv= (2 * nnzR) / ((time_spmv/ itr/2)*pow(10, 6));
    double Gflops_bicg= (2 * nnzR) / ((time_cg/ itr) * pow(10, 6));
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (rhs[i] * rhs[i]);
    }
    double l2_norm = sqrt(residual) / sqrt(sum_ori);
    printf("time_bicg=%lf ms,time_spmv=%lf ms, l2_norm=%lf\n", time_cg, time_spmv,l2_norm);
    printf("Gflops_bicg=%lf,Gflops_spmv=%lf, l2_norm=%lf\n", Gflops_bicg, Gflops_spmv,l2_norm);
    char *s = (char *)malloc(sizeof(char) * 100);
    sprintf(s, "iter=%d,time_spmv=%.3f,time_bicg=%.3f,nnzR=%d,l2_norm=%e,norm=%e\n", itr, time_spmv, time_cg, nnzR,l2_norm,residual/err_rel);
    FILE *file1 = fopen("bykrylov_bicg_syncfree_a100.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return 0;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    fclose(file1);
    free(rg);
    free(rh);
    free(pg);
    free(ph);
    free(sg);
    free(sh);
    free(tg);
    free(tp);
    free(vg);
    if (itr >= maxits)
        retval = 1;
    return retval;
}
int main(int argc, char **argv)
{
    int n;
    char *filename = argv[1];
    int block_nnz = atoi(argv[2]);
    int m, n_csr, nnzR, isSymmetric;
    FILE *p = fopen(filename, "r");
    int *RowPtr;
    int *ColIdx;
    double *Val;
    read_Dmatrix_32(&m, &n_csr, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
    double *x1 = (double *)malloc(m * sizeof(double));
    int nn;
    double *rhs1 = (double *)malloc(m * sizeof(double));

    for (int i = 0; i < m; i++)
    {
        x1[i] = 0.0;
        rhs1[i] = 1;
        int cc;
    }
    bicgstab(RowPtr, ColIdx, Val, rhs1, x1, nnzR, filename, m,block_nnz);
    fclose(p);
    free(x1);
    free(rhs1);
    return 0;
}
