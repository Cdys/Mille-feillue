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
__global__ void device_convert_int8(double *x, int8_t *y, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
    {
        y[tid] = (int8_t)(x[tid]);
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
    {
        // printf("%e\n",(*den));
        *result = (*num) / (*den);
    }
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

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_mixed_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                                                     int *signal_final1,
                                                                                                     int *d_ori_block_signal,
                                                                                                     double *k_alpha,
                                                                                                     double *k_snew,
                                                                                                     double *k_x,
                                                                                                     double *k_r,
                                                                                                     double *k_sold,
                                                                                                     double *k_beta,
                                                                                                     double *k_threshold,
                                                                                                     int *d_balance_tile_ptr,
                                                                                                     int *d_row_each_block,
                                                                                                     int *d_index_each_block,
                                                                                                     int balance_row,
                                                                                                     int *d_non_each_block_offset,
                                                                                                     int *d_vis,
                                                                                                     float *d_Blockcsr_Val_f,
                                                                                                     half *d_Blockcsr_Val_h,
                                                                                                     int8_t *d_Blockcsr_Val_8,
                                                                                                     float *d_x_f,
                                                                                                     half *d_x_h,
                                                                                                     int8_t *d_x_8,
                                                                                                     float *d_y_f)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int vis_spmv[WARP_PER_BLOCK];
    int *vis_spmv_val = &vis_spmv[local_warp_id];

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
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
        int offset = blki_blc * BLOCK_SIZE;

        // for(int iter=1;(iter<=100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
                // s_dot2[threadIdx.x] = 0;
            }
            //__syncthreads();
            //__threadfence();
            if (lane_id < BLOCK_SIZE)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            if (lane_id == 0)
            {
                vis_spmv_val[lane_id] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
            }
            __threadfence();
            if (global_id == 0)
            // if(lane_id==0)
            {
                signal_dot[0] = tilem; // 问题在signal_dot
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                int colid = d_tile_columnidx[blkj];
                // if ((d_vis[colid] != 1))
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        if (d_vis[colid] == 0) // FP64
                        {
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d); // 这个原子加占了25%的寄存器 其他的原子加占了12.5%
                        }
                        else if (d_vis[colid] == 4) // FP32
                        {
                            sum_f = 0.0;
                            // sum_d = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                int csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_f += d_x_f[x_offset + csrcol] * (float)(d_Blockcsr_Val_d[csroffset + rj]);
                                sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                        else if (d_vis[colid] == 3) // FP16
                        {
                            sum_h = 0.0;
                            // sum_f = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                        // if (d_vis[colid] == 4) // FP8
                        else
                        {
                            // sum_f = 0.0;
                            sum_8 = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                             for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                             {
                                 csrcol = d_Tile_csr_Col[csroffset + rj];
                                 sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                             }
                            // packed one thread 2 int_8 结果不对
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_8 += d_x_8[csrcol + x_offset] * d_Blockcsr_Val_8[csroffset + rj];
                            //     // sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
                            //         // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
                            //     }
                            // }
                            // packed one thread 4 int_8 结果不对
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = (float)sum_8;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                    }
                }

                if (lane_id == 0)
                {
                    //atomicAdd(&d_block_signal[blki], 1);
                    atomicSub(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < tilem)
            {
                //index_dot = iter * d_ori_block_signal[blki_blc];
                do
                {
                    //__threadfence_system();
                    __threadfence();
                } 
                //while (d_block_signal[blki_blc] != index_dot); // 会导致卡住
                while (d_block_signal[blki_blc] != 0);
                index_dot = offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
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
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    d_vis[blki_blc] = 0;
                    //  检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                // 检测k_alpha是否更新完毕
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    double d_x_last = d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    //d_x_8[index_dot] = (int8_t)(d_x_8[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_d[index_dot]);

                    // if(fabs(d_x_last-d_x_d[index_dot])<=1e-5)
                    // {
                    //     //printf("iter=%d blki_blc=%d\n",iter,blki_blc);
                    //     d_vis[blki_blc]=1;
                    // }
                    double val = fabs(d_x_d[index_dot] - d_x_last);
                    // if (val > 1e-5)
                    // {
                    //     vis_spmv_val[0] = 1;
                    // }
                    // if (val > 1e-8)
                    // {
                    //     vis_spmv_val[0] = 1;
                    // }
                    if (val <= 1e-7 && val > 1e-8) // fp8
                    {
                        vis_spmv_val[0] = 2;
                    }
                    else if (val <= 1e-6 && val > 1e-7) // fp16
                    {
                        vis_spmv_val[0] = 3;
                    }
                    else if (val <= 1e-5 && val > 1e-6) // fp32
                    {
                        vis_spmv_val[0] = 4;
                    }
                    else if (val > 1e-5)
                    {
                        vis_spmv_val[0] = 1; // fp64
                    }
                }
                __syncthreads();
                if (lane_id == 0)
                {

                    // switch (vis_spmv_val[0])
                    // {
                    // case 0:
                    //     d_vis[blki_blc] = 1;
                    //     break;
                    // case 2:
                    //     d_vis[blki_blc] = 2;
                    //     break;
                    // case 3:
                    //     d_vis[blki_blc] = 3;
                    //     break;
                    // case 4:
                    //     d_vis[blki_blc] = 4;
                    //     break;
                    // default:
                    //     break;
                    // }
                    // if (!vis_spmv_val[0])
                    //     d_vis[blki_blc] = 1;
                    if (vis_spmv_val[0] == 0) // bypass
                    {
                        d_vis[blki_blc] = 1;
                    }
                    if (vis_spmv_val[0] == 2) // fp8
                    {
                        d_vis[blki_blc] = 2;
                    }
                    if (vis_spmv_val[0] == 3) // fp16
                    {
                        d_vis[blki_blc] = 3;
                    }
                    if (vis_spmv_val[0] == 4) // fp32
                    {
                        d_vis[blki_blc] = 4;
                    }
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_shared_queue_mixed_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                                                                  int *signal_final1,
                                                                                                                  int *d_ori_block_signal,
                                                                                                                  double *k_alpha,
                                                                                                                  double *k_snew,
                                                                                                                  double *k_x,
                                                                                                                  double *k_r,
                                                                                                                  double *k_sold,
                                                                                                                  double *k_beta,
                                                                                                                  double *k_threshold,
                                                                                                                  int *d_balance_tile_ptr,
                                                                                                                  int *d_row_each_block,
                                                                                                                  int *d_index_each_block,
                                                                                                                  int balance_row,
                                                                                                                  int *d_non_each_block_offset,
                                                                                                                  int *d_balance_tile_ptr_shared_end,
                                                                                                                  int *d_vis,
                                                                                                                  float *d_Blockcsr_Val_f,
                                                                                                                  half *d_Blockcsr_Val_h,
                                                                                                                  int8_t *d_Blockcsr_Val_8,
                                                                                                                  float *d_x_f,
                                                                                                                  half *d_x_h,
                                                                                                                  int8_t *d_x_8,
                                                                                                                  float *d_y_f,
                                                                                                                  int shared_num)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    __shared__ int vis_spmv[WARP_PER_BLOCK];
    int *vis_spmv_val = &vis_spmv[local_warp_id];

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjshared_end = d_balance_tile_ptr_shared_end[blki_blc + 1];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
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
        int offset = blki_blc * BLOCK_SIZE;

        const int nnz_per_warp = 256;
        __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
        double *s_data_val = &s_data[local_warp_id * nnz_per_warp];
        __shared__ float s_data_float[nnz_per_warp * WARP_PER_BLOCK];
        float *s_data_float_val = &s_data_float[local_warp_id * nnz_per_warp];
        __shared__ half s_data_half[nnz_per_warp * WARP_PER_BLOCK];
        half *s_data_half_val = &s_data_half[local_warp_id * nnz_per_warp];
        __shared__ int8_t s_data_int8[nnz_per_warp * WARP_PER_BLOCK];
        int8_t *s_data_int8_val = &s_data_int8[local_warp_id * nnz_per_warp];
        for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
        {
            blkj = d_index_each_block[blkj_blc];
            shared_offset = d_non_each_block_offset[blkj_blc];
            csroffset = d_ptroffset1[blkj];
            s1 = d_nonzero_row_new[blkj];
            s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    index_s = rj + shared_offset;
                    s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
                    s_data_float_val[index_s] = (float)(s_data_val[index_s]);
                    s_data_half_val[index_s] = (half)(s_data_val[index_s]);
                    s_data_int8_val[index_s] = (int8_t)(s_data_val[index_s]);
                }
            }
            //  if (lane_id == 0)
            // {
            //     atomicAdd(signal_final1, 1); // 这个同步保证shared memory的正确性 速度会变慢
            // }
        }
        // do
        // {
        //     __threadfence();
        // } while (signal_final1[0] != shared_num); // 这个同步保证shared memory的正确性 速度会变慢
        //__threadfence();
        //__syncthreads();

        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 1; (iter <= 1000); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            //__syncthreads();
            //__threadfence();
            if (lane_id < BLOCK_SIZE)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            if (lane_id == 0)
            {
                vis_spmv_val[lane_id] = 0;
            }
            __syncthreads();
            // if (global_id < rowA)
            // {
            //     d_y_d[global_id] = 0;
            // }
            __threadfence();

            if (global_id == 0)
            // if(lane_id==0)
            {
                signal_dot[0] = tilem; // 问题在signal_dot
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                //if ((d_vis[colid] != 1))
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        shared_offset = d_non_each_block_offset[blkj_blc];
                        if (d_vis[colid] == 0) // FP64
                        {
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * s_data_val[index_s]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d); // 这个原子加占了25%的寄存器 其他的原子加占了12.5%
                        }
                        else if (d_vis[colid] == 4) // FP32
                        {
                            sum_f = 0.0;
                            // sum_d = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_f += d_x_f[x_offset + csrcol] * (float)s_data_val[index_s];
                                sum_f += d_x_f[x_offset + csrcol] * s_data_float_val[index_s];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                        else if (d_vis[colid] == 3) // FP16
                        {
                            sum_h = 0.0;
                            // sum_f = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      index_s = rj + shared_offset;
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],s_data_half_val[index_s],sum_h);
                            //      //sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                index_s = rj + shared_offset;
                                sum_h = __hfma(d_x_h[x_offset + csrcol], s_data_half_val[index_s], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)s_data_val[index_s], sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], s_data_half_val[index_s + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)s_data_val[index_s+1], sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                        //if (d_vis[colid] == 4) // FP8
                        else
                        {
                            // sum_f = 0.0;
                            sum_8 = 0.0;
                            int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                             for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                             {
                                 index_s = rj + shared_offset;
                                 csrcol = d_Tile_csr_Col[csroffset + rj];
                                 //sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                                 sum_8 += d_x_8[x_offset + csrcol]*s_data_int8_val[index_s];
                             }
                            // packed one thread 2 int_8 结果不对
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     index_s = rj + shared_offset;
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_8 += d_x_8[csrcol + x_offset] * s_data_int8_val[index_s];
                            //     // sum_8 += d_x_8[csrcol+x_offset] * (int8_t)s_data_val[index_s];
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * s_data_int8_val[index_s + 1];
                            //         // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1]+x_offset] * (int8_t)s_data_val[index_s+1];
                            //     }
                            // }
                            // packed one thread 4 int_8 结果不对
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = (float)sum_8;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            // 如果能都放在shared memory中 则不需要下面的循环
            //  for (blkj_blc = rowblkjshared_end; blkj_blc < rowblkjstop; blkj_blc++)
            //  {
            //      blkj = d_index_each_block[blkj_blc];
            //      blki = d_row_each_block[blkj_blc];
            //      int colid = d_tile_columnidx[blkj];
            //      if ((d_vis[colid] != 1))
            //      {
            //          x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
            //          csroffset = d_ptroffset1[blkj];
            //          s1 = d_nonzero_row_new[blkj];
            //          s2 = d_nonzero_row_new[blkj + 1];
            //          if (ri < s2 - s1)
            //          {
            //              ro = d_blockrowid_new[s1 + ri + 1];
            //              if (d_vis[colid] == 0) // FP64
            //              {
            //                  sum_d = 0.0;
            //                  for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
            //                  {
            //                      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
            //                  }
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d); // 这个原子加占了25%的寄存器 其他的原子加占了12.5%
            //              }
            //              if (d_vis[colid] == 4) // FP32
            //              {
            //                  sum_f = 0.0;
            //                  // sum_d = 0.0;
            //                  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
            //                  {
            //                      int csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      // sum_f += d_x_f[x_offset + csrcol] * (float)(d_Blockcsr_Val_d[csroffset + rj]);
            //                      sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
            //                  }
            //                  sum_d = (double)(sum_f);
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            //                  // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
            //              }
            //              if (d_vis[colid] == 3) // FP16
            //              {
            //                  sum_h = 0.0;
            //                  // sum_f = 0.0;
            //                  int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
            //                  // unpacked
            //                  //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
            //                  //  {
            //                  //      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                  //      sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
            //                  //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
            //                  //  }
            //                  // packed one thread 2 half
            //                  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
            //                  {
            //                      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
            //                      // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
            //                      if (rj + 1 < row_end)
            //                      {
            //                          sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
            //                          // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
            //                      }
            //                  }
            //                  sum_d = (double)(sum_h);
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            //                  // sum_f = __half2float(sum_h);
            //                  // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
            //              }
            //              if (d_vis[colid] == 4) // FP8
            //              {
            //                  //sum_f = 0.0;
            //                  sum_8 = 0.0;
            //                  int row_end = d_blockcsr_ptr_new[s1 + ri + 1];
            //                  // unpacked 结果正确
            //                  //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
            //                  //  {
            //                  //      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                  //      sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
            //                  //  }
            //                  // packed one thread 2 int_8 结果不对
            //                  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
            //                  {
            //                      csrcol = d_Tile_csr_Col[csroffset + rj];
            //                      sum_8 += d_x_8[csrcol+x_offset] * d_Blockcsr_Val_8[csroffset + rj];
            //                      //sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
            //                      if (rj + 1 < row_end)
            //                      {
            //                          sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1]+x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
            //                          //sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
            //                      }
            //                  }
            //                  // packed one thread 4 int_8 结果不对
            //                  //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
            //                  //  {
            //                  //      int csrcol = d_Tile_csr_Col[csroffset + rj];
            //                  //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
            //                  //      if(rj+1<row_end)
            //                  //      {
            //                  //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
            //                  //      }
            //                  //      if(rj+2<row_end)
            //                  //      {
            //                  //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
            //                  //      }
            //                  //  }
            //                  sum_d = (double)(sum_8);
            //                  atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
            //                  // sum_f = (float)sum_8;
            //                  // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
            //              }
            //          }
            //      }
            //      if (lane_id == 0)
            //      {
            //          atomicAdd(&d_block_signal[blki], 1);
            //      }
            //  }

            if (blki_blc < tilem)
            {
                index_dot = iter * d_ori_block_signal[blki_blc];
                do
                {
                    __threadfence_system();
                    //__threadfence();
                } while (d_block_signal[blki_blc] != index_dot);
                // while (d_block_signal[blki_blc] != 0);

                index_dot = offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
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
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();
                if ((lane_id == 0))
                {
                    //  检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                // 检测k_alpha是否更新完毕
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (BLOCK_SIZE * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }

                if ((lane_id < BLOCK_SIZE))
                {
                    double d_x_last = d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    //d_x_8[index_dot] = (int8_t)(d_x_8[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_d[index_dot]);
                    d_y_d[index_dot] = 0.0;
                    // if(iter%5==0)
                    {
                        double val = fabs(d_x_d[index_dot] - d_x_last);
                        // if (val > 1e-8)
                        // {
                        //     vis_spmv_val[0] = 1;
                        // }
                        if (val <= 1e-7 && val > 1e-8) // fp8
                        {
                            vis_spmv_val[0] = 2;
                        }
                        else if (val <= 1e-6 && val > 1e-7) // fp16
                        {
                            vis_spmv_val[0] = 3;
                        }
                        else if (val <= 1e-5 && val > 1e-6) // fp32
                        {
                            vis_spmv_val[0] = 4;
                        }
                        else if (val > 1e-5)
                        {
                            vis_spmv_val[0] = 1; // fp64
                        }
                    }
                }
                __syncthreads();
                // if (lane_id == 0&&iter%5==0)
                if (lane_id == 0)
                {
                    d_vis[blki_blc] = 0;
                    if (vis_spmv_val[0] == 0) // bypass
                    {
                        d_vis[blki_blc] = 1;
                    }
                    if (vis_spmv_val[0] == 2) // fp8
                    {
                        d_vis[blki_blc] = 2;
                        // printf("fp8\n");
                    }
                    if (vis_spmv_val[0] == 3) // fp16
                    {
                        d_vis[blki_blc] = 3;
                        // printf("fp16\n");
                    }
                    if (vis_spmv_val[0] == 4) // fp32
                    {
                        d_vis[blki_blc] = 4;
                        // printf("fp32 %d\n",iter);
                    }
                     atomicAdd(signal_final, 1);
                }
                // if (lane_id == 0)
                // {
                //     atomicAdd(signal_final, 1);
                // }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != tilem);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_mix_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                                                                   int *signal_final1,
                                                                                                                   int *d_ori_block_signal,
                                                                                                                   double *k_alpha,
                                                                                                                   double *k_snew,
                                                                                                                   double *k_x,
                                                                                                                   double *k_r,
                                                                                                                   double *k_sold,
                                                                                                                   double *k_beta,
                                                                                                                   double *k_threshold,
                                                                                                                   int *d_balance_tile_ptr,
                                                                                                                   int *d_row_each_block,
                                                                                                                   int *d_index_each_block,
                                                                                                                   int balance_row,
                                                                                                                   int *d_non_each_block_offset,
                                                                                                                   int vector_each_warp,
                                                                                                                   int vector_total,
                                                                                                                   int *d_vis,
                                                                                                                   double *d_x_d_last,
                                                                                                                   float *d_Blockcsr_Val_f,
                                                                                                                   half *d_Blockcsr_Val_h,
                                                                                                                   int8_t *d_Blockcsr_Val_8,
                                                                                                                   float *d_x_f,
                                                                                                                   half *d_x_h,
                                                                                                                   int8_t *d_x_8,
                                                                                                                   float *d_y_f)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    __shared__ int row_begin[WARP_PER_BLOCK];
    __shared__ int row_end[WARP_PER_BLOCK];
    // int *row_begin_val = &row_begin[local_warp_id];
    // int *row_end_val = &row_end[local_warp_id];
    __shared__ int vis_spmv[WARP_PER_BLOCK * 2];
    int *vis_spmv_val = &vis_spmv[local_warp_id * 2];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        // if(lane_id==0)
        //{
        // row_begin_val[0] = d_balance_tile_ptr[blki_blc];
        // row_end_val[0] = d_balance_tile_ptr[blki_blc + 1];
        //}
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
        int blkj_blc;
        int blkj;
        int blki;
        // int shared_offset;
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
        int offset = blki_blc * vector_each_warp;
        int iter;
        int u;
        int row_end;
        // if(lane_id<vector_each_warp)
        //     index_dot_val[lane_id]=d_ori_block_signal[(offset + lane_id)];
        // __syncthreads();
        //__threadfence();

        // for(int iter=1;(iter<=100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            __syncthreads();
            // //__threadfence();
            if (lane_id < 2)
            {
                vis_spmv_val[lane_id] = 0;
            }
            __syncthreads();
            // if (global_id < rowA)
            // {
            //     d_y_d[global_id] = 0.0;
            //     // d_y_f[global_id] = 0.0;
            // }
            __threadfence();
            //__threadfence_system();
            // if (global_id < tilem)
            // {
            //     d_block_signal[global_id] = d_ori_block_signal[global_id];
            // }
            // __threadfence();
            //__threadfence_system();
            if (global_id == 0)
            {
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
                // signal_final1[0] = 0;
            }
            __threadfence();
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            // for (blkj_blc = row_begin_val[0]; blkj_blc < row_end_val[0]; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc]; // 行号
                colid = d_tile_columnidx[blkj];
                // int flag=d_vis[colid];
                // if ((d_vis[colid] != 1))
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        switch (d_vis[colid])
                        {
                        // if (d_vis[colid] == 0) // FP64
                        case 0:
                            // case 1:
                            //{
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        //}
                        // else if (d_vis[colid] == 4) // FP32
                        case 4:
                            //{
                            sum_f = 0.0;
                            // sum_d = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_f += d_x_f[x_offset + csrcol] * (float)(d_Blockcsr_Val_d[csroffset + rj]);
                                sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // else if (d_vis[colid] == 3) // FP16
                        case 3:
                            //{
                            sum_h = 0.0;
                            // sum_f = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                                // sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            }
                            // packed one thread 2 half
                            // for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            // {
                            //     csrcol = d_Tile_csr_Col[csroffset + rj];
                            //     sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                            //     // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                            //     if (rj + 1 < row_end)
                            //     {
                            //         sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
                            //         // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                            //     }
                            // }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // if (d_vis[colid] == 2) // FP8
                        // else
                        case 2:
                            //{
                            // sum_f = 0.0;
                            sum_8 = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //  }
                            // packed one thread 2 int_8 结果不对
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[csrcol + x_offset] * d_Blockcsr_Val_8[csroffset + rj];
                                // sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                                if (rj + 1 < row_end)
                                {
                                    sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
                                    // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
                                }
                            }
                            // packed one thread 4 int_8 结果不对
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += d_x_8[csrcol+x_offset]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj+1]+x_offset]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj+2]+x_offset]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = (float)sum_8;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        default:
                            break;
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < vector_total)
            {
                // 下面这个等待保证bypass的准确性
                for (u = 0; u < vector_each_warp * 2; u++)
                {
                    int off = blki_blc * vector_each_warp * 2;
                    index_dot = iter * d_ori_block_signal[(off + u)];
                    do
                    {
                        //__threadfence_system();
                        __threadfence();
                    } while (d_block_signal[(off + u)] != index_dot);
                }

                // 下面这个等待保证性能
                //  for(u = 0; u < vector_each_warp; u++)
                //  {
                //      index_dot=iter*d_ori_block_signal[(offset + u)];
                //      do
                //      {
                //          __threadfence_system();
                //          //__threadfence();
                //      }
                //      while (d_block_signal[(offset + u)] != index_dot);//同步写的有问题 不能保证bypass的准确性
                //  }

                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
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
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();

                if ((lane_id == 0))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        // printf("k_alpha=%e\n",k_alpha[0]);
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                        // 把剩下块的点积算了
                    }
                }

                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    // printf("k_alpha=%e\n",k_alpha[0]);//k_alpha的值和不是无同步的对不上
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    // 下面为bypass检索
                    double d_x_last = d_x_d[index_dot];
                    // d_x_d_last[index_dot]=d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_y_d[index_dot] = 0.0;
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_8[index_dot]);
                    //d_x_8[index_dot] = (int8_t)(d_x_d[index_dot]);
                    d_vis[(index_dot / 16)] = 0;
                    // // if(fabs(d_x_last-d_x_d[index_dot])<=1e-8)
                    // // {
                    // //     d_vis[(index_dot/16)]=1;
                    // // }
                    double val = fabs(d_x_d[index_dot] - d_x_last);
                    // if (val > 1e-8)
                    // {
                    //     vis_spmv_val[(lane_id / 16)] = 1;
                    // }
                    if (val <= 1e-7 && val > 1e-8) // fp8
                    {
                        vis_spmv_val[(lane_id / 16)] = 2;
                    }
                    else if (val <= 1e-6 && val > 1e-7) // fp16
                    {
                        vis_spmv_val[(lane_id / 16)] = 3;
                    }
                    else if (val <= 1e-5 && val > 1e-6) // fp32
                    {
                        vis_spmv_val[(lane_id / 16)] = 4;
                    }
                    else if (val > 1e-5)
                    {
                        vis_spmv_val[(lane_id / 16)] = 1; // fp64
                    }
                    __syncthreads();
                    if (lane_id % 16 == 0)
                    {
                        switch (vis_spmv_val[lane_id / 16])
                        {
                        case 0:
                            d_vis[(index_dot / 16)] = 1;
                            break;
                        case 2:
                            d_vis[(index_dot / 16)] = 2;
                            break;
                        case 3:
                            d_vis[(index_dot / 16)] = 3;
                            break;
                        case 4:
                            d_vis[(index_dot / 16)] = 4;
                            break;
                        default:
                            break;
                        }
                    }
                    // if (lane_id == 0)
                    // {
                    //     // if (vis_spmv_val[0] == 0) // bypass
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 1;
                    //     // }
                    //     // // if (vis_spmv_val[0] == 1) // FP64
                    //     // // {
                    //     // //     d_vis[(index_dot / 16)] = 1;
                    //     // // }
                    //     // else if (vis_spmv_val[0] == 2) // fp8
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 2;
                    //     // }
                    //     // else if (vis_spmv_val[0] == 3) // fp16
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 3;
                    //     // }
                    //     // else if (vis_spmv_val[0] == 4) // fp32
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 4;
                    //     // }
                    // }
                    // if (lane_id == 16)
                    // {
                    //     // if (vis_spmv_val[1] == 0) // bypass
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 1;
                    //     // }
                    //     // // if (vis_spmv_val[1] == 1) // FP64
                    //     // // {
                    //     // //     d_vis[(index_dot / 16)] = 1;
                    //     // // }
                    //     // else if (vis_spmv_val[1] == 2) // fp8
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 2;
                    //     // }
                    //     // else if (vis_spmv_val[1] == 3) // fp16
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 3;
                    //     // }
                    //     // else if (vis_spmv_val[1] == 4) // fp32
                    //     // {
                    //     //     d_vis[(index_dot / 16)] = 4;
                    //     // }
                    // }
                    __syncthreads();
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
        }
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_shared_queue_mix_precision(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                                                                                int *signal_final1,
                                                                                                                                int *d_ori_block_signal,
                                                                                                                                double *k_alpha,
                                                                                                                                double *k_snew,
                                                                                                                                double *k_x,
                                                                                                                                double *k_r,
                                                                                                                                double *k_sold,
                                                                                                                                double *k_beta,
                                                                                                                                double *k_threshold,
                                                                                                                                int *d_balance_tile_ptr,
                                                                                                                                int *d_row_each_block,
                                                                                                                                int *d_index_each_block,
                                                                                                                                int balance_row,
                                                                                                                                int *d_non_each_block_offset,
                                                                                                                                int vector_each_warp,
                                                                                                                                int vector_total,
                                                                                                                                int *d_vis,
                                                                                                                                double *d_x_d_last,
                                                                                                                                float *d_Blockcsr_Val_f,
                                                                                                                                half *d_Blockcsr_Val_h,
                                                                                                                                int8_t *d_Blockcsr_Val_8,
                                                                                                                                float *d_x_f,
                                                                                                                                half *d_x_h,
                                                                                                                                int8_t *d_x_8,
                                                                                                                                float *d_y_f,
                                                                                                                                int *d_balance_tile_ptr_shared_end,
                                                                                                                                int shared_num)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    // __shared__ int row_begin[WARP_PER_BLOCK];
    // __shared__ int row_end[WARP_PER_BLOCK];
    // int *row_begin_val = &row_begin[local_warp_id];
    // int *row_end_val = &row_end[local_warp_id];
    __shared__ int vis_spmv[WARP_PER_BLOCK * 2];
    int *vis_spmv_val = &vis_spmv[local_warp_id * 2];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjshared_end = d_balance_tile_ptr_shared_end[blki_blc + 1];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        // row_begin_val[0] = d_balance_tile_ptr[blki_blc];
        // row_end_val[0] = d_balance_tile_ptr[blki_blc + 1];
        double sum_d = 0.0;
        float sum_f = 0.0;
        int8_t sum_8 = 0.0;
        half sum_h = __float2half(0.0f);
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
        int offset = blki_blc * vector_each_warp;
        int iter;
        int u;
        int row_end;

        const int nnz_per_warp = 256;
        __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
        double *s_data_val = &s_data[local_warp_id * nnz_per_warp];
        __shared__ float s_data_float[nnz_per_warp * WARP_PER_BLOCK];
        float *s_data_float_val = &s_data_float[local_warp_id * nnz_per_warp];
        __shared__ half s_data_half[nnz_per_warp * WARP_PER_BLOCK];
        half *s_data_half_val = &s_data_half[local_warp_id * nnz_per_warp];
        __shared__ int8_t s_data_int8[nnz_per_warp * WARP_PER_BLOCK];
        int8_t *s_data_int8_val = &s_data_int8[local_warp_id * nnz_per_warp];
        for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
        {
            blkj = d_index_each_block[blkj_blc];
            shared_offset = d_non_each_block_offset[blkj_blc];
            csroffset = d_ptroffset1[blkj];
            s1 = d_nonzero_row_new[blkj];
            s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    index_s = rj + shared_offset;
                    s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
                    // s_data_float_val[index_s] = (float)(d_Blockcsr_Val_d[csroffset + rj]);
                    // s_data_half_val[index_s] = (half)(d_Blockcsr_Val_d[csroffset + rj]);
                    // s_data_int8_val[index_s] = (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                    // s_data_float_val[index_s] = d_Blockcsr_Val_f[csroffset + rj];
                    // s_data_half_val[index_s] = d_Blockcsr_Val_h[csroffset + rj];
                    // s_data_int8_val[index_s] = d_Blockcsr_Val_8[csroffset + rj];
                    s_data_float_val[index_s] = (float)(s_data_val[index_s]);
                    s_data_half_val[index_s] = (half)(s_data_val[index_s]);
                    s_data_int8_val[index_s] = (int8_t)(s_data_val[index_s]);
                }
            }
            if (lane_id == 0)
            {
                atomicAdd(signal_final1, 1); // 这个同步保证shared memory的正确性 速度会变慢
            }
        }
        do
        {
            __threadfence();
        } while (signal_final1[0] != shared_num); // 这个同步保证shared memory的正确性 速度会变慢
        //__threadfence();
        //__syncthreads();

        // for(int iter=1;(iter<=100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (iter = 1; (iter <= 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            __syncthreads();
            // //__threadfence();
            if (lane_id < 2)
            {
                vis_spmv_val[lane_id] = 0;
            }
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0.0;
                // d_y_f[global_id] = 0.0;
            }
            __threadfence();

            if (global_id == 0)
            {
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
                // signal_final1[0] = 0;
            }
            __threadfence();
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                // if ((d_vis[colid] != 1))
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        shared_offset = d_non_each_block_offset[blkj_blc];
                        switch (d_vis[colid])
                        {
                        // if (d_vis[colid] == 0) // FP64
                        case 0:
                            //{
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * s_data_val[index_s]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        //}
                        // else if (d_vis[colid] == 4) // FP32
                        //{
                        case 4:
                            sum_f = 0.0;
                            // sum_d = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_f += d_x_f[x_offset + csrcol] * (float)s_data_val[index_s];
                                sum_f += d_x_f[x_offset + csrcol] * s_data_float_val[index_s];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                        //}
                        // else if (d_vis[colid] == 3) // FP16
                        //{
                        case 3:
                            sum_h = 0.0;
                            // sum_f = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      index_s = rj + shared_offset;
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],s_data_half_val[index_s],sum_h);
                            //      //sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                index_s = rj + shared_offset;
                                sum_h = __hfma(d_x_h[x_offset + csrcol], s_data_half_val[index_s], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)s_data_val[index_s], sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], s_data_half_val[index_s + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)s_data_val[index_s+1], sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // if (d_vis[colid] == 4) // FP8
                        // else
                        // {
                        case 2:
                            // sum_f = 0.0;
                            sum_8 = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      index_s = rj + shared_offset;
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      //sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      sum_8 += d_x_8[x_offset + csrcol]*s_data_int8_val[index_s];
                            //  }
                            // packed one thread 2 int_8 结果不对
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                index_s = rj + shared_offset;
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[csrcol + x_offset] * s_data_int8_val[index_s];
                                // sum_8 += d_x_8[csrcol+x_offset] * (int8_t)s_data_val[index_s];
                                if (rj + 1 < row_end)
                                {
                                    sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * s_data_int8_val[index_s + 1];
                                    // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1]+x_offset] * (int8_t)s_data_val[index_s+1];
                                }
                            }
                            // packed one thread 4 int_8 结果不对
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = (float)sum_8;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        default:
                            break;
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            // 如果能都放在shared memory中 则不需要下面的循环
            for (blkj_blc = rowblkjshared_end; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                colid = d_tile_columnidx[blkj];
                // if ((d_vis[colid] != 1))
                {
                    x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; // 列号
                    csroffset = d_ptroffset1[blkj];
                    s1 = d_nonzero_row_new[blkj];
                    s2 = d_nonzero_row_new[blkj + 1];
                    if (ri < s2 - s1)
                    {
                        ro = d_blockrowid_new[s1 + ri + 1];
                        switch (d_vis[colid])
                        {
                        // if (d_vis[colid] == 0) // FP64
                        //{
                        case 0:
                            sum_d = 0.0;
                            for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                            }
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            break;
                        // else if (d_vis[colid] == 4) // FP32
                        //{
                        case 4:
                            sum_f = 0.0;
                            // sum_d = 0.0;
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                            {
                                int csrcol = d_Tile_csr_Col[csroffset + rj];
                                // sum_f += d_x_f[x_offset + csrcol] * (float)(d_Blockcsr_Val_d[csroffset + rj]);
                                sum_f += d_x_f[x_offset + csrcol] * d_Blockcsr_Val_f[csroffset + rj];
                            }
                            sum_d = (double)(sum_f);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // else if (d_vis[colid] == 3) // FP16
                        //{
                        case 3:
                            sum_h = 0.0;
                            // sum_f = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_h = __hfma(d_x_h[x_offset + csrcol],d_Blockcsr_Val_h[csroffset + rj],sum_h);
                            //      //sum_h += d_x_h[x_offset + csrcol] * d_Blockcsr_Val_h[csroffset + rj];
                            //  }
                            // packed one thread 2 half
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_h = __hfma(d_x_h[x_offset + csrcol], d_Blockcsr_Val_h[csroffset + rj], sum_h);
                                // sum_h = __hfma(d_x_h[x_offset + csrcol], (half)(d_Blockcsr_Val_d[csroffset + rj]), sum_h);
                                if (rj + 1 < row_end)
                                {
                                    sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], d_Blockcsr_Val_h[csroffset + rj + 1], sum_h);
                                    // sum_h = __hfma(d_x_h[x_offset + d_Tile_csr_Col[csroffset + rj + 1]], (half)(d_Blockcsr_Val_d[csroffset + rj + 1]), sum_h);
                                }
                            }
                            sum_d = (double)(sum_h);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = __half2float(sum_h);
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        //}
                        // if (d_vis[colid] == 4) // FP8
                        // else
                        case 2:
                            //{
                            // sum_f = 0.0;
                            sum_8 = 0.0;
                            row_end = d_blockcsr_ptr_new[s1 + ri + 1];
                            // unpacked 结果正确
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < row_end; rj += 2)
                            //  {
                            //      csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += d_x_8[x_offset + csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //  }
                            // packed one thread 2 int_8 结果不对
                            for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id * 2; rj < row_end; rj += 4)
                            {
                                csrcol = d_Tile_csr_Col[csroffset + rj];
                                sum_8 += d_x_8[csrcol + x_offset] * d_Blockcsr_Val_8[csroffset + rj];
                                // sum_8 += d_x_8[csrcol + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                                if (rj + 1 < row_end)
                                {
                                    sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * d_Blockcsr_Val_8[csroffset + rj + 1];
                                    // sum_8 += d_x_8[d_Tile_csr_Col[csroffset + rj + 1] + x_offset] * (int8_t)(d_Blockcsr_Val_d[csroffset + rj + 1]);
                                }
                            }
                            // packed one thread 4 int_8 结果不对
                            //  for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id*3; rj < row_end; rj += 6)
                            //  {
                            //      int csrcol = d_Tile_csr_Col[csroffset + rj];
                            //      sum_8 += s_x_warp_8[csrcol]*d_Blockcsr_Val_8[csroffset + rj];
                            //      if(rj+1<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+1]]*d_Blockcsr_Val_8[csroffset + rj+1];
                            //      }
                            //      if(rj+2<row_end)
                            //      {
                            //          sum_8 += s_x_warp_8[d_Tile_csr_Col[csroffset + rj+2]]*d_Blockcsr_Val_8[csroffset + rj+2];
                            //      }
                            //  }
                            sum_d = (double)(sum_8);
                            atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                            // sum_f = (float)sum_8;
                            // atomicAdd(&d_y_f[blki * BLOCK_SIZE + ro], sum_f);
                            break;
                        default:
                            break;
                            //}
                        }
                    }
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < vector_total)
            {
                // 下面这个等待保证bypass的准确性
                for (u = 0; u < vector_each_warp * 2; u++)
                {
                    int off = blki_blc * vector_each_warp * 2;
                    index_dot = iter * d_ori_block_signal[(off + u)];
                    do
                    {
                        //__threadfence_system();
                        __threadfence();
                    } while (d_block_signal[(off + u)] != index_dot);
                }

                // 下面这个等待保证性能
                //  for(u = 0; u < vector_each_warp; u++)
                //  {
                //      index_dot=iter*d_ori_block_signal[(offset + u)];
                //      do
                //      {
                //          __threadfence_system();
                //          //__threadfence();
                //      }
                //      while (d_block_signal[(offset + u)] != index_dot);//同步写的有问题 不能保证bypass的准确性
                //  }

                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    s_dot1_val[lane_id] += (d_y_d[index_dot] * d_x_d[index_dot]);
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
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();

                if ((lane_id == 0))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        // printf("k_alpha=%e\n",k_alpha[0]);
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                        // 把剩下块的点积算了
                    }
                }

                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    // printf("k_alpha=%e\n",k_alpha[0]);//k_alpha的值和不是无同步的对不上
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];

                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id] += (k_r[index_dot] * k_r[index_dot]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot = (offset + u) * 32 + lane_id;
                    // 下面为bypass检索
                    double d_x_last = d_x_d[index_dot];
                    // d_x_d_last[index_dot]=d_x_d[index_dot];
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_y_d[index_dot] = 0.0;
                    d_x_f[index_dot] = (float)(d_x_d[index_dot]);
                    d_x_h[index_dot] = __double2half(d_x_d[index_dot]);
                    d_x_8[index_dot] = (int8_t)(d_x_8[index_dot]);
                    d_vis[(index_dot / 16)] = 0;
                    // if(fabs(d_x_last-d_x_d[index_dot])<=1e-8)
                    // {
                    //     d_vis[(index_dot/16)]=1;
                    // }
                    double val = fabs(d_x_d[index_dot] - d_x_last);
                    // if (val > 1e-8)
                    // {
                    //     vis_spmv_val[(lane_id / 16)] = 1;
                    // }
                    if (val <= 1e-7 && val > 1e-8) // fp8
                    {
                        vis_spmv_val[(lane_id / 16)] = 2;
                    }
                    else if (val <= 1e-6 && val > 1e-7) // fp16
                    {
                        vis_spmv_val[(lane_id / 16)] = 3;
                    }
                    else if (val <= 1e-5 && val > 1e-6) // fp32
                    {
                        vis_spmv_val[(lane_id / 16)] = 4;
                    }
                    else if (val > 1e-5)
                    {
                        vis_spmv_val[(lane_id / 16)] = 1; // fp64
                    }
                    __syncthreads();
                    // if(lane_id%16==0)
                    // {
                    //     switch (vis_spmv_val[lane_id/16])
                    //     {
                    //     case 0:
                    //         d_vis[(index_dot / 16)] = 1;
                    //         break;
                    //     case 2:
                    //         d_vis[(index_dot / 16)] = 2;
                    //         break;
                    //     case 3:
                    //          d_vis[(index_dot / 16)] = 3;
                    //          break;
                    //     case 4:
                    //          d_vis[(index_dot / 16)] = 4;
                    //          break;
                    //     default:
                    //         break;
                    //     }
                    // }
                    if (lane_id == 0)
                    {
                        // d_vis[(index_dot/16)]=0;
                        if (vis_spmv_val[0] == 0) // bypass
                        {
                            d_vis[(index_dot / 16)] = 1;
                        }
                        if (vis_spmv_val[0] == 2) // fp8
                        {
                            d_vis[(index_dot / 16)] = 2;
                        }
                        if (vis_spmv_val[0] == 3) // fp16
                        {
                            d_vis[(index_dot / 16)] = 3;
                        }
                        if (vis_spmv_val[0] == 4) // fp32
                        {
                            d_vis[(index_dot / 16)] = 4;
                        }
                    }
                    if (lane_id == 16)
                    {
                        // d_vis[(index_dot/16)]=0;
                        if (vis_spmv_val[1] == 0) // bypass
                        {
                            d_vis[(index_dot / 16)] = 1;
                        }
                        if (vis_spmv_val[1] == 2) // fp8
                        {
                            d_vis[(index_dot / 16)] = 2;
                        }
                        if (vis_spmv_val[1] == 3) // fp16
                        {
                            d_vis[(index_dot / 16)] = 3;
                        }
                        if (vis_spmv_val[1] == 4) // fp32
                        {
                            d_vis[(index_dot / 16)] = 4;
                        }
                    }
                    __syncthreads();
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
        }
    }
}

__forceinline__ __global__ void load_to_shared_time(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                    int *signal_final1,
                                                    int *d_ori_block_signal,
                                                    double *k_alpha,
                                                    double *k_snew,
                                                    double *k_x,
                                                    double *k_r,
                                                    double *k_sold,
                                                    double *k_beta,
                                                    double *k_threshold,
                                                    int *d_balance_tile_ptr,
                                                    int *d_row_each_block,
                                                    int *d_index_each_block,
                                                    int balance_row,
                                                    int *d_non_each_block_offset,
                                                    int vector_each_warp,
                                                    int vector_total,
                                                    int *d_vis,
                                                    double *d_x_d_last,
                                                    float *d_Blockcsr_Val_f,
                                                    half *d_Blockcsr_Val_h,
                                                    int8_t *d_Blockcsr_Val_8,
                                                    float *d_x_f,
                                                    half *d_x_h,
                                                    int8_t *d_x_8,
                                                    float *d_y_f,
                                                    int *d_balance_tile_ptr_shared_end,
                                                    int shared_num)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjshared_end = d_balance_tile_ptr_shared_end[blki_blc + 1];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
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

        const int nnz_per_warp = 256;
        __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
        double *s_data_val = &s_data[local_warp_id * nnz_per_warp];
        __shared__ float s_data_float[nnz_per_warp * WARP_PER_BLOCK];
        float *s_data_float_val = &s_data_float[local_warp_id * nnz_per_warp];
        __shared__ half s_data_half[nnz_per_warp * WARP_PER_BLOCK];
        half *s_data_half_val = &s_data_half[local_warp_id * nnz_per_warp];
        __shared__ int8_t s_data_int8[nnz_per_warp * WARP_PER_BLOCK];
        int8_t *s_data_int8_val = &s_data_int8[local_warp_id * nnz_per_warp];
        for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
        {
            blkj = d_index_each_block[blkj_blc];
            shared_offset = d_non_each_block_offset[blkj_blc];
            csroffset = d_ptroffset1[blkj];
            s1 = d_nonzero_row_new[blkj];
            s2 = d_nonzero_row_new[blkj + 1];
            if (ri < s2 - s1)
            {
                for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                {
                    index_s = rj + shared_offset;
                    s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
                    // s_data_float_val[index_s] = (float)(d_Blockcsr_Val_d[csroffset + rj]);
                    // s_data_half_val[index_s] = (half)(d_Blockcsr_Val_d[csroffset + rj]);
                    // s_data_int8_val[index_s] = (int8_t)(d_Blockcsr_Val_d[csroffset + rj]);
                    // s_data_float_val[index_s] = d_Blockcsr_Val_f[csroffset + rj];
                    // s_data_half_val[index_s] = d_Blockcsr_Val_h[csroffset + rj];
                    // s_data_int8_val[index_s] = d_Blockcsr_Val_8[csroffset + rj];
                    s_data_float_val[index_s] = (float)(s_data_val[index_s]);
                    s_data_half_val[index_s] = (half)(s_data_val[index_s]);
                    s_data_int8_val[index_s] = (int8_t)(s_data_val[index_s]);
                }
            }
            if (lane_id == 0)
            {
                atomicAdd(signal_final1, 1); // 这个同步保证shared memory的正确性 速度会变慢
            }
        }
        do
        {
            __threadfence();
        } while (signal_final1[0] != shared_num); // 这个同步保证shared memory的正确性 速度会变慢
    }
}

__forceinline__ __global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_optimize(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                                                              int *signal_final1,
                                                                                                              int *d_ori_block_signal,
                                                                                                              double *k_alpha,
                                                                                                              double *k_snew,
                                                                                                              double *k_x,
                                                                                                              double *k_r,
                                                                                                              double *k_sold,
                                                                                                              double *k_beta,
                                                                                                              double *k_threshold,
                                                                                                              int *d_balance_tile_ptr,
                                                                                                              int *d_row_each_block,
                                                                                                              int *d_index_each_block,
                                                                                                              int balance_row,
                                                                                                              int *d_non_each_block_offset,
                                                                                                              int vector_each_warp,
                                                                                                              int vector_total)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    // __shared__ double s_x_d[WARP_PER_BLOCK * BLOCK_SIZE];
    // double *s_x_warp_d = &s_x_d[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    // __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];

    // 存矩阵的数值
    //  const int nnz_per_warp=32;//不同的大小能跑不同的矩阵
    //  __shared__ double s_data[nnz_per_warp*WARP_PER_BLOCK];
    //  double *s_data_val = &s_data[local_warp_id * nnz_per_warp];

    __shared__ double s_dot1[WARP_PER_BLOCK * 32];
    double *s_dot1_val = &s_dot1[local_warp_id * 32];
    __shared__ double s_dot2[WARP_PER_BLOCK * 32];
    double *s_dot2_val = &s_dot2[local_warp_id * 32];
    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

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
        int ri;
        int virtual_lane_id;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        //  for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
        //  {
        //      //int blkj=d_index_each_block[blkj_blc];
        //      blkj=d_index_each_block[blkj_blc];
        //      shared_offset=d_non_each_block_offset[blkj_blc];
        //      csroffset = d_ptroffset1[blkj];
        //      ri = lane_id >> 1;
        //      virtual_lane_id = lane_id & 0x1;
        //      s1 = d_nonzero_row_new[blkj];
        //      s2 = d_nonzero_row_new[blkj + 1];
        //      if (ri < s2 - s1)
        //      {
        //          for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
        //          {
        //              index_s=rj+shared_offset;
        //              if(index_s<nnz_per_warp)
        //              s_data_val[index_s]=d_Blockcsr_Val_d[csroffset + rj];
        //          }
        //      }
        //  }
        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        // for (int iter = 0; (iter < 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            if (lane_id < 32)
            {
                s_dot1_val[lane_id] = 0.0;
                s_dot2_val[lane_id] = 0.0;
            }
            __syncthreads();
            //__threadfence();
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            __threadfence();
            //__threadfence_system();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
            }
            __threadfence();
            //__threadfence_system();
            if (global_id == 0)
            // if(lane_id==0)
            {
                // signal_dot[0] = tilem; // 问题在signal_dot
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
                // signal_final1[0] = 0;
            }
            __threadfence();
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                ri = lane_id >> 1;
                virtual_lane_id = lane_id & 0x1;
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
                        // if(index_s<nnz_per_warp)
                        // {
                        // // // // //     //只有加上printf才能跑
                        // // // // //     //printf("%e %e\n",s_data_val[index_s],d_Blockcsr_Val_d[csroffset + rj]);
                        //     sum_d += d_x_d[x_offset + csrcol]*s_data_val[index_s];
                        // }
                        // else
                        // {
                        sum_d = sum_d + (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                        // }
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
                int offset = blki_blc * vector_each_warp;
                for (int u = 0; u < vector_each_warp; u++)
                {
                    // do
                    // {
                    //     __threadfence();
                    // } while (d_block_signal[(offset + u)] != 0);
                    s_dot1_val[lane_id] += (d_y_d[(offset + u) * 32 + lane_id] * d_x_d[(offset + u) * 32 + lane_id]);
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
                    atomicAdd(k_alpha, s_dot1[0]);
                }
                __threadfence();

                if ((lane_id == 0))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }

                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                for (int u = 0; u < vector_each_warp; u++)
                {

                    k_x[(offset + u) * 32 + lane_id] = k_x[(offset + u) * 32 + lane_id] + s_alpha[local_warp_id] * d_x_d[(offset + u) * 32 + lane_id];

                    k_r[(offset + u) * 32 + lane_id] = k_r[(offset + u) * 32 + lane_id] - s_alpha[local_warp_id] * d_y_d[(offset + u) * 32 + lane_id];
                    __threadfence();

                    s_dot2_val[lane_id] += (k_r[(offset + u) * 32 + lane_id] * k_r[(offset + u) * 32 + lane_id]);
                }
                __syncthreads();
                i = (32 * WARP_PER_BLOCK) / 2;
                while (i != 0)
                {
                    if (threadIdx.x < i)
                    {
                        s_dot2[threadIdx.x] += s_dot2[threadIdx.x + i];
                    }
                    __syncthreads();
                    i /= 2;
                }
                if (threadIdx.x == 0)
                {
                    atomicAdd(k_snew, s_dot2[0]);
                }
                __threadfence();
                if (lane_id == 0)
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != vector_total);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                for (int u = 0; u < vector_each_warp; u++)
                {
                    d_x_d[(offset + u) * 32 + lane_id] = k_r[(offset + u) * 32 + lane_id] + s_beta[local_warp_id] * d_x_d[(offset + u) * 32 + lane_id];
                }
                if (lane_id == 0)
                {
                    atomicAdd(signal_final, 1);
                }
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != vector_total);
        }
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_shared_below_tilem_16(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                               int *signal_final1,
                                                                               int *d_ori_block_signal,
                                                                               double *k_alpha,
                                                                               double *k_snew,
                                                                               double *k_x,
                                                                               double *k_r,
                                                                               double *k_sold,
                                                                               double *k_beta,
                                                                               double *k_threshold,
                                                                               int *d_balance_tile_ptr,
                                                                               int *d_row_each_block,
                                                                               int *d_index_each_block,
                                                                               int balance_row,
                                                                               int *d_non_each_block_offset,
                                                                               int vector_each_warp,
                                                                               int vector_total)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    // __shared__ double s_x_d[WARP_PER_BLOCK * BLOCK_SIZE];
    // double *s_x_warp_d = &s_x_d[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    // __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];

    // 存矩阵的数值
    // const int nnz_per_warp = 256; // 不同的大小能跑不同的矩阵
    // __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
    // double *s_data_val = &s_data[local_warp_id * nnz_per_warp];

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        // int rowblkjstart = d_balance_tile_ptr[blki_blc];
        // int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        // for (int blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
        // {
        //     int blkj = d_index_each_block[blkj_blc];
        //     int shared_offset = d_non_each_block_offset[blkj_blc];
        //     int csroffset = d_ptroffset1[blkj];
        //     int ri = lane_id >> 1;
        //     int virtual_lane_id = lane_id & 0x1;
        //     int s1 = d_nonzero_row_new[blkj];
        //     int s2 = d_nonzero_row_new[blkj + 1];
        //     if (ri < s2 - s1)
        //     {
        //         for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
        //         {
        //             int index_s = rj + shared_offset;
        //             if (index_s < nnz_per_warp)
        //                 s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
        //         }
        //     }
        // }
        //__threadfence();
        //__syncthreads();
        // if(lane_id==0)
        // {
        //     //printf("cnt=%d blki_blc=%d\n",cnt,blki_blc);
        //     printf("\n");
        // }
        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 0; (iter < 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                // sharedFlag[threadIdx.x] = 0;
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            //__syncthreads();
            //__threadfence();
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            //__threadfence();
            __threadfence_system();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
                // signal_final[0]=0;
                // signal_dot[0]=tilem;
            }
            //__threadfence();
            __threadfence_system();
            if (global_id == 0)
            // if(lane_id==0)
            {
                // signal_dot[0] = tilem; // 问题在signal_dot
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            //__threadfence();
            //__threadfence_system();
            // if(global_id==0)
            // {
            //     printf("%e\n",sqrt(k_snew[0]));
            // }
            //__threadfence_system();
            // if(lane_id==0)
            // {
            //     printf("");
            // }
            double sum_d = 0.0;
            int rowblkjstart = d_balance_tile_ptr[blki_blc];
            int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
            // if (lane_id < rowblkjstop - rowblkjstart)
            // {
            //     s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            //     s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
            // }
            for (int blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                int blkj = d_index_each_block[blkj_blc];
                int blki = d_row_each_block[blkj_blc];
                // int colid = s_columnid_local[blkj - rowblkjstart];
                int colid = d_tile_columnidx[blkj];
                int x_offset = colid * BLOCK_SIZE;
                // int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
                int csroffset = d_ptroffset1[blkj];
                int ri = lane_id >> 1;
                int virtual_lane_id = lane_id & 0x1;
                int s1 = d_nonzero_row_new[blkj];
                int s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                int shared_offset = d_non_each_block_offset[blkj_blc];
                // if (lane_id < BLOCK_SIZE)
                // {
                //     s_x_warp_d[lane_id] = d_x_d[x_offset + lane_id];
                // }
                if (ri < s2 - s1)
                {
                    int ro = d_blockrowid_new[s1 + ri + 1];
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        int index_s = rj + shared_offset;
                        // //if (index_s < nnz_per_warp)
                        // {
                        //     //     //只有加上printf才能跑
                        //     //     //printf("%e %e\n",s_data_val[index_s],d_Blockcsr_Val_d[csroffset + rj]);
                        //     sum_d += d_x_d[x_offset + csrcol] * s_data_val[index_s];
                        // }
                        // else
                        // {
                        sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                        // }
                        // printf("Val_d=%e s_data=%e rj=%d blki_blc=%d offset_start=%d offset=%d diff=%d\n",d_Blockcsr_Val_d[csroffset + rj],s_data_val[csroffset + rj-csroffset_start],rj,blki_blc,csroffset_start,csroffset + rj,csroffset + rj-csroffset_start);
                    }
                    // if(lane_id==0)
                    // printf("");
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                // if (lane_id == 0)
                // {
                //     atomicSub(&d_block_signal[blki], 1);
                // }
            }

            // if (blki_blc < tilem)
            if (blki_blc < vector_total)
            {
                for (int u = 0; u < vector_each_warp; u++)
                {
                    // do
                    // {
                    //     __threadfence_system();
                    // } while (d_block_signal[(blki_blc * vector_each_warp + u)] != 0);
                    if ((lane_id < BLOCK_SIZE))
                        atomicAdd(k_alpha, (d_y_d[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] * d_x_d[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id]));
                    __threadfence_system();
                }
                if ((lane_id == 0))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                // 检测k_alpha是否更新完毕

                // do
                // {
                //     //__threadfence();
                //     __threadfence_system();
                // } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                for (int u = 0; u < vector_each_warp; u++)
                {
                    if ((lane_id < BLOCK_SIZE))
                        k_x[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] = k_x[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] + s_alpha[local_warp_id] * d_x_d[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id];
                    __threadfence_system();
                    if ((lane_id < BLOCK_SIZE))
                        k_r[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] = k_r[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] - s_alpha[local_warp_id] * d_y_d[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id];
                    __threadfence_system();
                    if ((lane_id < BLOCK_SIZE))
                        atomicAdd(k_snew, (k_r[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] * k_r[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id]));
                    __threadfence_system();
                }

                // if (lane_id == 0)
                // {
                //     atomicAdd(signal_dot, 1);
                // }
                // do
                // {
                //     //__threadfence();
                //     __threadfence_system();
                // } while (signal_dot[0] != vector_total);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                for (int u = 0; u < vector_each_warp; u++)
                {
                    if ((lane_id < BLOCK_SIZE))
                        d_x_d[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] = k_r[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id] + s_beta[local_warp_id] * d_x_d[(blki_blc * vector_each_warp + u) * BLOCK_SIZE + lane_id];
                }

                // if ((lane_id < BLOCK_SIZE))
                //     d_x_d[blki_blc * BLOCK_SIZE + lane_id] = k_r[blki_blc * BLOCK_SIZE + lane_id] + s_beta[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id];
                // if (lane_id == 0)
                // {
                //     atomicAdd(signal_final, 1);
                // }
            }
            // 加个同步 表示所有warp都算完了
            //  if(lane_id==0)
            //  {
            //      atomicAdd(signal_final,1);
            //  }
            // do
            // {
            //     //__threadfence();
            //     __threadfence_system();
            // } while (signal_final[0] != vector_total);
            // while(signal_final[0]!=balance_row);
            // 验证是否同步完成
            //  if(lane_id==0)
            //  {
            //      printf("signal_final=%d %d\n",signal_final[0],rowblkblock);
            //  }
        }
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_shared(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                                int *signal_final1,
                                                                int *d_ori_block_signal,
                                                                double *k_alpha,
                                                                double *k_snew,
                                                                double *k_x,
                                                                double *k_r,
                                                                double *k_sold,
                                                                double *k_beta,
                                                                double *k_threshold,
                                                                int *d_balance_tile_ptr,
                                                                int *d_row_each_block,
                                                                int *d_index_each_block,
                                                                int balance_row,
                                                                int *d_non_each_block_offset)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int blki_blc = global_id >> 5;
    const int local_warp_id = threadIdx.x >> 5;

    // __shared__ double s_x_d[WARP_PER_BLOCK * BLOCK_SIZE];
    // double *s_x_warp_d = &s_x_d[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    // __shared__ int s_columnid[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_columnid_local = &s_columnid[local_warp_id * PREFETCH_SMEM_TH];
    // __shared__ int s_ptroffset1[WARP_PER_BLOCK * PREFETCH_SMEM_TH];
    // int *s_ptroffset1_local = &s_ptroffset1[local_warp_id * PREFETCH_SMEM_TH];

    // 存矩阵的数值
    // const int nnz_per_warp = 32; // 不同的大小能跑不同的矩阵
    // __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
    // double *s_data_val = &s_data[local_warp_id * nnz_per_warp];

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        // int rowblkjstart = d_balance_tile_ptr[blki_blc];
        // int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        // for (int blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
        // {
        //     int blkj = d_index_each_block[blkj_blc];
        //     int shared_offset = d_non_each_block_offset[blkj_blc];
        //     int csroffset = d_ptroffset1[blkj];
        //     int ri = lane_id >> 1;
        //     int virtual_lane_id = lane_id & 0x1;
        //     int s1 = d_nonzero_row_new[blkj];
        //     int s2 = d_nonzero_row_new[blkj + 1];
        //     if (ri < s2 - s1)
        //     {
        //         for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
        //         {
        //             int index_s = rj + shared_offset;
        //             if (index_s < nnz_per_warp)
        //                 s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
        //         }
        //     }
        // }
        //__threadfence();
        //__syncthreads();
        // if(lane_id==0)
        // {
        //     //printf("cnt=%d blki_blc=%d\n",cnt,blki_blc);
        //     printf("\n");
        // }
        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 0; (iter < 100); iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                // sharedFlag[threadIdx.x] = 0;
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            //__syncthreads();
            //__threadfence();
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            //__threadfence();
            __threadfence_system();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
                // signal_final[0]=0;
                // signal_dot[0]=tilem;
            }
            //__threadfence();
            __threadfence_system();
            if (global_id == 0)
            // if(lane_id==0)
            {
                signal_dot[0] = tilem; // 问题在signal_dot
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            //__threadfence();
            //__threadfence_system();
            // if(global_id==0)
            // {
            //     printf("%e\n",sqrt(k_snew[0]));
            // }
            //__threadfence_system();
            // if(lane_id==0)
            // {
            //     printf("");
            // }
            double sum_d = 0.0;
            int rowblkjstart = d_balance_tile_ptr[blki_blc];
            int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
            // if (lane_id < rowblkjstop - rowblkjstart)
            // {
            //     s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            //     s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
            // }
            for (int blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                int blkj = d_index_each_block[blkj_blc];
                int blki = d_row_each_block[blkj_blc];
                // int colid = s_columnid_local[blkj - rowblkjstart];
                int colid = d_tile_columnidx[blkj];
                int x_offset = colid * BLOCK_SIZE;
                // int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
                int csroffset = d_ptroffset1[blkj];
                int ri = lane_id >> 1;
                int virtual_lane_id = lane_id & 0x1;
                int s1 = d_nonzero_row_new[blkj];
                int s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                int shared_offset = d_non_each_block_offset[blkj_blc];
                // if (lane_id < BLOCK_SIZE)
                // {
                //     s_x_warp_d[lane_id] = d_x_d[x_offset + lane_id];
                // }
                if (ri < s2 - s1)
                {
                    int ro = d_blockrowid_new[s1 + ri + 1];
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        int index_s = rj + shared_offset;
                        // if (index_s < nnz_per_warp)
                        // {
                        //     //     //只有加上printf才能跑
                        //     //     //printf("%e %e\n",s_data_val[index_s],d_Blockcsr_Val_d[csroffset + rj]);
                        //     sum_d += d_x_d[x_offset + csrcol] * s_data_val[index_s];
                        // }
                        // else
                        {
                            sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                        }
                        // printf("Val_d=%e s_data=%e rj=%d blki_blc=%d offset_start=%d offset=%d diff=%d\n",d_Blockcsr_Val_d[csroffset + rj],s_data_val[csroffset + rj-csroffset_start],rj,blki_blc,csroffset_start,csroffset + rj,csroffset + rj-csroffset_start);
                    }
                    // if(lane_id==0)
                    // printf("");
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                // if (lane_id == 0)
                // {
                //     atomicSub(&d_block_signal[blki], 1);
                // }
            }
            if (blki_blc < tilem)
            {
                // do
                // {
                //     //__threadfence();
                //     __threadfence_system();
                // } while (d_block_signal[blki_blc] != 0);
                if ((lane_id < BLOCK_SIZE))
                    atomicAdd(k_alpha, (d_y_d[blki_blc * BLOCK_SIZE + lane_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]));
                //__threadfence();
                __threadfence_system();
                if ((lane_id == 0))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                // 检测k_alpha是否更新完毕

                // do
                // {
                //     //__threadfence();
                //     __threadfence_system();
                // } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                if ((lane_id < BLOCK_SIZE))
                    k_x[blki_blc * BLOCK_SIZE + lane_id] = k_x[blki_blc * BLOCK_SIZE + lane_id] + s_alpha[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id];
                //__threadfence();
                __threadfence_system();
                if ((lane_id < BLOCK_SIZE))
                    k_r[blki_blc * BLOCK_SIZE + lane_id] = k_r[blki_blc * BLOCK_SIZE + lane_id] - s_alpha[local_warp_id] * d_y_d[blki_blc * BLOCK_SIZE + lane_id];
                //__threadfence();
                __threadfence_system();
                if ((lane_id < BLOCK_SIZE))
                    atomicAdd(k_snew, (k_r[blki_blc * BLOCK_SIZE + lane_id] * k_r[blki_blc * BLOCK_SIZE + lane_id]));
                //__threadfence();
                __threadfence_system();
                // if (lane_id == 0)
                // {
                //     atomicAdd(signal_dot, 1);
                // }
                // do
                // {
                //     //__threadfence();
                //     __threadfence_system();
                // } while (signal_dot[0] != tilem);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }

                if ((lane_id < BLOCK_SIZE))
                    d_x_d[blki_blc * BLOCK_SIZE + lane_id] = k_r[blki_blc * BLOCK_SIZE + lane_id] + s_beta[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id];
                // if (lane_id == 0)
                // {
                //     atomicAdd((int *)signal_final, 1);
                // }
            }
            // 加个同步 表示所有warp都算完了
            //  if(lane_id==0)
            //  {
            //      atomicAdd(signal_final,1);
            //  }
            // do
            // {
            //     //__threadfence();
            //     __threadfence_system();
            // } while (signal_final[0] != tilem);
            // while(signal_final[0]!=balance_row);
            // 验证是否同步完成
            //  if(lane_id==0)
            //  {
            //      printf("signal_final=%d %d\n",signal_final[0],rowblkblock);
            //  }
        }
    }
}

__global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_v1(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                                            int *signal_final1,
                                                            int *d_ori_block_signal,
                                                            double *k_alpha,
                                                            double *k_snew,
                                                            double *k_x,
                                                            double *k_r,
                                                            double *k_sold,
                                                            double *k_beta,
                                                            double *k_threshold,
                                                            int *d_balance_tile_ptr,
                                                            int *d_row_each_block,
                                                            int *d_index_each_block,
                                                            int balance_row,
                                                            int *d_non_each_block_offset)
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

    // 存矩阵的数值
    // const int nnz_per_warp = 10; // 不同的大小能跑不同的矩阵
    // __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
    // double *s_data_val = &s_data[local_warp_id * nnz_per_warp];

    // 同步数组
    __shared__ int sharedFlag[WARP_PER_BLOCK];
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        // int rowblkjstart = d_balance_tile_ptr[blki_blc];
        // int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        // for (int blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
        // {
        //     int blkj = d_index_each_block[blkj_blc];
        //     int shared_offset = d_non_each_block_offset[blkj_blc];
        //     int csroffset = d_ptroffset1[blkj];
        //     int ri = lane_id >> 1;
        //     int virtual_lane_id = lane_id & 0x1;
        //     int s1 = d_nonzero_row_new[blkj];
        //     int s2 = d_nonzero_row_new[blkj + 1];
        //     if (ri < s2 - s1)
        //     {
        //         for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
        //         {
        //             int index_s = rj + shared_offset;
        //             if (index_s < nnz_per_warp)
        //                 s_data_val[index_s] = d_Blockcsr_Val_d[csroffset + rj];
        //             // printf("offset=%d blki_blc=%d blkj_blc=%d shared_offset=%d\n",rj+shared_offset,blki_blc,blkj_blc,shared_offset);
        //             // printf("offset=%d blki_blc=%d nnz_block=%d\n",rj+shared_offset,blki_blc,rowblkjstop-rowblkjstart);
        //             // printf("index_s=%d csroffset_start=%d csroffset + rj=%d blki_blc=%d\n",index_s,csroffset_start,csroffset + rj,blki_blc);
        //         }
        //     }
        // }
        // __threadfence();
        //__syncthreads();
        // if(lane_id==0)
        // {
        //     //printf("cnt=%d blki_blc=%d\n",cnt,blki_blc);
        //     printf("\n");
        // }
        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        // for(int iter=0;(iter<100);iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                sharedFlag[threadIdx.x] = 0;
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            //__syncthreads();
            __threadfence();
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
                // signal_final[0]=0;
                // signal_dot[0]=tilem;
            }
            __threadfence();
            if (global_id == 0)
            // if(lane_id==0)
            {
                signal_dot[0] = tilem; // 问题在signal_dot
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();
            // if(global_id==0)
            // {
            //     printf("%e\n",sqrt(k_snew[0]));
            // }
            // if(lane_id==0)
            // {
            //     printf("");
            // }
            double sum_d = 0.0;
            int rowblkjstart = d_balance_tile_ptr[blki_blc];
            int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
            // if (lane_id < rowblkjstop - rowblkjstart)
            // {
            //     s_columnid_local[lane_id] = d_tile_columnidx[rowblkjstart + lane_id];
            //     s_ptroffset1_local[lane_id] = d_ptroffset1[rowblkjstart + lane_id];
            // }
            for (int blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            {
                int blkj = d_index_each_block[blkj_blc];
                int blki = d_row_each_block[blkj_blc];
                // int colid = s_columnid_local[blkj - rowblkjstart];
                int colid = d_tile_columnidx[blkj];
                int x_offset = colid * BLOCK_SIZE;
                // int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
                int csroffset = d_ptroffset1[blkj];
                int ri = lane_id >> 1;
                int virtual_lane_id = lane_id & 0x1;
                int s1 = d_nonzero_row_new[blkj];
                int s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                int shared_offset = d_non_each_block_offset[blkj_blc];
                // if (lane_id < BLOCK_SIZE)
                // {
                //     s_x_warp_d[lane_id] = d_x_d[x_offset + lane_id];
                // }
                if (ri < s2 - s1)
                {
                    int ro = d_blockrowid_new[s1 + ri + 1];
                    for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        int csrcol = d_Tile_csr_Col[csroffset + rj];
                        int index_s = rj + shared_offset;
                        // if (index_s < nnz_per_warp)
                        // {
                        //     // 只有加上printf才能跑
                        //     // printf("%e %e\n",s_data_val[index_s],d_Blockcsr_Val_d[csroffset + rj]);
                        //     sum_d += d_x_d[x_offset + csrcol] * s_data_val[index_s];
                        // }
                        // else
                        {
                            sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                        }
                        // printf("Val_d=%e s_data=%e rj=%d blki_blc=%d offset_start=%d offset=%d diff=%d\n",d_Blockcsr_Val_d[csroffset + rj],s_data_val[csroffset + rj-csroffset_start],rj,blki_blc,csroffset_start,csroffset + rj,csroffset + rj-csroffset_start);
                    }
                    // if(lane_id==0)
                    // printf("");
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicSub(&d_block_signal[blki], 1);
                }
            }
            // do{
            //     __threadfence();
            // }while(d_block_signal[blki_blc]!=0);
            // if(blki_blc<tilem)
            {
                if (blki_blc < tilem)
                {
                    do
                    {
                        __threadfence();
                    } while (d_block_signal[blki_blc] != 0);
                    // if(lane_id==0)
                    // printf("%d %d\n",blki_blc,d_block_signal[blki_blc]);
                }

                if ((lane_id < BLOCK_SIZE) && (blki_blc < tilem))
                    atomicAdd(k_alpha, (d_y_d[blki_blc * BLOCK_SIZE + lane_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]));
                __threadfence();
                // if(lane_id==0)
                // printf("%d %d %d\n",blki_blc,signal_dot[0],tilem);

                if ((lane_id == 0) && (blki_blc < tilem))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                // 检测k_alpha是否更新完毕
                if (blki_blc < tilem)
                {
                    do
                    {
                        __threadfence();
                    } while (signal_dot[0] != 0);
                }
                // if(lane_id==0)
                // printf("%d %d %d\n",blki_blc,signal_dot[0],tilem);

                if ((lane_id == 0) && (blki_blc < tilem))
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }

                if ((lane_id < BLOCK_SIZE) && (blki_blc < tilem))
                    k_x[blki_blc * BLOCK_SIZE + lane_id] = k_x[blki_blc * BLOCK_SIZE + lane_id] + s_alpha[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();

                if ((lane_id < BLOCK_SIZE) && (blki_blc < tilem))
                    k_r[blki_blc * BLOCK_SIZE + lane_id] = k_r[blki_blc * BLOCK_SIZE + lane_id] - s_alpha[local_warp_id] * d_y_d[blki_blc * BLOCK_SIZE + lane_id];
                __threadfence();

                if ((lane_id < BLOCK_SIZE) && (blki_blc < tilem))
                    atomicAdd(k_snew, (k_r[blki_blc * BLOCK_SIZE + lane_id] * k_r[blki_blc * BLOCK_SIZE + lane_id]));
                __threadfence();

                if ((lane_id == 0) && (blki_blc < tilem))
                {
                    atomicAdd(signal_dot, 1);
                }
                if (blki_blc < tilem)
                {
                    do
                    {
                        __threadfence();
                    } while (signal_dot[0] != tilem);
                }
                // if(lane_id==0)
                // printf("%d %d %d\n",blki_blc,signal_dot[0],tilem);

                if ((lane_id == 0) && (blki_blc < tilem))
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }

                if ((lane_id < BLOCK_SIZE) && (blki_blc < tilem))
                    d_x_d[blki_blc * BLOCK_SIZE + lane_id] = k_r[blki_blc * BLOCK_SIZE + lane_id] + s_beta[local_warp_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id];
            }
            // 加个同步 表示所有warp都算完了
            if (lane_id == 0)
            {
                atomicAdd(signal_final, 1);
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != balance_row);
        }
    }
}

// 显示CUDA设备信息
void show_GPU_info(void)
{
    int deviceCount;
    // 获取CUDA设备总数
    cudaGetDeviceCount(&deviceCount);
    // 分别获取每个CUDA设备的信息
    for (int i = 0; i < deviceCount; i++)
    {
        // 定义存储信息的结构体
        cudaDeviceProp devProp;
        // 将第i个CUDA设备的信息写入结构体中
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "使用GPU device " << i << ": " << devProp.name << std::endl;
        std::cout << "设备全局内存总量：" << devProp.totalGlobalMem / 1024 / 1024 << "MB" << std::endl;
        std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "设备上一个线程块（Block）中可用的32位寄存器数量： " << devProp.regsPerBlock << std::endl;
        std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
        std::cout << "设备上多处理器的数量：" << devProp.multiProcessorCount << std::endl;
        std::cout << "======================================================" << std::endl;
    }
}

// 一个warp算PREFETCH_SMEM_TH个块 PREFETCH_SMEM_TH定义在common.h中 当设置为1的时候 每个warp算一个块
// PREFETCH_SMEM_TH=4的时候 可以正常运行
__global__ void stir_spmv_cuda_kernel_newcsr(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
                                             int *signal_final1,
                                             int *d_ori_block_signal,
                                             double *k_alpha,
                                             double *k_snew,
                                             double *k_x,
                                             double *k_r,
                                             double *k_sold,
                                             double *k_beta,
                                             double *k_threshold)
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

    // 存矩阵的数值
    //  const int nnz_per_warp=256;//不同的大小能跑不同的矩阵
    //  __shared__ double s_data[nnz_per_warp*WARP_PER_BLOCK];
    //  double *s_data_val = &s_data[local_warp_id * nnz_per_warp];

    // 同步数组
    __shared__ int sharedFlag[WARP_PER_BLOCK];
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];

    if (blki_blc < rowblkblock)
    {
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
        // int csroffset_start=s_ptroffset1_local[0];
        // for (int blkj = rowblkjstart; blkj < rowblkjstop; blkj++)
        // {
        //     int colid = s_columnid_local[blkj - rowblkjstart];
        //     int x_offset = colid * BLOCK_SIZE;
        //     int csroffset = s_ptroffset1_local[blkj - rowblkjstart];
        //     int ri = lane_id >> 1;
        //     int virtual_lane_id = lane_id & 0x1;
        //     int s1 = d_nonzero_row_new[blkj];
        //     int s2 = d_nonzero_row_new[blkj + 1];
        //     if (ri < s2 - s1)
        //     {
        //         int ro = d_blockrowid_new[s1 + ri + 1];
        //         for (int rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
        //         {
        //             int index_s=csroffset + rj-csroffset_start;
        //             if(index_s<nnz_per_warp)
        //             s_data_val[index_s]=d_Blockcsr_Val_d[csroffset + rj];
        //         }
        //     }
        // }
        //__syncthreads();
        for (int iter = 0; (iter < 100) && (k_snew[0] > k_threshold[0]); iter++)
        // for(int iter=0;(iter<100);iter++)
        {
            if (threadIdx.x < WARP_PER_BLOCK)
            {
                sharedFlag[threadIdx.x] = 0;
                s_snew[threadIdx.x] = k_snew[0];
                s_alpha[threadIdx.x] = 0;
                s_beta[threadIdx.x] = 0;
            }
            //__syncthreads();
            __threadfence();
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
            }
            __threadfence();
            if (global_id < tilem)
            {
                d_block_signal[global_id] = d_ori_block_signal[global_id]; ////问题在d_ori_block_signal
                // signal_final[0]=0;
                // signal_dot[0]=tilem;
            }
            __threadfence();
            if (global_id == 0)
            // if(lane_id==0)
            {
                signal_dot[0] = tilem; // 问题在signal_dot
                k_alpha[0] = 0;
                signal_final[0] = 0;
            }
            __threadfence();
            // if(global_id==0)
            // {
            //     printf("%d %e\n",iter,sqrt(k_snew[0]));
            // }
            // if(lane_id==0)
            // {
            //     printf("");
            // }
            double sum_d = 0.0;
            int csroffset_start = s_ptroffset1_local[0];
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
                        int index_s = csroffset + rj - csroffset_start;
                        // if(index_s<nnz_per_warp)
                        // {
                        // sum_d += s_x_warp_d[csrcol]*s_data_val[index_s];
                        //}
                        // else
                        // {
                        sum_d += s_x_warp_d[csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                        // }
                        // printf("Val_d=%e s_data=%e rj=%d blki_blc=%d offset_start=%d offset=%d diff=%d\n",d_Blockcsr_Val_d[csroffset + rj],s_data_val[csroffset + rj-csroffset_start],rj,blki_blc,csroffset_start,csroffset + rj,csroffset + rj-csroffset_start);
                    }
                    // if(lane_id==0)
                    // printf("");
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
            }
            // 这里标记最后一个块被算完 sharedFlag数组通知其他warp内的线程
            if (lane_id == 0)
            {
                // if((atomicAdd(&d_block_signal[blki],rowblkjstop - rowblkjstart))+(rowblkjstop - rowblkjstart)==d_ori_block_signal[blki])
                if ((atomicSub(&d_block_signal[blki], rowblkjstop - rowblkjstart)) - (rowblkjstop - rowblkjstart) == 0)
                {
                    sharedFlag[local_warp_id] = 1;
                }
            }
            // 同步
            do
            {
                __threadfence();
            } while (d_block_signal[blki] != 0);
            // while(d_block_signal[blki]!=d_ori_block_signal[blki]);
            // 下面执行点积
            {
                if ((lane_id < BLOCK_SIZE) && (sharedFlag[local_warp_id] == 1))
                    atomicAdd(k_alpha, (d_y_d[blki * BLOCK_SIZE + lane_id] * d_x_d[blki * BLOCK_SIZE + lane_id]));
                __threadfence();
                if ((lane_id == 0) && (sharedFlag[local_warp_id] == 1))
                {
                    // 检测最后一个块的点积是否算完了(算k_alpha)
                    if (atomicSub(signal_dot, 1) - 1 == 0)
                    {
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                    }
                }
                // 检测k_alpha是否更新完毕
                do
                {
                    __threadfence();
                } while (signal_dot[0] != 0);
                if (lane_id == 0)
                {
                    s_alpha[local_warp_id] = s_snew[local_warp_id] / k_alpha[0];
                }
                __syncthreads();

                // 更新x和r
                // x = x + alpha*d
                if ((lane_id < BLOCK_SIZE) && (sharedFlag[local_warp_id] == 1))
                    k_x[blki * BLOCK_SIZE + lane_id] = k_x[blki * BLOCK_SIZE + lane_id] + s_alpha[local_warp_id] * d_x_d[blki * BLOCK_SIZE + lane_id];
                __threadfence();
                // r = r - alpha*q
                if ((lane_id < BLOCK_SIZE) && (sharedFlag[local_warp_id] == 1))
                    k_r[blki * BLOCK_SIZE + lane_id] = k_r[blki * BLOCK_SIZE + lane_id] - s_alpha[local_warp_id] * d_y_d[blki * BLOCK_SIZE + lane_id];
                __threadfence();
                if ((lane_id < BLOCK_SIZE) && (sharedFlag[local_warp_id] == 1))
                    atomicAdd(k_snew, (k_r[blki * BLOCK_SIZE + lane_id] * k_r[blki * BLOCK_SIZE + lane_id]));
                __threadfence();

                // 检测最后一个块的点积是否算完了 (算k_snew)
                if (lane_id == 0 && (sharedFlag[local_warp_id] == 1))
                {
                    atomicAdd(signal_dot, 1);
                }
                do
                {
                    __threadfence();
                } while (signal_dot[0] != tilem);

                if (lane_id == 0)
                {
                    s_beta[local_warp_id] = k_snew[0] / k_sold[0];
                }
                __syncthreads();
                if ((lane_id < BLOCK_SIZE) && (sharedFlag[local_warp_id] == 1))
                    d_x_d[blki * BLOCK_SIZE + lane_id] = k_r[blki * BLOCK_SIZE + lane_id] + s_beta[local_warp_id] * d_x_d[blki * BLOCK_SIZE + lane_id];
            }
            // 加个同步 表示所有warp都算完了
            if (lane_id == 0)
            {
                atomicAdd(signal_final, 1);
            }
            do
            {
                __threadfence();
            } while (signal_final[0] != rowblkblock);
            // 验证是否同步完成
            //  if(lane_id==0)
            //  {
            //      printf("signal_final=%d %d\n",signal_final[0],rowblkblock);
            //  }
        }
    }
}

void quickSort(int arr[], int row[], int index[], int low, int high)
{
    if (low < high)
    {
        int pivot = arr[high];
        int i = low - 1;

        for (int j = low; j <= high - 1; j++)
        {
            if (arr[j] < pivot)
            {
                i++;

                // 交换 non_each_block 数组的值
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;

                // 交换 row_each_block 数组的值
                temp = row[i];
                row[i] = row[j];
                row[j] = temp;

                // 交换 index_each_block 数组的值
                temp = index[i];
                index[i] = index[j];
                index[j] = temp;
            }
        }

        // 将 pivot 放到正确的位置
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        temp = row[i + 1];
        row[i + 1] = row[high];
        row[high] = temp;

        temp = index[i + 1];
        index[i + 1] = index[high];
        index[high] = temp;

        int partitionIndex = i + 1;

        // 递归调用快速排序对左侧和右侧的子数组进行排序
        quickSort(arr, row, index, low, partitionIndex - 1);
        quickSort(arr, row, index, partitionIndex + 1, high);
    }
}

void quickSort_col_Dub(int arr[], int row[], int index[], int col[], int low, int high)
{
    if (low < high)
    {
        int pivot = col[high]; // 使用 col 数组的值作为 pivot
        int i = low - 1;

        for (int j = low; j <= high - 1; j++)
        {
            // 按照 col 数组的值进行比较
            if (col[j] < pivot)
            {
                i++;

                // 交换数组的值
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;

                temp = row[i];
                row[i] = row[j];
                row[j] = temp;

                temp = index[i];
                index[i] = index[j];
                index[j] = temp;

                temp = col[i];
                col[i] = col[j];
                col[j] = temp;
            }
        }

        // 将 pivot 放到正确的位置
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        temp = row[i + 1];
        row[i + 1] = row[high];
        row[high] = temp;

        temp = index[i + 1];
        index[i + 1] = index[high];
        index[high] = temp;

        temp = col[i + 1];
        col[i + 1] = col[high];
        col[high] = temp;

        int partitionIndex = i + 1;

        // 递归调用快速排序对左侧和右侧的子数组进行排序
        quickSort_col_Dub(arr, row, index, col, low, partitionIndex - 1);
        quickSort_col_Dub(arr, row, index, col, partitionIndex + 1, high);
    }
}

void quickSort_col(int arr[], int row[], int index[], int col[], int low, int high)
{
    if (low < high)
    {
        int pivot = col[high]; // 使用 col 数组的值作为 pivot
        int i = low - 1;

        for (int j = low; j <= high - 1; j++)
        {
            // 按照 col 数组的值进行比较
            if (col[j] < pivot)
            {
                i++;

                // 交换数组的值
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;

                temp = row[i];
                row[i] = row[j];
                row[j] = temp;

                temp = index[i];
                index[i] = index[j];
                index[j] = temp;

                temp = col[i];
                col[i] = col[j];
                col[j] = temp;
            }
        }

        // 将 pivot 放到正确的位置
        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        temp = row[i + 1];
        row[i + 1] = row[high];
        row[high] = temp;

        temp = index[i + 1];
        index[i + 1] = index[high];
        index[high] = temp;

        temp = col[i + 1];
        col[i + 1] = col[high];
        col[high] = temp;

        int partitionIndex = i + 1;

        // 递归调用快速排序对左侧和右侧的子数组进行排序
        quickSort_col(arr, row, index, col, low, partitionIndex - 1);
        quickSort_col(arr, row, index, col, partitionIndex + 1, high);
    }
}

__global__ void sdot2_2(double *a, double *b, double *c, int n)
{

    // Define variables.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    double temp;
    temp = 0;
    // Define shared memories.
    __shared__ double s_data[1024];
    unsigned int tid = threadIdx.x;
    // Multiplication of data in the index.
    for (int i = index; i < n; i += stride)
    {
        temp += (a[i] * b[i]);
    }
    // Assign value to shared memory.
    s_data[tid] = temp;
    __syncthreads();
    // Add up products.
    for (int s = blockDim.x / 4; s > 0; s >>= 2)
    {
        if ((tid < s))
        {
            temp = s_data[tid];
            temp += s_data[tid + s];
            temp += s_data[tid + (s << 1)];
            temp += s_data[tid + (3 * s)];
            s_data[tid] = temp;
        }
        __syncthreads();
    }
    s_data[0] += s_data[1];
    if (tid == 0)
    {
        atomicAdd(c, s_data[0]);
    }
}
extern "C" void cg_solve_inc(int *RowPtr, int *ColIdx, MAT_VAL_TYPE *Val, MAT_VAL_LOW_TYPE *Val_Low, double *x, double *b, int n, int *iter, int maxiter, double threshold, char *filename, int nnzR, int ori, int block_nnz)
{
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);
    struct timeval t1, t2, t3, t4, t5, t6, t7, t8;
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
    // printf("num_seg=%d tilem=%d\n",num_seg,tilem);
    //  num_seg += 1;
    //  printf("rowA=%d colA=%d\n", rowA, colA);
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
    int *tile_nnz = matrix->tile_nnz;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
    MAT_VAL_LOW_TYPE *Blockcsr_Val_Low = matrix->Blockcsr_Val_Low;
    unsigned char *Tile_csr_Col = matrix->Tile_csr_Col;
    unsigned char *csr_compressedIdx = matrix->csr_compressedIdx;
    unsigned char *Blockcsr_Ptr = matrix->Blockcsr_Ptr;
    int csrsize = matrix->csrsize;
    int csrptrlen = matrix->csrptrlen;
    printf("num_seg=%d tilem=%d\n", num_seg, tilem);
    int csr_csize = csrsize % 2 == 0 ? csrsize / 2 : csrsize / 2 + 1;

    MAT_PTR_TYPE *d_tile_ptr;
    int *d_tile_columnidx;
    int *tile_rowidx = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_rowidx, 0, sizeof(int) * tilenum);
    int *d_tile_rowidx;
    cudaMalloc((void **)&d_tile_rowidx, tilenum * sizeof(int));
    cudaMalloc((void **)&d_tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE));
    cudaMalloc((void **)&d_tile_columnidx, tilenum * sizeof(int));

    cudaMemcpy(d_tile_ptr, tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_columnidx, tile_columnidx, tilenum * sizeof(int), cudaMemcpyHostToDevice);
    int *tile_columnidx_new = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_columnidx_new, 0, sizeof(int) * tilenum);
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

    // 矩阵低精度初始化
    float *d_Blockcsr_Val_float;
    half *d_Blockcsr_Val_half;
    half *Blockcsr_Val_half = (half *)malloc(sizeof(half) * csrsize);
    int8_t *Blockcsr_Val_int8 = (int8_t *)malloc(sizeof(int8_t) * csrsize);
    int8_t *d_Blockcsr_Val_int8;
    cudaMalloc((void **)&d_Blockcsr_Val_float, (csrsize) * sizeof(float));
    cudaMalloc((void **)&d_Blockcsr_Val_half, (csrsize) * sizeof(half));
    cudaMalloc((void **)&d_Blockcsr_Val_int8, (csrsize) * sizeof(int8_t));
    cudaMemcpy(d_Blockcsr_Val_float, Blockcsr_Val_Low, (csrsize) * sizeof(float), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < csrsize; i++)
    {
        Blockcsr_Val_half[i] = (half)(Blockcsr_Val_Low[i]);
        Blockcsr_Val_int8[i] = (int8_t)Blockcsr_Val[i];
    }
    cudaMemcpy(d_Blockcsr_Val_half, Blockcsr_Val_half, (csrsize) * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Blockcsr_Val_int8, Blockcsr_Val_int8, (csrsize) * sizeof(int8_t), cudaMemcpyHostToDevice);
    free(Blockcsr_Val_half);
    free(Blockcsr_Val_int8);

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

    cudaMalloc((void **)&d_x, rowA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_y, rowA * sizeof(MAT_VAL_TYPE));
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
    int num_blocks_new = ceil((double)(tilem) / (double)(num_threads / WARP_SIZE));

    float *k_d_float, *k_q_float;
    half *k_d_half;
    int8_t *k_d_int8;
    cudaMalloc((void **)&k_d_float, sizeof(float) * (n));
    cudaMalloc((void **)&k_d_half, sizeof(half) * (n));
    cudaMalloc((void **)&k_d_int8, sizeof(int8_t) * (n));
    cudaMalloc((void **)&k_q_float, sizeof(float) * (n));
    cudaMemset(k_q_float, 0, n * sizeof(float));

    double *k_b, *k_x, *k_r, *k_d, *k_q, *k_s;
    double *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;
    double t, s0, snew;
    double alpha;
    double *k_val;
    int iterations = 0;

    cudaMalloc((void **)&k_b, sizeof(double) * (n));
    cudaMemcpy(k_b, b, sizeof(double) * (n), cudaMemcpyHostToDevice);
    // for(int i=0;i<n;i++)
    // {
    //     printf("b=%e\n",b[i]);
    // }
    cudaMalloc((void **)&k_val, sizeof(double) * (nnzR));
    cudaMemcpy(k_val, Val, sizeof(double) * (nnzR), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&k_x, sizeof(double) * (n));
    cudaMalloc((void **)&k_r, sizeof(double) * (n + 1));
    cudaMalloc((void **)&k_d, sizeof(double) * (n + 1));
    double *d_last = (double *)malloc(sizeof(double) * (n + 1));
    memset(d_last, 0, sizeof(double) * (n + 1));
    double *d = (double *)malloc(sizeof(double) * (n + 1));
    memset(d, 0, sizeof(double) * (n + 1));
    int *vis_new = (int *)malloc(sizeof(int) * num_seg);
    memset(vis_new, 0, sizeof(int) * num_seg);
    cudaMalloc((void **)&k_q, sizeof(double) * (n));
    cudaMalloc((void **)&k_s, sizeof(double) * (n));
    cudaMalloc((void **)&k_alpha, sizeof(double));
    cudaMalloc((void **)&k_snew, sizeof(double));
    cudaMalloc((void **)&k_sold, sizeof(double));
    cudaMalloc((void **)&k_beta, sizeof(double));
    cudaMalloc((void **)&k_s0, sizeof(double));
    double *r = (double *)malloc(sizeof(double) * (n + 1));
    memset(r, 0, sizeof(double) * (n + 1));
    // dim3 BlockDim(NUM_THREADS);
    // dim3 GridDim(NUM_BLOCKS);

    dim3 BlockDim(128);
    dim3 GridDim((n / 128 + 1));

    veczero<<<1, BlockDim>>>(n, k_x);
    // r=b-Ax (r=b since x=0), and d=M^(-1)r
    cudaMemcpy(k_r, k_b, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
    // cublasDdot(cublasHandle, n, k_r, 1, k_r, 1, k_s0);
    cudaMemset(k_s0, 0, sizeof(double));

    sdot2_2<<<GridDim, BlockDim>>>(k_r, k_r, k_s0, n);
    // cudaMemcpy(r, k_r, sizeof(double) * (n), cudaMemcpyDeviceToHost);
    // double sum_r=0.0;
    // for(int i=0;i<n;i++)
    // {
    //     sum_r+=b[i]*b[i];
    // }
    // printf("sum_r=%e\n",sum_r);
    // r[n] = 1.1;
    cudaMemcpy(k_d, k_r, sizeof(double) * (n + 1), cudaMemcpyDeviceToDevice);
    device_convert<<<num_seg, BLOCK_SIZE>>>(k_d, k_d_float, n);
    device_convert_half<<<num_seg, BLOCK_SIZE>>>(k_d, k_d_half, n);
    device_convert_int8<<<num_seg, BLOCK_SIZE>>>(k_d, k_d_int8, n);
    // r[n] = 1.2;
    //  snew = s0
    scalarassign(k_snew, k_s0);
    // Copy snew and s0 back to host so that host can evaluate stopping condition
    cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s0, k_s0, sizeof(double), cudaMemcpyDeviceToHost);
    // printf("begin GPU CG\n");
    //  cudaMemset(d_vis_new, 0, num_seg * sizeof(int));
    //  cudaMemset(d_vis_mix, 0, num_seg * sizeof(int));
    double time_spmv = 0;

    printf("num_thread=%d\n", omp_get_max_threads());
    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
    gettimeofday(&t5, NULL);
#pragma omp parallel for
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
    printf("tilem=%d rowblkblock=%d tilenum=%d n=%d\n", tilem, rowblkblock, tilenum, rowA);
    int cnt_non_new = nonzero_row_new[tilenum];
    unsigned char *blockrowid_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockrowid_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    unsigned char *blockcsr_ptr_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockcsr_ptr_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    int csrcount_new1 = 0;
    int *block_signal = (int *)malloc(sizeof(int) * (tilem + 4));
    memset(block_signal, 0, sizeof(int) * (tilem + 4)); // 记录块数
    // #pragma omp parallel for
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

    // 负载均衡部分 重新分配每个warp需要计算的非零元
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
            // printf("%d ",tile_columnidx[blkj]);
        }
        // printf("\n");
    }
    // 快排根据每个块非零元素的个数从小到大
    // quickSort(non_each_block, row_each_block, index_each_block, 0, tilenum - 1);
    // quickSort_col_Dub(non_each_block, row_each_block, index_each_block,tile_columnidx, 0, tilenum - 1);
    // for(int i=0;i<tilenum;i++)
    // {
    //     tile_columnidx_new[i]=tile_columnidx[i];
    // }
    // quickSort_col(non_each_block, row_each_block, index_each_block,tile_columnidx_new, 0, tilenum - 1);
    printf("sort end\n");
    // for (int i = 0; i < tilenum; i++) {
    //     printf("%d ",tile_columnidx[index_each_block[i]]);
    // }
    // printf("\n");
    int *row_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));   // 记录每个块的行号
    int *index_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1)); // 排序前每个块的索引
    int *non_each_block_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(row_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(index_each_block_new, 0, sizeof(int) * (tilenum + 1));
    memset(non_each_block_new, 0, sizeof(int) * (tilenum + 1));
    // 需要一个额外的数组记录每个warp实际算的块数
    // int each_block_nnz=ceil((double)nnz_total / (double)tilem);
    // int each_block_nnz=100;
    // int each_block_nnz = 16;
    // int each_block_nnz=32;
    // int each_block_nnz = 48;
    // int each_block_nnz = 64;
    // int each_block_nnz = 4;
    int each_block_nnz = block_nnz;
    // int each_block_nnz = 8;
    // int each_block_nnz=640;
    printf("nnz_total=%d each_block_nnz=%d\n", nnz_total, each_block_nnz);
    int cnt = 0;
    int balance_row = 0;
    int index = 1;
    // int block_per_warp=248;
    // int cnt_block1=0;
    // 朴素的划分方式
    //  for (int i = 0; i < tilenum; i++) {
    //      cnt+=non_each_block[i];
    //      cnt_block1+=1;
    //      //printf("nnz=%d ", non_each_block[i]);
    //      if(cnt>=each_block_nnz||cnt_block1>=block_per_warp)
    //      {
    //          //printf("cnt=%d\n",cnt);
    //          balance_row++;
    //          cnt=0;
    //          cnt_block1=0;
    //      }
    //     //  if(cnt<each_block_nnz&&i==tilenum-1&&cnt!=0)
    //     //  {
    //     //      balance_row++;
    //     //  }
    //     if(cnt<each_block_nnz&&i==tilenum-1&&cnt!=0&&cnt_block1<block_per_warp)
    //     {
    //         balance_row++;
    //     }

    //  }
    //  printf("balance_row=%d\n",balance_row);
    //  int *balance_tile_ptr=(int *)malloc(sizeof(int)*(balance_row+1));
    //  memset(balance_tile_ptr,0,sizeof(int)*(balance_row+1));
    //  cnt=0;
    //  cnt_block1=0;
    //  for (int i = 0; i < tilenum; i++) {
    //      //printf("%d\n",non_each_block[i]);
    //      cnt+=non_each_block[i];
    //      cnt_block1+=1;

    //      if(cnt>=each_block_nnz||cnt_block1>=block_per_warp)
    //      {
    //          balance_tile_ptr[index]=i+1;
    //          index++;
    //          cnt=0;
    //          cnt_block1=0;
    //      }
    //      if(cnt<each_block_nnz&&i==tilenum-1&&cnt!=0&&cnt_block1<block_per_warp)
    //      {
    //          balance_tile_ptr[index]=i+1;
    //      }
    //  }
    //  for (int i = 0; i < index; i++)
    //  {
    //     cnt=0;
    //     for (int j = balance_tile_ptr[i]; j < balance_tile_ptr[i + 1]; j++)
    //     {
    //         cnt += non_each_block[j];
    //         //int blkj=index_each_block[j];
    //         //printf("%d ",tile_columnidx[blkj]);
    //     }
    //     printf("i=%d nnz_each_block=%d block_num=%d\n",i,cnt,balance_tile_ptr[i+1]-balance_tile_ptr[i]);
    //     //printf("\n");
    //  }
    //  printf("index=%d\n",index);
    // 使用双指针的方式 重新进行分配
    int i = 0;
    int j = tilenum - 1;
    cnt = 0;
    index = 1;
    int step = 0;
    // int block_per_warp=60;//cage12 Dubcova2 Dubcova3
    // int block_per_warp=70;//poisson3Da
    int block_per_warp = 180; // cage13 //appu 取得性能
    // int block_per_warp=240; //appu
    int cnt_block1 = 0;
    int nnz_list[12]={16,32,64,96,128,256,512,1024,2048,4096,nnzR/6912};
    while(1)
    {
    for(int k=0;k<12;k++)
    {
    each_block_nnz=nnz_list[k];
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
    if(index<6912)
    break;
    }
    printf("index=%d\n");
    if(index<6912)
    break;
    block_per_warp=block_per_warp*2;
    }
    // while (i < j)
    // {
    //     if (((non_each_block[i] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
    //     {
    //         cnt += non_each_block[i];
    //         i++;
    //         cnt_block1++;
    //     }
    //     else if (((non_each_block[i] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
    //     {
    //         i++;
    //         index++;
    //         cnt = 0;
    //         cnt_block1 = 0;
    //     }
    //     if (((non_each_block[j] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
    //     {
    //         cnt += non_each_block[j];
    //         j--;
    //         cnt_block1++;
    //     }
    //     else if (((non_each_block[j] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
    //     {
    //         j--;
    //         index++;
    //         cnt = 0;
    //         cnt_block1 = 0;
    //     }
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
        vector_each_warp_32 = vector_each_warp_16 * 2;
        vector_total_32 = tilem_32 / vector_each_warp_32;
        vector_total_32 = (vector_total_32 / WARP_PER_BLOCK + 1) * WARP_PER_BLOCK;
        printf("index=%d tilem=%d vector_each_warp_32=%d vector_total_32=%d\n", index, tilem, vector_each_warp_32, vector_total_32);
    }
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
    cnt_block1 = 0;
    while (i < j)
    {
        if (((non_each_block[i] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
        {
            cnt += non_each_block[i];
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[i] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
        {
            index_each_block_new[step] = index_each_block[i];
            row_each_block_new[step] = row_each_block[i];
            non_each_block_new[step] = non_each_block[i];
            i++;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1 = 0;
        }
        if (((non_each_block[j] + cnt) < each_block_nnz) && ((cnt_block1 + 1) < block_per_warp))
        {
            cnt += non_each_block[j];
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            cnt_block1++;
        }
        else if (((non_each_block[j] + cnt) >= each_block_nnz) || ((cnt_block1 + 1) >= block_per_warp))
        {
            index_each_block_new[step] = index_each_block[j];
            row_each_block_new[step] = row_each_block[j];
            non_each_block_new[step] = non_each_block[j];
            j--;
            step++;
            balance_tile_ptr_new[index] = step;
            index++;
            cnt = 0;
            cnt_block1 = 0;
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
    // 朴素的划分方式
    //  int *d_balance_tile_ptr;
    //  cudaMalloc((void **)&d_balance_tile_ptr, sizeof(int)*(balance_row+1));
    //  cudaMemcpy(d_balance_tile_ptr, balance_tile_ptr, sizeof(int) * (balance_row+1), cudaMemcpyHostToDevice);
    // 双指针划分方式
    int *d_balance_tile_ptr_new;
    cudaMalloc((void **)&d_balance_tile_ptr_new, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_new, balance_tile_ptr_new, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_row_each_block;
    int *d_index_each_block;
    cudaMalloc((void **)&d_row_each_block, sizeof(int) * (tilenum + 1));
    cudaMalloc((void **)&d_index_each_block, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_row_each_block, row_each_block_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_index_each_block, index_each_block_new, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);
    // 朴素的划分方式
    // cudaMemcpy(d_row_each_block, row_each_block, sizeof(int) * (tilenum+1), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_index_each_block, index_each_block, sizeof(int) * (tilenum+1), cudaMemcpyHostToDevice);
    // 朴素的划分方式
    //  for(int i=0;i<balance_row;i++)
    //  {
    //      for(int j=balance_tile_ptr[i];j<balance_tile_ptr[i+1];j++)
    //      {
    //          if(j==balance_tile_ptr[i])
    //          non_each_block_offset[j]=0;
    //          else
    //          {
    //              non_each_block_offset[j]=non_each_block[j-1];
    //              non_each_block_offset[j]+=non_each_block_offset[j-1];
    //          }
    //  }

    // 双指针的划分方式
    int cnt_block = 0;
    int cnt_nnz = 0;
    // for (int i = 0; i < index; i++)
    // {
    //     //printf("i=%d nnz_each_block=%d\n",i,non_each_block_new[i]);
    //     //int cnt_nnz=0;
    //     for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
    //     {
    //         //cnt_nnz+=non_each_block_new[j];
    //         cnt_block++;
    //         if (j == balance_tile_ptr_new[i])
    //             non_each_block_offset[j] = 0;
    //         else
    //         {
    //             non_each_block_offset[j] = non_each_block_new[j - 1];
    //             non_each_block_offset[j] += non_each_block_offset[j - 1];
    //         }
    //     }
    //     //printf("i=%d nnz_each_block=%d\n",i,cnt_nnz);
    // }
    for (int i = 0; i <= index; i++)
    {
        balance_tile_ptr_shared_end[i] = balance_tile_ptr_new[i];
    }
    int shared_nnz_each_block = 256;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            int blkj = index_each_block_new[j];
            // printf("%d ",tile_columnidx[blkj]);
            if (j == balance_tile_ptr_new[i])
                non_each_block_offset[j] = 0;
            cnt_nnz += non_each_block_new[j];
            cnt_block++;
            if (j != balance_tile_ptr_new[i] && cnt_nnz <= shared_nnz_each_block)
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
        // printf("\n");
        // printf("i=%d nnz_each_block=%d block_num=%d\n",i,cnt_nnz,balance_tile_ptr_new[i+1]-balance_tile_ptr_new[i]);
    }
    // 验证balance_tile_ptr_shared_end
    int cnt_nnz_shared = 0;
    int cnt_nnz_total = 0;
    int shared_num = 0;
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
            shared_num++;
        }
        for (int j = balance_tile_ptr_shared_end[i + 1]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            cnt_nnz += non_each_block_new[j];
        }
        // printf("cnt_nnz_shared=%d cnt_nnz=%d cnt_nnz_total=%d\n",cnt_nnz_shared,cnt_nnz,cnt_nnz_total);
    }

    printf("cnt_block=%d tilenum=%d\n", cnt_block, tilenum);
    int *d_non_each_block_offset;
    cudaMalloc((void **)&d_non_each_block_offset, sizeof(int) * (tilenum + 1));
    cudaMemcpy(d_non_each_block_offset, non_each_block_offset, sizeof(int) * (tilenum + 1), cudaMemcpyHostToDevice);

    int *d_balance_tile_ptr_shared_end;
    cudaMalloc((void **)&d_balance_tile_ptr_shared_end, sizeof(int) * (index + 1));
    cudaMemcpy(d_balance_tile_ptr_shared_end, balance_tile_ptr_shared_end, sizeof(int) * (index + 1), cudaMemcpyHostToDevice);
    int *d_block_signal;
    cudaMalloc((void **)&d_block_signal, sizeof(int) * (tilem + 4));
    int *signal_dot;
    cudaMalloc((void **)&signal_dot, sizeof(int));
    int *signal_final;
    cudaMalloc((void **)&signal_final, sizeof(int));
    int *signal_final1;
    cudaMalloc((void **)&signal_final1, sizeof(int));
    cudaMemset(signal_final1, 0, sizeof(int));
    double *k_threshold;
    cudaMalloc((void **)&k_threshold, sizeof(double));
    int *d_ori_block_signal;
    cudaMalloc((void **)&d_ori_block_signal, sizeof(int) * (tilem + 4));

    cudaMemcpy(d_block_signal, block_signal, sizeof(int) * (tilem + 4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ori_block_signal, block_signal, sizeof(int) * (tilem + 4), cudaMemcpyHostToDevice);
    gettimeofday(&t6, NULL);
    double time_format = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
    double pro_cnt = 0.0;
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
    threshold = epsilon * epsilon * s0;
    double *k_d_last;
    cudaMalloc((void **)&k_d_last, sizeof(double) * (n + 1));
    cudaMemset(k_d_last, 0, sizeof(double) * (n + 1));
    double time_load = 0;
    // printf("threshold=%e\n",threshold);
    double *k_x_new;
    cudaMemcpy(k_threshold, &threshold, sizeof(double), cudaMemcpyHostToDevice);
    gettimeofday(&t1, NULL);
    // while (iterations < 1 && snew > threshold)
    {
        if (index < tilem)
        {
            // index=balance_row;//朴素
            int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
            cudaMemset(d_block_signal, 0, sizeof(int) * (tilem + 1));
            if (vector_each_warp_32 * vector_total_32 * 32 > rowA)
            {
                printf("%d %d %d\n", rowA, (vector_each_warp_32 * vector_total_32 * 32), rowA / BLOCK_SIZE);
                rowA = vector_each_warp_32 * vector_total_32 * 32;
                printf("%d %d %d %d\n", rowA, (vector_each_warp_32 * vector_total_32 * 32), rowA / BLOCK_SIZE, tilem);
                printf("above\n");
            }
            int tilem_new = rowA / BLOCK_SIZE;
            int *d_block_signal_new;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            double *k_q_new;
            cudaMalloc((void **)&k_q_new, sizeof(double) * (rowA + 1));
            double *k_d_new;
            cudaMalloc((void **)&k_d_new, sizeof(double) * (rowA + 1));
            cudaMemset(k_d_new, 0, (rowA + 1) * sizeof(double));
            cudaMemcpy(k_d_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_r_new;
            cudaMalloc((void **)&k_r_new, sizeof(double) * (rowA + 1));
            cudaMemset(k_r_new, 0, (rowA + 1) * sizeof(double));
            cudaMemcpy(k_r_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            //double *k_x_new;
            cudaMalloc((void **)&k_x_new, sizeof(double) * (rowA + 1));
            cudaMemset(k_x_new, 0, (rowA + 1) * sizeof(double));
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            int *d_vis;
            cudaMalloc((void **)&d_vis, (rowA + 1) * sizeof(int));
            cudaMemset(d_vis, 0, (rowA + 1) * sizeof(int));
            float *k_d_float_new, *k_q_float_new;
            half *k_d_half_new;
            int8_t *k_d_int8_new;
            cudaMalloc((void **)&k_d_float_new, sizeof(float) * (rowA + 1));
            cudaMemcpy(k_d_float_new, k_d_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_half_new, sizeof(half) * (rowA + 1));
            cudaMemcpy(k_d_half_new, k_d_half, sizeof(half) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_int8_new, sizeof(int8_t) * (rowA + 1));
            cudaMemcpy(k_d_int8_new, &k_d_int8, sizeof(int8_t) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_q_float_new, sizeof(float) * (rowA + 1));
            cudaMemcpy(k_q_float_new, &k_q_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            // cudaDeviceSynchronize();
            // gettimeofday(&t7, NULL);
            // load_to_shared_time<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
            //                                                              d_tile_ptr, d_tile_columnidx,
            //                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                              d_ptroffset1, d_ptroffset2,
            //                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                              k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
            //                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
            //                                                              k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
            //                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,
            //                                                              vector_each_warp_32, vector_total_32, d_vis, k_d_last, d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
            //                                                              k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new, d_balance_tile_ptr_shared_end, shared_num);

            // cudaMemset(signal_final1, 0, sizeof(int));
            // cudaDeviceSynchronize();
            // gettimeofday(&t8, NULL);
            //time_load = (t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_mix_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                                                                        d_tile_ptr, d_tile_columnidx,
                                                                                                                                        d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                                                                        d_ptroffset1, d_ptroffset2,
                                                                                                                                        rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                                                                        k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                                                                        signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                                                                        k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
                                                                                                                                        d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,
                                                                                                                                        vector_each_warp_32, vector_total_32, d_vis, k_d_last, d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
                                                                                                                                        k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new);

            // stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce_shared_queue_mix_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
            //                                                                                                                                          d_tile_ptr, d_tile_columnidx,
            //                                                                                                                                          d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                                                                          d_ptroffset1, d_ptroffset2,
            //                                                                                                                                          rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                                                                          k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
            //                                                                                                                                          signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
            //                                                                                                                                          k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
            //                                                                                                                                          d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,
            //                                                                                                                                          vector_each_warp_32, vector_total_32, d_vis, k_d_last, d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
            //                                                                                                                                          k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new, d_balance_tile_ptr_shared_end, shared_num);

            // cudaMemcpy(&alpha, k_alpha, sizeof(double), cudaMemcpyDeviceToHost);
            // printf("alpha=%e\n",alpha);
            cudaDeviceSynchronize();
            // cudaMemcpy(d_last, k_d_last, sizeof(double)*n, cudaMemcpyDeviceToHost);
            // cudaMemcpy(d, k_d, sizeof(double)*n, cudaMemcpyDeviceToHost);
            // cudaMemcpy(vis_new, d_vis, num_seg * sizeof(int), cudaMemcpyDeviceToHost);
            // for(int u=0;u<n;u++)
            // {
            //     printf("%d %e %d\n",u,fabs(d[u] - d_last[u]),vis_new[u/16]);
            // }
            // printf("n=%d\n",n);
            // cudaMemcpy(d, k_q_new, sizeof(double)*n, cudaMemcpyDeviceToHost);
            // for(int u=0;u<n;u++)
            // {
            //     printf("%e\n",d[u]);
            // }
        }
        else
        {
            if(index==tilem)
            index=tilem+1;
            // 经过双指针重新分配 每个warp计算固定的非零元数目 待修改寄存器的占用率为62.5%
            cudaMemset(d_block_signal, 0, sizeof(int) * (tilem + 1));
            int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
            // stir_spmv_cuda_kernel_newcsr_nnz_balance<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
            //                                                                                   d_tile_ptr, d_tile_columnidx,
            //                                                                                   d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                   d_ptroffset1, d_ptroffset2,
            //                                                                                   rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                   k_d, k_q, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal,
            //                                                                                   signal_dot, signal_final, signal_final1, d_ori_block_signal,
            //                                                                                   k_alpha, k_snew, k_x, k_r, k_sold, k_beta, k_threshold,
            //                                                                                   d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset);
            // tilem = (tilem / WARP_PER_BLOCK) * WARP_PER_BLOCK;
            // cudaMemset(d_block_signal, 0, sizeof(int) * (tilem + 1));
            // int *d_vis;
            // cudaMalloc((void **)&d_vis, tilem * sizeof(int));
            // cudaMemset(d_vis, 0, tilem * sizeof(int));

            // 扩大容量
            int tilem_new = (tilem / WARP_PER_BLOCK + 2) * WARP_PER_BLOCK;
            int re_size = (tilem_new)*BLOCK_SIZE;
            printf("tilem=%d tilem_new=%d tilem/WARP_PER_BLOCK=%d tilem_new/WARP_PER_BLOCK=%d\n", tilem, tilem_new, tilem / WARP_PER_BLOCK, tilem_new / WARP_PER_BLOCK);
            int *d_block_signal_new;
            cudaMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            cudaMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            cudaMemset(d_ori_block_signal_new, 0, sizeof(int) * (tilem_new + 1));
            cudaMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), cudaMemcpyHostToDevice);
            double *k_q_new;
            cudaMalloc((void **)&k_q_new, sizeof(double) * re_size);
            double *k_d_new;
            cudaMalloc((void **)&k_d_new, sizeof(double) * re_size);
            cudaMemset(k_d_new, 0, re_size * sizeof(double));
            cudaMemcpy(k_d_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            double *k_r_new;
            cudaMalloc((void **)&k_r_new, sizeof(double) * re_size);
            cudaMemset(k_r_new, 0, re_size * sizeof(double));
            cudaMemcpy(k_r_new, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            //double *k_x_new;
            cudaMalloc((void **)&k_x_new, sizeof(double) * re_size);
            cudaMemset(k_x_new, 0, re_size * sizeof(double));
            cudaMemcpy(k_x_new, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
            int *d_vis;
            cudaMalloc((void **)&d_vis, tilem_new * sizeof(int));
            cudaMemset(d_vis, 0, tilem_new * sizeof(int));
            float *k_d_float_new, *k_q_float_new;
            half *k_d_half_new;
            int8_t *k_d_int8_new;
            cudaMalloc((void **)&k_d_float_new, sizeof(float) * (re_size));
            cudaMemcpy(k_d_float_new, k_d_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_half_new, sizeof(half) * (re_size));
            cudaMemcpy(k_d_half_new, k_d_half, sizeof(half) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_d_int8_new, sizeof(int8_t) * (re_size));
            cudaMemcpy(k_d_int8_new, &k_d_int8, sizeof(int8_t) * (n), cudaMemcpyDeviceToDevice);
            cudaMalloc((void **)&k_q_float_new, sizeof(float) * (re_size));
            cudaMemcpy(k_q_float_new, &k_q_float, sizeof(float) * (n), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
            gettimeofday(&t3, NULL);
            // 都放在global memory上
            stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_mixed_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
                                                                                                                          d_tile_ptr, d_tile_columnidx,
                                                                                                                          d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                                                          d_ptroffset1, d_ptroffset2,
                                                                                                                          rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                                                          k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                                                          signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                                                          k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
                                                                                                                          d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset, d_vis,
                                                                                                                          d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
                                                                                                                          k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new);
            // 放在shared memory上
            //  stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_shared_queue_mixed_precision<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
            //                                                                                    d_tile_ptr, d_tile_columnidx,
            //                                                                                    d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                    d_ptroffset1, d_ptroffset2,
            //                                                                                    rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                    k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
            //                                                                                    signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
            //                                                                                    k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
            //                                                                                    d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,d_balance_tile_ptr_shared_end,d_vis,
            //                                                                                    d_Blockcsr_Val_float, d_Blockcsr_Val_half, d_Blockcsr_Val_int8,
            //                                                                                    k_d_float_new, k_d_half_new, k_d_int8_new, k_q_float_new,shared_num);
        }
        cudaDeviceSynchronize();
        gettimeofday(&t4, NULL);
        time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        // cudaMemcpy(&alpha, k_alpha, sizeof(double), cudaMemcpyDeviceToHost);
        // printf("alpha=%e\n",alpha);
        // cublasDdot(cublasHandle, n, k_d, 1, k_q, 1, k_alpha);
        // sdot2_2<<<GridDim, BlockDim>>>(k_d,k_q,k_alpha,n);
        // sdot2_2<<<(n/64+1), 64>>>(k_d,k_q,k_alpha,n);
        // scalardiv<<<1, 1>>>(k_snew, k_alpha, k_alpha);

        // x = x + alpha*d
        // axpy<<<GridDim, BlockDim>>>(n, k_alpha, k_d, k_x, k_x);

        // r = r - alpha*q
        // ymax<<<GridDim, BlockDim>>>(n, k_alpha, k_q, k_r);
        // scalarassign(k_sold, k_snew);
        // cudaMemset(k_snew, 0, sizeof(double));
        // sdot2_2<<<GridDim, BlockDim>>>(k_r,k_r,k_snew,n);
        // beta = snew/sold
        // scalardiv<<<1, 1>>>(k_snew, k_sold, k_beta);
        // d = r + beta*d
        // axpy<<<GridDim, BlockDim>>>(n, k_beta, k_d, k_r, k_d);
        // record the last val
        //  Copy back snew so the host can evaluate the stopping condition
        cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
        printf("final residual=%e\n", sqrt(snew));
        // iterations++;
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    cudaMemcpy(x, k_x_new, sizeof(double) * (n), cudaMemcpyDeviceToHost);
    double time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("iter=%d,time_cg=%lf ms,time_spmv=%lf ms,time_format=%lf ms time_load=%lf ms\n", iterations, time_spmv, time_spmv, time_format, time_load);
    double *b_new = (double *)malloc(sizeof(double) * n);
    memset(b_new, 0, sizeof(double) * n);
    // printf("debug\n");
    for (int blki = 0; blki < tilem; blki++)
    {
        for (int ri = 0; ri < BLOCK_SIZE; ri++)
        {
            b_new[blki * BLOCK_SIZE + ri] = 0;
        }
        for (int blkj = matrix->tile_ptr[blki]; blkj < matrix->tile_ptr[blki + 1]; blkj++)
        {
            int csrcolidx = tile_columnidx[blkj];
            int x_offset = csrcolidx * BLOCK_SIZE;
            csroffset = matrix->csr_offset[blkj];
            // printf("csroffset=%d\n",csroffset);
            int cnt = 0;
            for (int ri = nonzero_row_new[blkj]; ri < nonzero_row_new[blkj + 1]; ri++)
            {
                double sum_new = 0;
                int ro = blockrowid_new[ri + 1];
                if (blockcsr_ptr_new[ri + 1] > blockcsr_ptr_new[ri])
                {
                    cnt = cnt + (blockcsr_ptr_new[ri + 1] - blockcsr_ptr_new[ri]);
                }
                for (int rj = blockcsr_ptr_new[ri]; rj < blockcsr_ptr_new[ri + 1]; rj++)
                {
                    // printf("rj=%d\n",rj);
                    // cnt++;
                    int csrcol = Tile_csr_Col[csroffset + rj];
                    sum_new += x[x_offset + csrcol] * matrix->Blockcsr_Val[csroffset + rj];
                }
                b_new[blki * BLOCK_SIZE + ro] += sum_new;
            }
            // printf("cnt=%d\n",cnt);
        }
    }
    double sum = 0;
    for (int i = 0; i < n; i++)
    {
        double r = b_new[i] - b[i];
        sum = sum + (r * r);
    }
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (b[i] * b[i]);
    }
    double l2_norm = sqrt(sum) / sqrt(sum_ori);
    // double Gflops_spmv= (2 * nnzR) / ((time_spmv/ iterations)*pow(10, 6));
    // printf("Gflops_Initial=%lf\n",Gflops_spmv);
    // printf("L2 Norm is %lf\n", l2_norm);
    char *s = (char *)malloc(sizeof(char) * 200);
    sprintf(s, "iter=%d,time_spmv=%.3f,time_cg=%.3f,nnzR=%d,l2_norm=%e,time_format=%lf,index=%d,each_nnz=%d,norm=%e\n", iterations, time_spmv, time_cg, nnzR, l2_norm, time_format, index, each_block_nnz,sqrt(snew));
    FILE *file1 = fopen("cg_syncfree_mixd_precision_test.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
    cudaFree(k_val);
    cudaFree(k_b);
    cudaFree(k_x);
    cudaFree(k_r);
    cudaFree(k_d);
    cudaFree(k_q);
    cudaFree(k_alpha);
    cudaFree(k_snew);
    cudaFree(k_sold);
    cudaFree(k_beta);
    cudaFree(k_s0);
    cudaFree(d_tile_ptr);
    cudaFree(d_tile_columnidx);
    cudaFree(d_csr_compressedIdx);
    cudaFree(d_Blockcsr_Val);
    cudaFree(d_Blockcsr_Ptr);
    cudaFree(d_blkcoostylerowidx);
    cudaFree(d_blkcoostylerowidx_colstart);
    cudaFree(d_blkcoostylerowidx_colstop);
    cudaFree(d_ptroffset1);
    cudaFree(d_ptroffset2);
    cudaFree(d_x);
    cudaFree(d_y);
    free(matrix);
    free(ptroffset1);
    free(ptroffset2);
    free(y_golden);
    free(y);
    free(blkcoostylerowidx);
    free(blkcoostylerowidx_colstart);
    free(blkcoostylerowidx_colstop);
    free(tile_ptr);
    free(tile_columnidx);
    free(tile_nnz);
    free(csr_offset);
    free(csrptr_offset);
    free(Blockcsr_Val);
    free(Blockcsr_Val_Low);
    free(csr_compressedIdx);
    free(Blockcsr_Ptr);
}
int main(int argc, char **argv)
{
    char *filename = argv[1];
    int block_nnz = atoi(argv[2]);
    // char *file_rhs = argv[2];
    int m, n, nnzR, isSymmetric;
    int *RowPtr;
    int *ColIdx;
    MAT_VAL_TYPE *Val;
    read_Dmatrix_32(&m, &n, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
    if(m!=n)
    return 0;
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
    int device;
    cudaGetDevice(&device);
    // show_GPU_info();
    int maxWarps;
    cudaDeviceGetAttribute(&maxWarps, cudaDevAttrMaxThreadsPerMultiProcessor, device);

    printf("Max warps per multiprocessor: %d\n", maxWarps);

    int maxSMs;
    cudaDeviceGetAttribute(&maxSMs, cudaDevAttrMultiProcessorCount, device);

    printf("Max SM count: %d\n", maxSMs);

    // int tt;
    // fscanf(file_rhs,"%d",&tt);
    // FILE *fp1 = fopen(file_rhs, "a+");
    // int tt;
    // fscanf(fp1, "%d", &tt);
    for (int i = 0; i < n; i++)
    {
        X[i] = 1;
        // fscanf(fp1, "%lf", &X[i]);
    }
    int iter = 0;
    for (int i = 0; i < n; i++)
        for (int j = RowPtr[i]; j < RowPtr[i + 1]; j++)
        {
            Y_golden[i] += Val[j] * X[ColIdx[j]];
            // printf("val=%e\n",Val[j]);
        }
    // for(int i=0;i<n;i++)
    // {
    //     printf("Y_golden=%e\n",Y_golden[i]);
    // }
    // for (int i = 0; i < n; i++)
    // {
    //   fscanf(fp1, "%lf", &X[i]);
    //
    // }

    cg_solve_inc(RowPtr, ColIdx, Val, Val_Low, X, Y_golden, n, &iter, 10, 1e-5, filename, nnzR, ori, block_nnz);
    // cg_cusparse(filename,RowPtr,ColIdx,Val,Y_golden,n,X,nnzR);
}
