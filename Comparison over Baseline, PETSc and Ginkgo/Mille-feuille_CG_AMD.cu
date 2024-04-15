#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hip/hip_fp16.h>
#include <sys/time.h>
#include "csr2block.h"
#include "blockspmv_cpu.h"
#include "utils.h"
//#include <hipblas.h>
#include <hip/hip_runtime.h>
//#include <hipsparse.h>
#include <hip/hip_runtime_api.h>
// #include <helper_functions.h>
// #include <helper_cuda.h>
//#include "./biio2.0/src/biio.h"
#include "mmio_highlevel.h"
#include "common.h"
#include <iostream>
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

__global__ void veczero(int n, double *vec)
{
    for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
        vec[i] = 0;
}


// Sets dest=src for scalars on the GPU.
void scalarassign(double *dest, double *src)
{
    hipMemcpy(dest, src, sizeof(double), hipMemcpyDeviceToDevice);
}

// Sets dest=src for n-length vectors on the GPU.
void vecassign(double *dest, double *src, int n)
{
    hipMemcpy(dest, src, sizeof(double) * n, hipMemcpyDeviceToDevice);
}


__global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block(int tilem, int tilenum, int rowA, int colA, int nnzA,
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
    // const int blki_blc = global_id >> 6;
    // const int local_warp_id = threadIdx.x >> 6;

    __shared__ double s_dot1[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot1_val = &s_dot1[local_warp_id * BLOCK_SIZE];
    __shared__ double s_dot2[WARP_PER_BLOCK * BLOCK_SIZE];
    double *s_dot2_val = &s_dot2[local_warp_id * BLOCK_SIZE];
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    

    // 同步数组
    __shared__ double s_snew[WARP_PER_BLOCK];
    __shared__ double s_alpha[WARP_PER_BLOCK];
    __shared__ double s_beta[WARP_PER_BLOCK];
    // printf("(0)k_snew[0]=%e\n",k_snew[0]);//正确
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

        
        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
        for (int iter = 1; (iter <= 100); iter++)
        {
            // printf("(1)k_snew[0]=%e\n",k_snew[0]);
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
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
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
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    // if(lane_id==0) printf("ro=%d,blki_blc=%d\n",ro,blki_blc);///////
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }
            if (blki_blc < tilem)
            {
                index_dot=iter*d_ori_block_signal[blki_blc];
                do
                {
                    __threadfence_system();
                    //__threadfence();
                } while (d_block_signal[blki_blc] != index_dot);
                //while (d_block_signal[blki_blc] != 0);
                
                index_dot=offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id]+=(d_y_d[index_dot] * d_x_d[index_dot]);
                    //atomicAdd(k_alpha, (d_y_d[blki_blc * BLOCK_SIZE + lane_id] * d_x_d[blki_blc * BLOCK_SIZE + lane_id]));
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
                    s_dot2_val[lane_id]+=(k_r[index_dot] * k_r[index_dot]);
                    //atomicAdd(k_snew, (k_r[blki_blc * BLOCK_SIZE + lane_id] * k_r[blki_blc * BLOCK_SIZE + lane_id]));
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
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                if (lane_id == 0)
                {
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



__global__ void stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_shared_queue(int tilem, int tilenum, int rowA, int colA, int nnzA,
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

        const int nnz_per_warp = 312;
        __shared__ double s_data[nnz_per_warp * WARP_PER_BLOCK];
        double *s_data_val = &s_data[local_warp_id * nnz_per_warp];
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
                }
            }
            if(lane_id==0)
            {
                atomicAdd(signal_final1, 1);//这个同步保证shared memory的正确性 速度会变慢
            }
        }
        do
        {
            __threadfence();
        } while (signal_final1[0] != shared_num);//这个同步保证shared memory的正确性 速度会变慢
        
        // for(int iter=0;(iter<100)&&(k_snew[0]>k_threshold[0]);iter++)
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
            __threadfence();
            if (global_id < rowA)
            {
                d_y_d[global_id] = 0;
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

            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjshared_end; blkj_blc++)
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
                        sum_d += d_x_d[x_offset + csrcol] * s_data_val[index_s];
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }


            //如果能都放在shared memory中 则不需要下面的循环
            for (blkj_blc = rowblkjshared_end; blkj_blc < rowblkjstop; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc];
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE;
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_d += d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj];
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }


            if (blki_blc < tilem)
            {
                index_dot=iter*d_ori_block_signal[blki_blc];
                do
                {
                    //__threadfence_system();
                    __threadfence();
                } while (d_block_signal[blki_blc] != index_dot);
                
                index_dot=offset + lane_id;
                if ((lane_id < BLOCK_SIZE))
                {
                    s_dot1_val[lane_id]+=(d_y_d[index_dot] * d_x_d[index_dot]);
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
                //检测k_alpha是否更新完毕
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
                    __threadfence();
                    k_r[index_dot] = k_r[index_dot] - s_alpha[local_warp_id] * d_y_d[index_dot];
                    __threadfence();
                    s_dot2_val[lane_id]+=(k_r[index_dot] * k_r[index_dot]);
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
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                if (lane_id == 0)
                {
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

    // __shared__ int row_begin[WARP_PER_BLOCK];
    // __shared__ int row_end[WARP_PER_BLOCK];
    // int *row_begin_val = &row_begin[local_warp_id];
    // int *row_end_val = &row_end[local_warp_id];
    // __shared__ double sum_d[WARP_PER_BLOCK];
    // double *sum_d_val = &sum_d[local_warp_id];
    // __shared__ int index_dot[WARP_PER_BLOCK*5];//后面的系数根据不同的矩阵去调整
    // int *index_dot_val = &index_dot[local_warp_id*5];

    if (blki_blc < balance_row)
    {
        // 存到shared memory
        int rowblkjstart = d_balance_tile_ptr[blki_blc];
        int rowblkjstop = d_balance_tile_ptr[blki_blc + 1];
        //if(lane_id==0)
        //{
            //row_begin_val[0] = d_balance_tile_ptr[blki_blc];
            //row_end_val[0] = d_balance_tile_ptr[blki_blc + 1];
        //}
        double sum_d = 0.0;
        int blkj_blc;
        int blkj;
        int blki;
        //int shared_offset;
        int csroffset;
        int ri = lane_id >> 1;;
        int virtual_lane_id = lane_id & 0x1;;
        int s1;
        int s2;
        int colid;
        int x_offset;
        int ro;
        int rj;
        int index_s;
        int csrcol;
        int index_dot;
        int offset=blki_blc * vector_each_warp;
        int iter;
        int u;
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
            __threadfence();
            // //__threadfence();
            // if (global_id < rowA)
            // {
            //     d_y_d[global_id] = 0.0;
            // }
            //__threadfence_system();
            // if (global_id < tilem)
            // {
            //     d_block_signal[global_id] = d_ori_block_signal[global_id]; //放在这里 开maxrregcount=32不会卡死 但是寄存器占用率会变成50%
            // }
            // __threadfence();
            //__threadfence_system();
            if (global_id == 0)
            {
                signal_dot[0] = vector_total;
                k_alpha[0] = 0;
                signal_final[0] = 0;
                signal_final1[0] = 0;
            }
            __threadfence();
            for (blkj_blc = rowblkjstart; blkj_blc < rowblkjstop; blkj_blc++)
            //for (blkj_blc = row_begin_val[0]; blkj_blc < row_end_val[0]; blkj_blc++)
            {
                blkj = d_index_each_block[blkj_blc];
                blki = d_row_each_block[blkj_blc]; //行号
                x_offset = d_tile_columnidx[blkj] * BLOCK_SIZE; //列号
                //x_offset = d_tile_columnidx[blkj_blc] * BLOCK_SIZE; //列号
                csroffset = d_ptroffset1[blkj];
                s1 = d_nonzero_row_new[blkj];
                s2 = d_nonzero_row_new[blkj + 1];
                sum_d = 0.0;
                //sum_d_val[0]=0.0;
                //shared_offset = d_non_each_block_offset[blkj_blc];
                if (ri < s2 - s1)
                {
                    ro = d_blockrowid_new[s1 + ri + 1];
                    for (rj = d_blockcsr_ptr_new[s1 + ri] + virtual_lane_id; rj < d_blockcsr_ptr_new[s1 + ri + 1]; rj += 2)
                    {
                        csrcol = d_Tile_csr_Col[csroffset + rj];
                        sum_d += (d_x_d[x_offset + csrcol] * d_Blockcsr_Val_d[csroffset + rj]);
                    }
                    atomicAdd(&d_y_d[blki * BLOCK_SIZE + ro], sum_d);//这个原子加占了25%的寄存器 其他的原子加占了12.5%
                }
                if (lane_id == 0)
                {
                    atomicAdd(&d_block_signal[blki], 1);
                }
            }

            if (blki_blc < vector_total)
            {
                //下面这个等待保证准确性
                for(u = 0; u < vector_each_warp*2; u++)
                {
                    int off=blki_blc * vector_each_warp*2;
                    index_dot=iter*d_ori_block_signal[(off + u)];
                    do
                    {
                        __threadfence_system();
                        //__threadfence();
                    } 
                    while (d_block_signal[(off + u)] != index_dot);
                }

                //下面这个等待保证性能
                // for(u = 0; u < vector_each_warp; u++)
                // {
                //     int off=blki_blc * vector_each_warp*2;
                //     index_dot=iter*d_ori_block_signal[(offset + u)];
                //     do
                //     {
                //         __threadfence_system();
                //         //__threadfence();
                //     } 
                //     while (d_block_signal[(offset + u)] != index_dot);//同步写的有问题 不能保证bypass的准确性
                // }
                
                for (u = 0; u < vector_each_warp; u++)
                {
                    // do
                    // {
                    //     __threadfence();
                    // } while (d_block_signal[(offset + u)] != 0);
                    index_dot=(offset + u) * 32 + lane_id;
                    //index_dot=index_dot_val[u]+lane_id;
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
                        //printf("k_alpha=%e\n",k_alpha[0]);
                        k_sold[0] = k_snew[0];
                        k_snew[0] = 0;
                        //把剩下块的点积算了
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
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    //index_dot=index_dot_val[u]+lane_id;
                    k_x[index_dot] = k_x[index_dot] + s_alpha[local_warp_id] * d_x_d[index_dot];
                    __threadfence();
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
                __threadfence();
                __syncthreads();
                for (u = 0; u < vector_each_warp; u++)
                {
                    index_dot=(offset + u) * 32 + lane_id;
                    //index_dot=index_dot_val[u]+lane_id;
                    d_x_d[index_dot] = k_r[index_dot] + s_beta[local_warp_id] * d_x_d[index_dot];
                    d_y_d[index_dot] = 0.0;
                }
                __threadfence();
                __syncthreads();
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




// 显示CUDA设备信息
void show_GPU_info(void)
{
    int deviceCount;
    // 获取CUDA设备总数
    hipGetDeviceCount(&deviceCount);
    // 分别获取每个CUDA设备的信息
    for (int i = 0; i < deviceCount; i++)
    {
        // 定义存储信息的结构体
        hipDeviceProp_t devProp;
        // 将第i个CUDA设备的信息写入结构体中
        hipGetDeviceProperties(&devProp, i);
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
    // hipblasHandle_t cublasHandle = 0;
    // hipblasStatus_t hipblasStatus_t;
    // hipblasStatus_t = hipblasCreate(&cublasHandle);
    struct timeval t1, t2, t3, t4, t5, t6;
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
    // num_seg += 1;
    // printf("rowA=%d colA=%d\n", rowA, colA);
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
    hipSetDevice(device_id);
    hipDeviceProp_t deviceProp;
    hipGetDeviceProperties(&deviceProp, device_id);
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

    int csr_csize = csrsize % 2 == 0 ? csrsize / 2 : csrsize / 2 + 1;

    MAT_PTR_TYPE *d_tile_ptr;
    int *d_tile_columnidx;
    int *tile_rowidx = (int *)malloc(sizeof(int) * tilenum);
    memset(tile_rowidx, 0, sizeof(int) * tilenum);
    int *d_tile_rowidx;
    hipMalloc((void **)&d_tile_rowidx, tilenum * sizeof(int));
    hipMalloc((void **)&d_tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE));
    hipMalloc((void **)&d_tile_columnidx, tilenum * sizeof(int));

    hipMemcpy(d_tile_ptr, tile_ptr, (tilem + 1) * sizeof(MAT_PTR_TYPE), hipMemcpyHostToDevice);
    hipMemcpy(d_tile_columnidx, tile_columnidx, tilenum * sizeof(int), hipMemcpyHostToDevice);
    int *tile_columnidx_new=(int *)malloc(sizeof(int)*tilenum);
    memset(tile_columnidx_new,0,sizeof(int)*tilenum);
    // CSR
    unsigned char *d_csr_compressedIdx = (unsigned char *)malloc((csr_csize) * sizeof(unsigned char));
    MAT_VAL_TYPE *d_Blockcsr_Val;
    unsigned char *d_Blockcsr_Ptr;

    hipMalloc((void **)&d_csr_compressedIdx, (csr_csize) * sizeof(unsigned char));
    hipMalloc((void **)&d_Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE));
    hipMalloc((void **)&d_Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char));

    hipMemcpy(d_csr_compressedIdx, csr_compressedIdx, (csr_csize) * sizeof(unsigned char), hipMemcpyHostToDevice);
    hipMemcpy(d_Blockcsr_Val, Blockcsr_Val, (csrsize) * sizeof(MAT_VAL_TYPE), hipMemcpyHostToDevice);
    hipMemcpy(d_Blockcsr_Ptr, Blockcsr_Ptr, (csrptrlen) * sizeof(unsigned char), hipMemcpyHostToDevice);

    // MAT_VAL_TYPE *blockcsrval=(MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE)*(csrsize));
    // memset(blockcsrval,0,sizeof(MAT_VAL_TYPE)*(csrsize));//////////////////

    unsigned int *d_blkcoostylerowidx;
    int *d_blkcoostylerowidx_colstart;
    int *d_blkcoostylerowidx_colstop;

    hipMalloc((void **)&d_blkcoostylerowidx, rowblkblock * sizeof(unsigned int));
    hipMalloc((void **)&d_blkcoostylerowidx_colstart, rowblkblock * sizeof(int));
    hipMalloc((void **)&d_blkcoostylerowidx_colstop, rowblkblock * sizeof(int));

    hipMemcpy(d_blkcoostylerowidx, blkcoostylerowidx, rowblkblock * sizeof(unsigned int), hipMemcpyHostToDevice);
    hipMemcpy(d_blkcoostylerowidx_colstart, blkcoostylerowidx_colstart, rowblkblock * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_blkcoostylerowidx_colstop, blkcoostylerowidx_colstop, rowblkblock * sizeof(int), hipMemcpyHostToDevice);

    int *d_ptroffset1;
    int *d_ptroffset2;

    hipMalloc((void **)&d_ptroffset1, tilenum * sizeof(int));
    hipMalloc((void **)&d_ptroffset2, tilenum * sizeof(int));
    hipMemcpy(d_ptroffset1, ptroffset1, tilenum * sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(d_ptroffset2, ptroffset2, tilenum * sizeof(int), hipMemcpyHostToDevice);

    // x and y
    MAT_VAL_TYPE *d_x;
    MAT_VAL_TYPE *d_y;

    hipMalloc((void **)&d_x, rowA * sizeof(MAT_VAL_TYPE));
    hipMalloc((void **)&d_y, rowA * sizeof(MAT_VAL_TYPE));
    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = ceil((double)rowblkblock / (double)(num_threads / WARP_SIZE));
    int num_blocks_new = ceil((double)(tilem) / (double)(num_threads / WARP_SIZE));
    double *k_b, *k_x, *k_r, *k_d, *k_q, *k_s;
    double *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;
    double t, s0, snew;
    double alpha;
    double *k_val;
    int iterations = 0;

    hipMalloc((void **)&k_b, sizeof(double) * (n));
    hipMemcpy(k_b, b, sizeof(double) * (n), hipMemcpyHostToDevice);
    hipMalloc((void **)&k_val, sizeof(double) * (nnzR));
    hipMemcpy(k_val, Val, sizeof(double) * (nnzR), hipMemcpyHostToDevice);

    hipMalloc((void **)&k_x, sizeof(double) * (n));
    hipMalloc((void **)&k_r, sizeof(double) * (n + 1));
    hipMalloc((void **)&k_d, sizeof(double) * (n + 1));
    hipMalloc((void **)&k_q, sizeof(double) * (n));
    hipMalloc((void **)&k_s, sizeof(double) * (n));
    hipMalloc((void **)&k_alpha, sizeof(double));
    hipMalloc((void **)&k_snew, sizeof(double));
    hipMalloc((void **)&k_sold, sizeof(double));
    hipMalloc((void **)&k_beta, sizeof(double));
    hipMalloc((void **)&k_s0, sizeof(double));
    double *r = (double *)malloc(sizeof(double) * (n + 1));
    memset(r, 0, sizeof(double) * (n + 1));
    // dim3 BlockDim(NUM_THREADS);
    // dim3 GridDim(NUM_BLOCKS);

    dim3 BlockDim(128);
    dim3 GridDim((n / 128 + 1));

    veczero<<<1, BlockDim>>>(n, k_x);
    // r=b-Ax (r=b since x=0), and d=M^(-1)r
    hipMemcpy(k_r, k_b, sizeof(double) * (n), hipMemcpyDeviceToDevice);
    // hipblasDdot(cublasHandle, n, k_r, 1, k_r, 1, k_s0);
    hipMemset(k_s0, 0, sizeof(double));
    sdot2_2<<<GridDim, BlockDim>>>(k_r, k_r, k_s0, n);
    // r[n] = 1.1;
    hipMemcpy(k_d, k_r, sizeof(double) * (n + 1), hipMemcpyDeviceToDevice);
    // r[n] = 1.2;
    //  snew = s0
    scalarassign(k_snew, k_s0);
    // Copy snew and s0 back to host so that host can evaluate stopping condition
    hipMemcpy(&snew, k_snew, sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&s0, k_s0, sizeof(double), hipMemcpyDeviceToHost);
    // printf("begin GPU CG\n");
    //  hipMemset(d_vis_new, 0, num_seg * sizeof(int));
    //  hipMemset(d_vis_mix, 0, num_seg * sizeof(int));
    double time_spmv = 0;
    double time_spmv_10 =0;
    double *new_y=(double*)malloc(sizeof(double)*(n));
    memset(new_y,0,sizeof(double)*(n));

    printf("num_thread=%d\n", omp_get_max_threads());
    // tile_newcsr
    int csroffset = 0;
    int csrcount = 0;
    int *nonzero_row_new = (int *)malloc(sizeof(int) * (tilenum + 1));
    memset(nonzero_row_new, 0, sizeof(int) * (tilenum + 1));
    gettimeofday(&t5, NULL);
// #pragma omp parallel for
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
    int *block_signal = (int *)malloc(sizeof(int) * (tilem + 1));
    memset(block_signal, 0, sizeof(int) * (tilem + 1)); // 记录块数
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
            //printf("%d ",tile_columnidx[blkj]);
        }
        //printf("\n");
    }
    // 快排根据每个块非零元素的个数从小到大
    //quickSort(non_each_block, row_each_block, index_each_block, 0, tilenum - 1);//不排有性能 但是共享内存会卡 待解决
    //quickSort_col_Dub(non_each_block, row_each_block, index_each_block,tile_columnidx, 0, tilenum - 1);
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
    int i = 0;
    int j = tilenum - 1;
    cnt = 0;
    index = 1;
    int step = 0;
    //int block_per_warp=60;//cage12 Dubcova2 Dubcova3
    //int block_per_warp=70;//poisson3Da
    int block_per_warp=180;//cage13 //appu 取得性能
    //int block_per_warp=240; //appu
    int cnt_block1=0;
    int nnz_list[12]={16,32,64,96,128,256,512,1024,2048,4096,nnzR/SM_NUM};//2048根據不同的機器不同
    while(1)
    {
    for(int k=0;k<12;k++)
    {
    each_block_nnz=nnz_list[k+1];
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
    if(index<SM_NUM&&index!=tilem-1&&index!=tilem+1&&index!=tilem)
    break;
    }
    printf("index=%d\n",index);
    if(index<SM_NUM&&index!=tilem-1&&index!=tilem+1&&index!=tilem)
    break;
    block_per_warp=block_per_warp*2;
    }   
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
        vector_each_warp_32 = vector_each_warp_16*2;
        vector_total_32 = tilem_32 / vector_each_warp_32;
        vector_total_32 = (vector_total_32/WARP_PER_BLOCK+1)*WARP_PER_BLOCK;
        //vector_total_32 = (vector_total_32/WARP_PER_BLOCK)*WARP_PER_BLOCK;
        printf("index=%d tilem=%d vector_each_warp_32=%d vector_total_32=%d\n", index, tilem, vector_each_warp_32, vector_total_32);
    }
    // if(index<tilem||index>6912) return;
    if (index > SM_NUM)
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
    hipMalloc((void **)&d_balance_tile_ptr_new, sizeof(int) * (index + 1));
    hipMemcpy(d_balance_tile_ptr_new, balance_tile_ptr_new, sizeof(int) * (index + 1), hipMemcpyHostToDevice);
    int *d_row_each_block;
    int *d_index_each_block;
    hipMalloc((void **)&d_row_each_block, sizeof(int) * (tilenum + 1));
    hipMalloc((void **)&d_index_each_block, sizeof(int) * (tilenum + 1));
    hipMemcpy(d_row_each_block, row_each_block_new, sizeof(int) * (tilenum + 1), hipMemcpyHostToDevice);
    hipMemcpy(d_index_each_block, index_each_block_new, sizeof(int) * (tilenum + 1), hipMemcpyHostToDevice);

    // 双指针的划分方式
    int cnt_block = 0;
    int cnt_nnz = 0;
    for (int i = 0; i <= index; i++)
    {
        balance_tile_ptr_shared_end[i] = balance_tile_ptr_new[i];
    }
    int cnt_nnz_shared=0;
    int shared_nnz_each_block=256;
    for (int i = 0; i < index; i++)
    {
        cnt_nnz = 0;
        cnt_nnz_shared=0;
        for (int j = balance_tile_ptr_new[i]; j < balance_tile_ptr_new[i + 1]; j++)
        {
            int blkj=index_each_block_new[j];
            //printf("%d ",tile_columnidx[blkj]);
            if (j == balance_tile_ptr_new[i])
                non_each_block_offset[j] = 0;
            cnt_nnz += non_each_block_new[j];
            cnt_block++;
            if (j != balance_tile_ptr_new[i] && cnt_nnz <=shared_nnz_each_block)
            {
                cnt_nnz_shared+=non_each_block_new[j - 1];
                non_each_block_offset[j] = non_each_block_new[j - 1];
                non_each_block_offset[j] += non_each_block_offset[j - 1];
            }
            if (cnt_nnz > shared_nnz_each_block)
            {
                balance_tile_ptr_shared_end[i + 1] = j;
                break;
            }
        }

    }
    // 验证balance_tile_ptr_shared_end
    cnt_nnz_shared = 0;
    int cnt_nnz_total = 0;
    int shared_num=0;
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
    }

    printf("cnt_block=%d tilenum=%d\n", cnt_block, tilenum);
    int *d_non_each_block_offset;
    hipMalloc((void **)&d_non_each_block_offset, sizeof(int) * (tilenum + 1));
    hipMemcpy(d_non_each_block_offset, non_each_block_offset, sizeof(int) * (tilenum + 1), hipMemcpyHostToDevice);

    int *d_balance_tile_ptr_shared_end;
    hipMalloc((void **)&d_balance_tile_ptr_shared_end, sizeof(int) * (index + 1));
    hipMemcpy(d_balance_tile_ptr_shared_end, balance_tile_ptr_shared_end, sizeof(int) * (index + 1), hipMemcpyHostToDevice);
    int *d_block_signal;
    hipMalloc((void **)&d_block_signal, sizeof(int) * (tilem + 1));
    int *signal_dot;
    hipMalloc((void **)&signal_dot, sizeof(int));
    int *signal_final;
    hipMalloc((void **)&signal_final, sizeof(int));
    int *signal_final1;
    hipMalloc((void **)&signal_final1, sizeof(int));
    hipMemset(signal_final1, 0, sizeof(int));
    double *k_threshold;
    hipMalloc((void **)&k_threshold, sizeof(double));
    int *d_ori_block_signal;
    hipMalloc((void **)&d_ori_block_signal, sizeof(int) * (tilem + 1));
    hipMemcpy(d_block_signal, block_signal, sizeof(int) * (tilem + 1), hipMemcpyHostToDevice);
    hipMemcpy(d_ori_block_signal, block_signal, sizeof(int) * (tilem + 1), hipMemcpyHostToDevice);
    gettimeofday(&t6, NULL);
    double time_format = (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
    double pro_cnt = 0.0;
    unsigned char *d_blockrowid_new;
    unsigned char *d_blockcsr_ptr_new;
    int *d_nonzero_row_new;
    unsigned char *d_Tile_csr_Col;
    hipMalloc((void **)&d_blockrowid_new, sizeof(unsigned char) * (cnt_non_new + 1));
    hipMalloc((void **)&d_blockcsr_ptr_new, sizeof(unsigned char) * (cnt_non_new + 1));
    hipMalloc((void **)&d_nonzero_row_new, sizeof(int) * (tilenum + 1));
    hipMalloc((void **)&d_Tile_csr_Col, sizeof(unsigned char) * (matrix->csrsize));
    hipMemcpy(d_blockrowid_new, blockrowid_new, sizeof(unsigned char) * (cnt_non_new + 1), hipMemcpyHostToDevice);
    hipMemcpy(d_blockcsr_ptr_new, blockcsr_ptr_new, sizeof(unsigned char) * (cnt_non_new + 1), hipMemcpyHostToDevice);
    hipMemcpy(d_nonzero_row_new, nonzero_row_new, sizeof(int) * (tilenum + 1), hipMemcpyHostToDevice);
    hipMemcpy(d_Tile_csr_Col, Tile_csr_Col, sizeof(unsigned char) * (matrix->csrsize), hipMemcpyHostToDevice);
    hipMemcpy(d_tile_rowidx, tile_rowidx, sizeof(int) * (tilenum), hipMemcpyHostToDevice);
    threshold = epsilon * epsilon * s0;

    hipMemcpy(k_threshold, &threshold, sizeof(double), hipMemcpyHostToDevice);
    gettimeofday(&t1, NULL);
    // while (iterations < 1 && snew > threshold)
    {
        if (index < tilem)
        {
            int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
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
            hipMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            hipMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            double *k_q_new;
            hipMalloc((void **)&k_q_new, sizeof(double) * (rowA));
            double *k_d_new;
            hipMalloc((void **)&k_d_new, sizeof(double) * (rowA));
            double *k_r_new;
            hipMalloc((void **)&k_r_new, sizeof(double) * (rowA));
            double *k_x_new;
            hipMalloc((void **)&k_x_new, sizeof(double) * (rowA));
            

            for(int ll=0;ll<=100;ll++)
            {
            hipMemset(d_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            hipMemset(d_ori_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            hipMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), hipMemcpyHostToDevice);
            hipMemset(k_d_new, 0, (rowA) * sizeof(double));
            hipMemcpy(k_d_new, k_r, sizeof(double) * (n), hipMemcpyDeviceToDevice);
            hipMemset(k_r_new, 0, (rowA) * sizeof(double));
            hipMemcpy(k_r_new, k_r, sizeof(double) * (n), hipMemcpyDeviceToDevice);
            hipMemset(k_x_new, 0, (rowA) * sizeof(double));
            hipMemcpy(k_x_new, k_x, sizeof(double) * (n), hipMemcpyDeviceToDevice);

            hipDeviceSynchronize();
            gettimeofday(&t3, NULL);
            //下面两个为重新扩容之后的kernel保证计算的准确性
             stir_spmv_cuda_kernel_newcsr_nnz_balance_below_tilem_32_block_reduce<<<num_blocks_nnz_balance, num_threads>>>(tilem, tilenum, rowA, colA, nnzR,
                                                                                                             d_tile_ptr, d_tile_columnidx,
                                                                                                             d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                                             d_ptroffset1, d_ptroffset2,
                                                                                                             rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                                             k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                                             signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                                             k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
                                                                                                             d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,
                                                                                                             vector_each_warp_32, vector_total_32);
            hipDeviceSynchronize();
            gettimeofday(&t4, NULL);
            if(ll>0)
            time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
            if(ll<10)
            {
                time_spmv_10 += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
            }
            }
            time_spmv/=100;
            time_spmv_10/=10;
            hipFree(d_block_signal_new);
            hipFree(d_ori_block_signal_new);
            hipFree(k_q_new);
            hipFree(k_d_new);
            hipFree(k_r_new);
            hipFree(k_x_new);
        }
        else
        {
            printf("index>tilem\n");
            // 经过双指针重新分配 每个warp计算固定的非零元数目 待修改寄存器的占用率为62.5%
            int num_blocks_nnz_balance = ceil((double)(index) / (double)(num_threads / WARP_SIZE));
            
            //扩大容量
            int tilem_new=(tilem/WARP_PER_BLOCK+2)*WARP_PER_BLOCK;
            int re_size=(tilem_new)*BLOCK_SIZE;
            printf("tilem=%d tilem_new=%d tilem/WARP_PER_BLOCK=%d tilem_new/WARP_PER_BLOCK=%d\n",tilem,tilem_new,tilem/WARP_PER_BLOCK,tilem_new/WARP_PER_BLOCK);
            int *d_block_signal_new;
            hipMalloc((void **)&d_block_signal_new, sizeof(int) * (tilem_new + 1));
            int *d_ori_block_signal_new;
            hipMalloc((void **)&d_ori_block_signal_new, sizeof(int) * (tilem_new + 1));
            double *k_q_new;
            hipMalloc((void **)&k_q_new, sizeof(double) * re_size);
            double *k_d_new;
            hipMalloc((void **)&k_d_new, sizeof(double) * re_size);
            double *k_r_new;
            hipMalloc((void **)&k_r_new, sizeof(double) * re_size);
            double *k_x_new;
            hipMalloc((void **)&k_x_new, sizeof(double) * re_size);
            

            for(int u=0;u<=100;u++)
            {
            hipMemset(d_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            hipMemset(d_ori_block_signal_new,0,sizeof(int) * (tilem_new + 1));
            hipMemcpy(d_ori_block_signal_new, block_signal, sizeof(int) * (tilem + 1), hipMemcpyHostToDevice);
            hipMemset(k_q_new, 0,  re_size* sizeof(double));
            hipMemset(k_d_new, 0,  re_size* sizeof(double));
            hipMemcpy(k_d_new, k_r, sizeof(double) * (n), hipMemcpyDeviceToDevice);
            hipMemset(k_r_new, 0, re_size * sizeof(double));
            hipMemcpy(k_r_new, k_r, sizeof(double) * (n), hipMemcpyDeviceToDevice);
            hipMemset(k_x_new, 0, re_size * sizeof(double));
            hipMemcpy(k_x_new, k_x, sizeof(double) * (n), hipMemcpyDeviceToDevice);

            hipDeviceSynchronize();
            gettimeofday(&t3, NULL);   
            stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
                                                                                              d_tile_ptr, d_tile_columnidx,
                                                                                              d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
                                                                                              d_ptroffset1, d_ptroffset2,
                                                                                              rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
                                                                                              k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
                                                                                              signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
                                                                                              k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
                                                                                              d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset);
            
            hipDeviceSynchronize();
            gettimeofday(&t4, NULL);
            if(u>0)
            time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
            if(u<10)
            {
                time_spmv_10 += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
            }
            }
            time_spmv/=100;
            time_spmv_10/=10;
            // stir_spmv_cuda_kernel_newcsr_nnz_balance_redce_block_shared_queue<<<num_blocks_nnz_balance, num_threads>>>(tilem_new, tilenum, rowA, colA, nnzR,
            //                                                                                   d_tile_ptr, d_tile_columnidx,
            //                                                                                   d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
            //                                                                                   d_ptroffset1, d_ptroffset2,
            //                                                                                   rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
            //                                                                                   k_d_new, k_q_new, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col, d_block_signal_new,
            //                                                                                   signal_dot, signal_final, signal_final1, d_ori_block_signal_new,
            //                                                                                   k_alpha, k_snew, k_x_new, k_r_new, k_sold, k_beta, k_threshold,
            //                                                                                   d_balance_tile_ptr_new, d_row_each_block, d_index_each_block, index, d_non_each_block_offset,d_balance_tile_ptr_shared_end,shared_num);
            

            
            hipFree(d_block_signal_new);
            hipFree(d_ori_block_signal_new);
            hipFree(k_q_new);
            hipFree(k_d_new);
            hipFree(k_r_new);
            hipFree(k_x_new);
        }
        hipMemcpy(&snew, k_snew, sizeof(double), hipMemcpyDeviceToHost);
        printf("final residual=%e\n", sqrt(snew));
        // iterations++;
    }
    hipDeviceSynchronize();
    gettimeofday(&t2, NULL);
    hipMemcpy(x, k_x, sizeof(double) * (n), hipMemcpyDeviceToHost);
    double time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("iter=%d,time_cg=%lf ms,time_spmv=%lf ms,time_format=%lf ms\n", iterations, time_spmv, time_spmv, time_format);
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
    printf("%e\n",sqrt(snew));
    printf("%e\n",sqrt(sum));
    printf("%e\n",sqrt(l2_norm));

    // double Gflops_spmv= (2 * nnzR) / ((time_spmv/ iterations)*pow(10, 6));
    // printf("Gflops_Initial=%lf\n",Gflops_spmv);
    // printf("L2 Norm is %lf\n", l2_norm);
    char *s = (char *)malloc(sizeof(char) * 200);
    if(time_spmv>time_spmv_10)
    time_spmv=time_spmv_10;
    //sprintf(s, "iter=%d,time_spmv=%.3f,time_cg=%.3f,nnzR=%d,l2_norm=%e,time_format=%lf,index=%d,each_nnz=%d\n", iterations, time_spmv, time_cg, nnzR, l2_norm, time_format, index, each_block_nnz);
    sprintf(s, "%d,%.3f,%.3f,%d,%e,%lf,%d,%d,%e\n", iterations, time_spmv,time_spmv, nnzR, l2_norm, time_format, index, each_block_nnz,sqrt(snew));
    FILE *file1 = fopen("cg_syncfree_amd_mi200.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
    hipFree(k_val);
    hipFree(k_b);
    hipFree(k_x);
    hipFree(k_r);
    hipFree(k_d);
    hipFree(k_q);
    hipFree(k_alpha);
    hipFree(k_snew);
    hipFree(k_sold);
    hipFree(k_beta);
    hipFree(k_s0);
    hipFree(d_tile_ptr);
    hipFree(d_tile_columnidx);
    hipFree(d_csr_compressedIdx);
    hipFree(d_Blockcsr_Val);
    hipFree(d_Blockcsr_Ptr);
    hipFree(d_blkcoostylerowidx);
    hipFree(d_blkcoostylerowidx_colstart);
    hipFree(d_blkcoostylerowidx_colstop);
    hipFree(d_ptroffset1);
    hipFree(d_ptroffset2);
    hipFree(d_x);
    hipFree(d_y);
    hipFree(d_blockrowid_new);
    hipFree(d_blockcsr_ptr_new);
    hipFree(d_nonzero_row_new);
    hipFree(d_Tile_csr_Col);
    hipFree(d_tile_rowidx);
    hipFree(d_block_signal);
    hipFree(d_ori_block_signal);
    hipFree(signal_final1);
    hipFree(signal_final);
    hipFree(signal_dot);
    hipFree(d_balance_tile_ptr_shared_end);
    hipFree(d_non_each_block_offset);
    hipFree(d_row_each_block);
    hipFree(d_index_each_block);
    hipFree(d_balance_tile_ptr_new);
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
    free(r);
    free(nonzero_row_new);
    free(blockrowid_new);
    free(blockcsr_ptr_new);
    free(block_signal);
    free(non_each_block);
    free(non_each_block_offset);
    free(row_each_block);
    free(index_each_block);
    free(row_each_block_new);
    free(index_each_block_new);
    free(non_each_block_new);
    free(balance_tile_ptr_new);
    free(balance_tile_ptr_shared_end);
}
int main(int argc, char **argv)
{
    char *filename = argv[1];
    int block_nnz = atoi(argv[2]);
    // char *file_rhs = argv[2];
    int m, n, nnzR, isSymmetric;
    mmio_info(&m,&n,&nnzR,&isSymmetric, filename);
    int *RowPtr=(int *)malloc(sizeof(int)*(n+1));
    int *ColIdx=(int *)malloc(sizeof(int)*(nnzR));
    double *Val=(double *)malloc(sizeof(double)*(nnzR));
    mmio_data(RowPtr, ColIdx, Val, filename);
    // int *RowPtr;
    // int *ColIdx;
    // MAT_VAL_TYPE *Val;
    //read_Dmatrix_32(&m, &n, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
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
    hipGetDevice(&device);
    //show_GPU_info();
    int maxWarps;
    hipDeviceGetAttribute(&maxWarps, hipDeviceAttributeMaxThreadsPerMultiProcessor, device);

    printf("Max warps per multiprocessor: %d\n", maxWarps);

    int maxSMs;
    hipDeviceGetAttribute(&maxSMs, hipDeviceAttributeMultiprocessorCount, device);

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
            Y_golden[i] += Val[j] * X[ColIdx[j]];
    // for (int i = 0; i < n; i++)
    // {
    //   fscanf(fp1, "%lf", &X[i]);
    //
    // }

    cg_solve_inc(RowPtr, ColIdx, Val, Val_Low, X, Y_golden, n, &iter, 10, 1e-5, filename, nnzR, ori, block_nnz);
    // cg_cusparse(filename,RowPtr,ColIdx,Val,Y_golden,n,X,nnzR);
}
