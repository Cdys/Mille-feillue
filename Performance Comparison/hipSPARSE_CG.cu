#include "hip/hip_runtime.h"
#include <stdio.h>
#include <hip/hip_fp16.h>
#include <sys/time.h>
#include <omp.h>
#include "csr2block.h"
#include "blockspmv_cpu.h"
#include "utils.h"
#include <hipblas.h>
#include <hip/hip_runtime.h>
#include <hipsparse.h>
#include <hip/hip_runtime_api.h>
// #include <helper_functions.h> 
// #include <helper_cuda.h>
// #include "./biio2.0/src/biio.h"
#include "common.h"
#include "mmio_highlevel.h"
#include <hipsparse.h>

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
        y[tid] = half(x[tid]);
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
    hipMemcpy(dest, src, sizeof(double), hipMemcpyDeviceToDevice);
}

// Sets dest=src for n-length vectors on the GPU.
void vecassign(double *dest, double *src, int n)
{
    hipMemcpy(dest, src, sizeof(double) * n, hipMemcpyDeviceToDevice);
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
extern "C" void cg_solve_inc(int *RowPtr, int *ColIdx, MAT_VAL_TYPE *Val, MAT_VAL_LOW_TYPE *Val_Low, double *x, double *b, int n, int *iter, int maxiter, double threshold, char *filename, int nnzR, int ori)
{
    int *dA_csrOffsets, *dA_columns;
    double *dA_values;
    hipMalloc((void **)&dA_csrOffsets,
                           (n + 1) * sizeof(int));
    hipMalloc((void **)&dA_columns, nnzR * sizeof(int));
    hipMalloc((void **)&dA_values, nnzR * sizeof(double));
    hipMemcpy(dA_csrOffsets, RowPtr,(n + 1) * sizeof(int),hipMemcpyHostToDevice);
    hipMemcpy(dA_columns, ColIdx, nnzR * sizeof(int),hipMemcpyHostToDevice);
    hipMemcpy(dA_values, Val, nnzR * sizeof(double),hipMemcpyHostToDevice);
    hipblasHandle_t cublasHandle = 0;
    hipblasStatus_t hipblasStatus_t;
    hipblasStatus_t = hipblasCreate(&cublasHandle);
    struct timeval t1, t2, t3, t4,t5,t6;
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

    double *k_b, *k_x, *k_r, *k_d, *k_q, *k_s;
    double *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;
    double t, s0, snew;
    double *k_alpha_new;
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
    hipMalloc((void **)&k_alpha_new, sizeof(double));
    hipMalloc((void **)&k_snew, sizeof(double) * NUM_BLOCKS);
    hipMalloc((void **)&k_sold, sizeof(double));
    hipMalloc((void **)&k_beta, sizeof(double));
    hipMalloc((void **)&k_s0, sizeof(double));
    double *r = (double *)malloc(sizeof(double) * (n + 1));
    memset(r, 0, sizeof(double) * (n + 1));
    dim3 BlockDim(NUM_THREADS);
    dim3 GridDim(NUM_BLOCKS);//这里改一下
    // dim3 BlockDim(128);
    // dim3 GridDim((n/128+1));
    //dim3 BlockDim(256);
    //dim3 GridDim((n/256+1));

    veczero<<<1, BlockDim>>>(n, k_x);
    // r=b-Ax (r=b since x=0), and d=M^(-1)r
    hipMemcpy(k_r, k_b, sizeof(double) * (n), hipMemcpyDeviceToDevice);
    hipblasDdot(cublasHandle, n, k_r, 1, k_r, 1, k_s0);
    // r[n] = 1.1;
    hipMemcpy(k_d, k_r, sizeof(double) * (n + 1), hipMemcpyDeviceToDevice);
    // r[n] = 1.2;
    //  snew = s0
    scalarassign(k_snew, k_s0);
    // Copy snew and s0 back to host so that host can evaluate stopping condition
    hipMemcpy(&snew, k_snew, sizeof(double), hipMemcpyDeviceToHost);
    hipMemcpy(&s0, k_s0, sizeof(double), hipMemcpyDeviceToHost);
    //printf("begin GPU CG\n");
    // hipMemset(d_vis_new, 0, num_seg * sizeof(int));
    // hipMemset(d_vis_mix, 0, num_seg * sizeof(int));
    double time_spmv = 0;


    printf("num_thread=%d\n",omp_get_max_threads());
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
    int cnt_non_new = nonzero_row_new[tilenum];
    unsigned char *blockrowid_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockrowid_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    unsigned char *blockcsr_ptr_new = (unsigned char *)malloc(sizeof(unsigned char) * (cnt_non_new + 1));
    memset(blockcsr_ptr_new, 0, sizeof(unsigned char) * (cnt_non_new + 1));
    int csrcount_new1 = 0;
// #pragma omp parallel for
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
    gettimeofday(&t6, NULL);
    double time_format= (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
    double pro_cnt=0.0;
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
    hipsparseHandle_t handle = NULL;
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    hipsparseCreate(&handle);
    // hipsparseCreateCsr(&matA, n, n, nnzR,
    //                                   dA_csrOffsets, dA_columns, dA_values,
    //                                   HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
    //                                   HIPSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE_AX);
     
    // hipsparseCreateDnVec(&vecX, n, k_d, HIP_R_64F);
    // hipsparseCreateDnVec(&vecY, n, k_q, HIP_R_64F);
    double alpha = 1.0f;
    double beta = 0.0f;
    hipsparseMatDescr_t descrA;
    hipsparseCreateMatDescr(&descrA);
    //warm up
    for(int i = 0; i < 1; ++i)
    {
        hipsparseDcsrmv(handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,
            n,n,nnzR,&alpha,
            descrA,dA_values,dA_csrOffsets,dA_columns,k_d,&beta,k_q);  
        hipblasDdot(cublasHandle, n, k_d, 1, k_q, 1, k_alpha_new);   
    }                 
    hipDeviceSynchronize();
    gettimeofday(&t1, NULL);
    while (iterations < 5) //&& snew > epsilon * epsilon * s0)
    {
        // q = Ad
        hipMemset(k_q, 0, n * sizeof(double));
        hipDeviceSynchronize();
        gettimeofday(&t3, NULL);
        hipsparseDcsrmv(handle,HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        n,n,nnzR,&alpha,
                        descrA,dA_values,dA_csrOffsets,dA_columns,k_d,&beta,k_q);                      
        hipDeviceSynchronize();
        gettimeofday(&t4, NULL);
        time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;

        hipblasDdot(cublasHandle, n, k_d, 1, k_q, 1, k_alpha);
        scalardiv<<<1, 1>>>(k_snew, k_alpha, k_alpha);

        // x = x + alpha*d
        axpy<<<GridDim, BlockDim>>>(n, k_alpha, k_d, k_x, k_x);

        // r = r - alpha*q
        ymax<<<GridDim, BlockDim>>>(n, k_alpha, k_q, k_r);
        scalarassign(k_sold, k_snew);
        hipblasDdot(cublasHandle, n, k_r, 1, k_r, 1, k_snew);
        // beta = snew/sold
        scalardiv<<<1, 1>>>(k_snew, k_sold, k_beta);
        // d = r + beta*d
        axpy<<<GridDim, BlockDim>>>(n, k_beta, k_d, k_r, k_d);
        // record the last val
        //  Copy back snew so the host can evaluate the stopping condition
        hipMemcpy(&snew, k_snew, sizeof(double), hipMemcpyDeviceToHost);
        printf("%e\n", sqrt(snew));
        iterations++;
    }
    hipDeviceSynchronize();
    hipMemcpy(x, k_x, sizeof(double) * (n), hipMemcpyDeviceToHost);
    gettimeofday(&t2, NULL);
    double time_cg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("iter=%d,time_cg=%lf ms,time_spmv=%lf ms,time_format=%lf ms\n",iterations, time_cg, time_spmv,time_format);
    double *b_new = (double *)malloc(sizeof(double) * n);
    memset(b_new, 0, sizeof(double) * n);
    //printf("debug\n");
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
            for (int ri = nonzero_row_new[blkj]; ri < nonzero_row_new[blkj + 1]; ri++)
            {
                double sum_new = 0;
                int ro = blockrowid_new[ri + 1];
                for (int rj = blockcsr_ptr_new[ri]; rj < blockcsr_ptr_new[ri + 1]; rj++)
                {
                    int csrcol = Tile_csr_Col[csroffset + rj];
                    sum_new += x[x_offset + csrcol] * matrix->Blockcsr_Val[csroffset + rj];
                }
                b_new[blki * BLOCK_SIZE + ro] += sum_new;
            }
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
    //double Gflops_spmv= (2 * nnzR) / ((time_spmv/ iterations)*pow(10, 6));
    //printf("Gflops_Initial=%lf\n",Gflops_spmv);
    //printf("L2 Norm is %lf\n", l2_norm);
    char *s = (char *)malloc(sizeof(char) * 200);
    sprintf(s, "%d,%.3f,%.3f,%d,%lf\n", iterations, time_spmv, time_cg, nnzR, l2_norm);
    FILE *file1 = fopen("cg_cusparse_amd_mi200_5.csv", "a");
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
  //char *file_rhs = argv[2];
  int m, n, nnzR, isSymmetric;
//   int *RowPtr;
//   int *ColIdx;
//   MAT_VAL_TYPE *Val;
//   read_Dmatrix_32(&m, &n, &nnzR, &RowPtr, &ColIdx, &Val, &isSymmetric, filename);
mmio_info(&m,&n,&nnzR,&isSymmetric, filename);
    int *RowPtr=(int *)malloc(sizeof(int)*(n+1));
    int *ColIdx=(int *)malloc(sizeof(int)*(nnzR));
    double *Val=(double *)malloc(sizeof(double)*(nnzR));
    mmio_data(RowPtr, ColIdx, Val, filename);
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
