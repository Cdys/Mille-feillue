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

__global__ void sdot2_2(double *a, double *b, double *c, int n)
{

    // Define variables.
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // index/16
    // d_vis[index/16]==1
    int stride = blockDim.x * gridDim.x;
    double temp;
    temp = 0;
    // Define shared memories.
    //__shared__ double s_data[1024];
    __shared__ double s_data[256];//根据BlockDim的大小决定
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
    //if (tid==0)
    if (tid == 0&&s_data[0]!=0)
    {
        atomicAdd(c, s_data[0]);
    }
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


__global__ void te1(double *p, double *last_val, int *vis_new, int *vis_mix, unsigned int *vis_mix_16, unsigned int *vis_mix_32)
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
        vis_mix[blockIdx.x] = 1;
        // vis_mix_16[blockIdx.x/16] |= (1 << (blockIdx.x % 16));
        atomicOr(&(vis_mix_16[blockIdx.x / 16]), (1 << (blockIdx.x % 16)));
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

int bicgstab(int *RowPtr, int *ColIdx, double *Val, double *rhs, double *x,
             int nnzR, char *filename, int n) // bicg
{
    int *dA_csrOffsets, *dA_columns;
    double *dA_values;
    cudaMalloc((void **)&dA_csrOffsets,
                           (n + 1) * sizeof(int));
    cudaMalloc((void **)&dA_columns, nnzR * sizeof(int));
    cudaMalloc((void **)&dA_values, nnzR * sizeof(double));
    cudaMemcpy(dA_csrOffsets, RowPtr,(n + 1) * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dA_columns, ColIdx, nnzR * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dA_values, Val, nnzR * sizeof(double),cudaMemcpyHostToDevice);
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
    //printf("rowA=%d,rowblkblock=%d\n", tilem, rowblkblock);
    char *tilewidth = matrix->tilewidth;
    int *csr_offset = matrix->csr_offset;
    int *csrptr_offset = matrix->csrptr_offset;
    MAT_VAL_TYPE *Blockcsr_Val = matrix->Blockcsr_Val;
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
    cudaMalloc((void **)&d_tile_rowidx, tilenum * sizeof(int));
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

    // printf("duandian\n");

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
    double time_bicg = 0;
    double time_spmv = 0;
    double time_dot = 0;
    double time_sptrsv = 0;
    struct timeval t1, t2, t3, t4, t5, t6,t7,t8;
    double *rg, *rh, *pg, *ph, *sg, *sh, *tg, *vg, *tp;
    double *k_rg, *k_rh, *k_pg, *k_ph, *k_sg, *k_sh, *k_tg, *k_vg, *k_tp;
    double r0 = 0, r1 = 0, pra = 0, prb = 0, prc = 0;
    double residual, err_rel = 0;
    double *k_r0, *k_r1, *k_pra, *k_prb, *k_prc;
    double *k_residual, *k_err_rel;
    double *k_x;
    double *k_tmp1, *k_tmp2;
    int i, retval = 0;
    int itr = 0.;
    double tol = 1e-10;
    int maxits = 1000;
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
    cudaMalloc((void **)&k_ph, sizeof(double) * n);
    cudaMalloc((void **)&k_sg, sizeof(double) * n);
    cudaMalloc((void **)&k_sh, sizeof(double) * n);
    cudaMalloc((void **)&k_tg, sizeof(double) * n);
    cudaMalloc((void **)&k_vg, sizeof(double) * n);
    cudaMalloc((void **)&k_tp, sizeof(double) * n);
    cudaMalloc((void **)&k_x, sizeof(double) * n);

    cudaMalloc((void **)&k_r0, sizeof(double));
    cudaMalloc((void **)&k_r1, sizeof(double));
    cudaMalloc((void **)&k_pra, sizeof(double));
    cudaMalloc((void **)&k_prb, sizeof(double));
    cudaMalloc((void **)&k_prc, sizeof(double));
    cudaMalloc((void **)&k_residual, sizeof(double));
    cudaMalloc((void **)&k_err_rel, sizeof(double));
    cudaMalloc((void **)&k_tmp1, sizeof(double));
    cudaMalloc((void **)&k_tmp2, sizeof(double));
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

    // dim3 BlockDim(128);
    // dim3 GridDim((n/128+1));

    // dim3 BlockDim(256);
    // dim3 GridDim((n/256+1));

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
    cudaMemcpy(k_pra, &pra, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prb, &prb, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_prc, &prc, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_residual, &residual, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(k_err_rel, &err_rel, sizeof(double), cudaMemcpyHostToDevice);

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseDnVecDescr_t vecX1, vecY1;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, n, n, nnzR,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, n, k_pg, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, n, k_vg, CUDA_R_64F);

    cusparseCreateDnVec(&vecX1, n, k_sg, CUDA_R_64F);
    cusparseCreateDnVec(&vecY1, n, k_tg, CUDA_R_64F);
    double alpha = 1.0f;
    double beta = 0.0f;
    cusparseSpMV_bufferSize(
         handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
         CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    for (itr = 0; itr < 10; itr++)
    {
        scalarassign(k_r0, k_r1);
        cudaMemset(k_vg, 0, n * sizeof(double));
        cudaMemset(k_tg, 0, n * sizeof(double));
        cudaDeviceSynchronize();
        gettimeofday(&t3, NULL);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        // stir_spmv_cuda_kernel_newcsr<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnz,
        //                                                               d_tile_ptr, d_tile_columnidx,
        //                                                               d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
        //                                                               d_ptroffset1, d_ptroffset2,
        //                                                               rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
        //                                                               k_pg, k_vg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col);
        cudaDeviceSynchronize();
        gettimeofday(&t4, NULL);
        time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
        cudaDeviceSynchronize();
        gettimeofday(&t7, NULL);
        cudaMemset(k_pra, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_rh, k_vg, k_pra, n);
        //cublasDdot(cublasHandle, n, k_rh, 1, k_vg, 1, k_pra);
        cudaDeviceSynchronize();
        gettimeofday(&t8, NULL);
        time_dot += (t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;
        scalardiv<<<1, 1>>>(k_pra, k_r1);
        //sg=rg-pra*vg
        yminus_mult<<<GridDim, BlockDim>>>(n, k_sg, k_rg, k_vg, k_pra);
        cudaDeviceSynchronize();
        gettimeofday(&t5, NULL);
        //tg=A*sg
        // stir_spmv_cuda_kernel_newcsr<<<num_blocks, num_threads>>>(tilem, tilen, rowA, colA, nnz,
        //                                                             d_tile_ptr, d_tile_columnidx,
        //                                                             d_csr_compressedIdx, d_Blockcsr_Val, d_Blockcsr_Ptr,
        //                                                             d_ptroffset1, d_ptroffset2,
        //                                                             rowblkblock, d_blkcoostylerowidx, d_blkcoostylerowidx_colstart, d_blkcoostylerowidx_colstop,
        //                                                             k_sg, k_tg, d_blockrowid_new, d_blockcsr_ptr_new, d_nonzero_row_new, d_Tile_csr_Col);
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha, matA, vecX1, &beta, vecY1, CUDA_R_64F,
                            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
        cudaDeviceSynchronize();
        gettimeofday(&t6, NULL);
        time_spmv += (t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;
        cudaDeviceSynchronize();
        gettimeofday(&t7, NULL);
        //ktmp1=tg*sg;
        //ktmp2=tg*tg; 下面的两个点积fusion一下
        cudaMemset(k_tmp1, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_tg, k_sg, k_tmp1, n);
        //cublasDdot(cublasHandle, n, k_tg, 1, k_sg, 1, k_tmp1);
        cudaMemset(k_tmp2, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_tg, k_tg, k_tmp2, n);
        //cublasDdot(cublasHandle, n, k_tg, 1, k_tg, 1, k_tmp2);
        cudaDeviceSynchronize();
        gettimeofday(&t8, NULL);
        time_dot += (t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;
        //prc=tmp1/tmp2;
        scalardiv_new<<<1, 1>>>(k_prc, k_tmp1, k_tmp2);
        //x=x+pra*pg+prc*sg
        //rg=sg-prc*tg;
        yminus_mult_new<<<GridDim, BlockDim>>>(n, k_x, k_pg, k_sg, k_rg, k_tg, k_pra, k_prc);
        //residual=sqrt(rg*rg); 和下面的点积fusion一下
        cudaMemset(k_residual, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_rg, k_rg, k_residual, n);
        cudaMemcpy(&residual, k_residual, sizeof(double), cudaMemcpyDeviceToHost);
        residual=sqrt(residual);
        //cublasDnrm2(cublasHandle, n, k_rg, 1, &residual);
        cudaDeviceSynchronize();
        gettimeofday(&t7, NULL);
        //r1=rg*rh;
        cudaMemset(k_r1, 0, sizeof(double));
        sdot2_2<<<GridDim, BlockDim>>>(k_rg, k_rh, k_r1, n);
        //cublasDdot(cublasHandle, n, k_rg, 1, k_rh, 1, k_r1);
        cudaDeviceSynchronize();
        gettimeofday(&t8, NULL);
        time_dot += (t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;
        //prb=r1*pra/r0*prc
        scalardiv_five<<<1, 1>>>(k_prb, k_r1, k_pra, k_r0, k_prc);
        //pg=rg+prb*(pg-prc*vg);
        yminus_final<<<GridDim, BlockDim>>>(n, k_pg, k_rg, k_prb, k_prc, k_vg);
        printf("%d %e\n", itr, residual / err_rel);
        if (residual <= tol)
            break;
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    time_bicg = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    // double time_overall_serial = time_bicg / itr;
    // double GFlops_serial = 2 * nnzR / time_overall_serial / pow(10, 6);
    double Gflops_spmv= (2 * nnzR) / ((time_spmv/ itr/2)*pow(10, 6));
    double Gflops_bicg= (2 * nnzR) / ((time_bicg/ itr) * pow(10, 6));
    double sum_ori = 0;
    for (int i = 0; i < n; i++)
    {
        sum_ori = sum_ori + (rhs[i] * rhs[i]);
    }
    double norm=residual;
    double l2_norm = residual / sqrt(sum_ori);
    printf("time_bicg=%lf ms,time_spmv=%lf ms, l2_norm=%lf,time_dot=%lf\n", time_bicg, time_spmv,l2_norm,time_dot);
    printf("Gflops_bicg=%lf,Gflops_spmv=%lf, l2_norm=%e,norm=%e\n", Gflops_bicg, Gflops_spmv,l2_norm,norm);
    char *s = (char *)malloc(sizeof(char) * 200);
    sprintf(s, "iter=%d,time_spmv=%.3f,time_bicg=%.3f,nnzR=%d,l2_norm=%e,time_dot=%lf,norm=%e\n", itr, time_spmv, time_bicg, nnzR,l2_norm,time_dot,norm);
    FILE *file1 = fopen("bicg_cusparse.csv", "a");
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
    bicgstab(RowPtr, ColIdx, Val, rhs1, x1, nnzR, filename, m);
    fclose(p);
    free(x1);
    free(rhs1);
    return 0;
}
