/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
 #include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
 #include <cusparse.h>         // cusparseSpMV
 #include <stdio.h>            // printf
 #include <stdlib.h>           // EXIT_FAILURE
 
 #include <cuda_fp16.h>
 #include <cuda_bf16.h>
 #include <stdint.h>
 
 #include <iostream>
 
 #include "biio.h"
 
 #define THREADS_PER_BLOCK 32
 #define THREADS_PER_VECTOR 32
 
 texture<int, 1> tex_colm;
 texture<int2, 1> tex_val;
 double *scratchpad;
 #define NUM_THREADS 128
 #define NUM_BLOCKS 16
 
 #define THREAD_ID threadIdx.x + blockIdx.x *blockDim.x
 #define THREAD_COUNT gridDim.x *blockDim.x
 
 #define epsilon 1e-6
 #define IMAX 1000
 #define MAT_VAL_TYPE double
 double utime()
 {
     struct timeval tv;
 
     gettimeofday(&tv, NULL);
 
     return (tv.tv_sec + double(tv.tv_usec) * 1e-6);
 }
 static __device__ double fetch_double(texture<int2, 1> val, int elem)
 {
     int2 v = tex1Dfetch(val, elem);
     return __hiloint2double(v.y, v.x);
 }
 __global__ void csr_spmv(int n, double *src, double *dest, int *findrm)
 {
     for (int row = THREAD_ID; row < n; row += THREAD_COUNT)
     {
         dest[row] = 0;
         int a = findrm[row];
         int b = findrm[row + 1];
         for (int k = a; k < b; k++)
             dest[row] += fetch_double(tex_val, k - 1) * src[tex1Dfetch(tex_colm, k - 1) - 1];
     }
 }

//  size_t bytes_per_spmv(const csr_matrix<int, float> &mtxS, const csr_matrix<int, double> &mtxD)
//  {
//      size_t bytes = 0;
//      bytes += 2 * sizeof(int) * mtxS.num_rows;        // row pointer
//      bytes += 1 * sizeof(int) * mtxS.num_nonzeros;    // column index
//      bytes += 1 * sizeof(int) * mtxD.num_nonzeros;    // column index
//      bytes += 2 * sizeof(float) * mtxS.num_nonzeros;  // A[i,j] and x[j]
//      bytes += 2 * sizeof(double) * mtxD.num_nonzeros; // A[i,j] and x[j]
//      bytes += 2 * sizeof(double) * mtxD.num_rows;     // y[i] = y[i] + ...
//      return bytes;
//  }
 
 void scalarassign(double *dest, double *src)
 {
     cudaMemcpy(dest, src, sizeof(double), cudaMemcpyDeviceToDevice);
 }
 __global__ void ymax(int n, double *a, double *x, double *y)
 {
     for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
         y[i] = y[i] - (*a) * x[i];
 }
 
 __global__ void axpy(int n, double *a, double *x, double *y, double *r)
 {
     for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
         r[i] = y[i] + (*a) * x[i];
 }
 
 __global__ void vecdot_reduce(double *partial, double *result)
 {
     __shared__ double tmp[NUM_BLOCKS];
 
     if (threadIdx.x < NUM_BLOCKS)
         tmp[threadIdx.x] = partial[threadIdx.x];
     else
         tmp[threadIdx.x] = 0;
 
     for (int i = blockDim.x / 2; i >= 1; i = i / 2)
     {
         __syncthreads();
         if (threadIdx.x < i)
             tmp[threadIdx.x] += tmp[i + threadIdx.x];
     }
 
     if (threadIdx.x == 0)
         *result = tmp[0];
 }
 __global__ void vecdot_partial(int n, double *vec1, double *vec2, double *partial)
 {
     __shared__ double tmp[NUM_THREADS];
     tmp[threadIdx.x] = 0;
 
     for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
         tmp[threadIdx.x] += vec1[i] * vec2[i];
 
     for (int i = blockDim.x / 2; i >= 1; i = i / 2)
     {
         __syncthreads();
         if (threadIdx.x < i)
             tmp[threadIdx.x] += tmp[i + threadIdx.x];
     }
 
     if (threadIdx.x == 0)
         partial[blockIdx.x] = tmp[0];
 }
 __global__ void scalardiv(double *num, double *den, double *result)
 {
     if (threadIdx.x == 0 && blockIdx.x == 0)
         *result = (*num) / (*den);
 }
 void vecdot(int n, double *vec1, double *vec2, double *result)
 {
     dim3 BlockDim(NUM_THREADS);
     dim3 GridDim(NUM_BLOCKS);
 
     vecdot_partial<<<GridDim, BlockDim>>>(n, vec1, vec2, scratchpad);
     vecdot_reduce<<<1, NUM_BLOCKS>>>(scratchpad, result);
 }
 __global__ void veczero(int n, double *vec)
 {
     for (int i = THREAD_ID; i < n; i += THREAD_COUNT)
         vec[i] = 0;
 }
 __global__ void device_convert(double *x, float *y, int n)
 {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < n)
     {
         y[tid] = x[tid];
     }
 }
 __global__ void add_mix(double *y, float *y_float, int n)
 {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < n)
     {
         y[tid] = y[tid] + y_float[tid];
     }
 }
 #define CHECK_CUDA(func)                                               \
     {                                                                  \
         cudaError_t status = (func);                                   \
         if (status != cudaSuccess)                                     \
         {                                                              \
             printf("CUDA API failed at line %d with error: %s (%d)\n", \
                    __LINE__, cudaGetErrorString(status), status);      \
             return EXIT_FAILURE;                                       \
         }                                                              \
     }
 
 #define CHECK_CUSPARSE(func)                                               \
     {                                                                      \
         cusparseStatus_t status = (func);                                  \
         if (status != CUSPARSE_STATUS_SUCCESS)                             \
         {                                                                  \
             printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                    __LINE__, cusparseGetErrorString(status), status);      \
             return EXIT_FAILURE;                                           \
         }                                                                  \
     }
 
 extern "C" void gpucg_solve_(char *filename,int *row_ptr,int *col_idx,double *val,
                              double *b_p, int n, double *x_p,int nnz)
 {
     int *dA_csrOffsets, *dA_columns;
     VALUE_TYPE_AX *dA_values, *dX, *dY;
     int memory_all=0;
     double cnt_memory=0;
     cudaMalloc((void **)&dA_csrOffsets,
                           (n + 1) * sizeof(int));
     cudaMalloc((void **)&dA_columns, nnz * sizeof(int));
     cudaMalloc((void **)&dA_values, nnz * sizeof(VALUE_TYPE_AX));
     memory_all+=((n + 1) * sizeof(int));
     memory_all+=(nnz * sizeof(int));
     memory_all+=(nnz * sizeof(VALUE_TYPE_AX));
     cnt_memory+=(double)memory_all/1048576;
     memory_all=0;
     cudaMemcpy(dA_csrOffsets, row_ptr,(n + 1) * sizeof(int),cudaMemcpyHostToDevice);
     cudaMemcpy(dA_columns, col_idx, nnz * sizeof(int),cudaMemcpyHostToDevice);
     cudaMemcpy(dA_values, val, nnz * sizeof(VALUE_TYPE_AX),cudaMemcpyHostToDevice);
     double *k_b, *k_x, *k_r, *k_d, *k_q, *k_s;
     // Diagonal matrix on the GPU (stored as a vector)
     double *k_jac;
     // Scalars on the GPU
     double *k_alpha, *k_snew, *k_beta, *k_sold, *k_s0;
 
     double *test_x = (double *)malloc(sizeof(double) * (n));
     double *test_x_r = (double *)malloc(sizeof(double) * (n));
     // Scalars on the host
     double t, s0, snew;
     int iterations = 0;
     cusparseHandle_t handle = NULL;
     cusparseSpMatDescr_t matA;
     cusparseDnVecDescr_t vecX, vecY;
     void *dBuffer = NULL;
     size_t bufferSize = 0;
     // Begin timing
     // for(int i=0;i<(n);i++)
     //   printf("%lf\n",b_p[i]);
     // Allocate space on the GPU for the CSR matrix and RHS vector, and copy from host to GPU
     
     cudaMalloc((void **)&k_b, sizeof(double) * (n));
     cudaMemcpy(k_b, b_p, sizeof(double) * (n), cudaMemcpyHostToDevice);
 
     // Allocate space for vectors on the GPU
     cudaMalloc((void **)&k_x, sizeof(double) * (n));
     cudaMalloc((void **)&k_r, sizeof(double) * (n));
     cudaMalloc((void **)&k_d, sizeof(double) * (n));
     cudaMalloc((void **)&k_q, sizeof(double) * (n));
     cudaMalloc((void **)&k_s, sizeof(double) * (n));
     cudaMalloc((void **)&k_jac, sizeof(double) * (n));
     cudaMalloc((void **)&k_alpha, sizeof(double));
     cudaMalloc((void **)&scratchpad, sizeof(double) * NUM_BLOCKS);
     cudaMalloc((void **)&k_snew, sizeof(double) * NUM_BLOCKS);
     cudaMalloc((void **)&k_sold, sizeof(double));
     cudaMalloc((void **)&k_beta, sizeof(double));
     cudaMalloc((void **)&k_s0, sizeof(double));
 
     // Dimensions of blocks and grid on the GPU
     dim3 BlockDim(NUM_THREADS);
     dim3 GridDim(NUM_BLOCKS);
 
     // Bind the matrix to the texture cache - this was not done earlier as we modified the matrix
     // cudaBindTexture(NULL, tex_val, k_val, sizeof(double) * (*matrix_val_size));
 
     // Initialise result vector (x=0)
     veczero<<<1, BlockDim>>>(n, k_x);
     // int n = n;
     //  // r=b-Ax (r=b since x=0), and d=M^(-1)r
     cudaMemcpy(k_r, k_b, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
     vecdot(n, k_r, k_r, k_s0);
     cudaMemcpy(k_d, k_r, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
     // snew = s0
     scalarassign(k_snew, k_s0);
 
     // Copy snew and s0 back to host so that host can evaluate stopping condition
     cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
     cudaMemcpy(&s0, k_s0, sizeof(double), cudaMemcpyDeviceToHost);
     //printf("s0=%lf\n", s0);
     cusparseCreate(&handle);
     // Create sparse matrix A in CSR format

     cusparseCreateCsr(&matA, n, n, nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE_AX);
     cudaDeviceSynchronize();
     struct timeval start, end;
     cudaDeviceSynchronize();
     cusparseCreateDnVec(&vecX, n, k_d, COMPUTE_TYPE_AX);
     // Create dense vector y
     cusparseCreateDnVec(&vecY, n, k_q, COMPUTE_TYPE_Y);
     // allocate an external buffer if needed
     ALPHA_TYPE alpha = 1.0f;
     ALPHA_TYPE beta = 0.0f;
     cusparseSpMV_bufferSize(
         handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
         &alpha, matA, vecX, &beta, vecY, COMPUTE_TYPE,
         CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
     cudaMalloc(&dBuffer, bufferSize);
     memory_all+=(bufferSize*sizeof(dBuffer));//buffersize=0?
     cnt_memory+=(double)memory_all/1048576;
     //printf("%d,%d\n",memory_all,bufferSize);
     printf("memory_MB=%lf\n",cnt_memory);
     //  While i < imax and snew > epsilon^2*s0
     //gettimeofday(&t1, NULL);
     double time_spmv = 0.0;
     struct timeval t3, t4;
     t = -utime();
     while (iterations < 1)
     {
         cudaMemset(k_q, 0, n * sizeof(double));
         cudaDeviceSynchronize();
         gettimeofday(&t3, NULL);
         cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY, COMPUTE_TYPE,
            CUSPARSE_MV_ALG_DEFAULT, dBuffer);
         cudaDeviceSynchronize();
         gettimeofday(&t4, NULL);
         time_spmv += (t4.tv_sec - t3.tv_sec) * 1000.0 + (t4.tv_usec - t3.tv_usec) / 1000.0;
         // q = Ad
         // csr_spmv<<<GridDim, BlockDim>>>(n, k_d, k_q, k_findrm);
         // alpha = snew/(d.q)
         //cudaMemcpy(b_p, k_q, sizeof(double) * (n), cudaMemcpyDeviceToHost);
        //  for(int i=0;i<n;i++)
        //  printf("%lf\n",b_p[i]);
        //  vecdot(n, k_d, k_q, k_alpha);
        //  scalardiv<<<1, 1>>>(k_snew, k_alpha, k_alpha);
        //  // x = x + alpha*d
        //  axpy<<<GridDim, BlockDim>>>(n, k_alpha, k_d, k_x, k_x);
        //  // r = r - alpha*q
        //  ymax<<<GridDim, BlockDim>>>(n, k_alpha, k_q, k_r);
        //  // sold = snew
        //  scalarassign(k_sold, k_snew);
        //  // snew = r.r
        //  vecdot(n, k_r, k_r, k_snew);
        //  // beta = snew/sold
        //  scalardiv<<<1, 1>>>(k_snew, k_sold, k_beta);
        //  // d = r + beta*d
        //  axpy<<<GridDim, BlockDim>>>(n, k_beta, k_d, k_r, k_d);
        //  // Copy back snew so the host can evaluate the stopping condition
        //  cudaMemcpy(&snew, k_snew, sizeof(double), cudaMemcpyDeviceToHost);
         //printf("%e\n", sqrt(snew));
         iterations++;
     }
     cudaThreadSynchronize();
     // Copy result vector back from GPU
     double *b_new = (double *)malloc(sizeof(double) * n);
     cudaMemset(k_q,0,n * sizeof(double));
     cudaMemcpy(k_d, k_x, sizeof(double) * (n), cudaMemcpyDeviceToDevice);
     t += utime();
     cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, COMPUTE_TYPE,
        CUSPARSE_MV_ALG_DEFAULT, dBuffer);
     cudaMemcpy(b_new, k_q, sizeof(double) * (n), cudaMemcpyDeviceToHost);
     double sum = 0;
     for (int i = 0; i < n; i++)
     {
         double r = b_new[i] - b_p[i];
         sum = sum + (r * r);
     }
     double sum_ori = 0;
     for (int i = 0; i < n; i++)
     {
         sum_ori = sum_ori + (b_p[i] * b_p[i]);
     }
     double l2_norm = sqrt(sum) / sqrt(sum_ori);
     //printf("L2 Norm is %lf\n", l2_norm);
     printf("time_spmv=%lf\n", time_spmv);
    //  char *s=(char *)malloc(sizeof (char )*150);
    //  sprintf(s,"iter=%d,time_spmv=%.3f,time_cg=%.3f,nnzR=%d,l2_norm=%lf\n",iterations,time_spmv,t*1000,nnz,l2_norm);
    //  FILE *file1=fopen("cg_cusparse.txt","a");
    //  if(file1 == NULL)
    //  {
    //     printf("open error!\n");
    //     return;
    //  }
    //  fwrite(filename,strlen(filename),1,file1);
    //  fwrite(",",strlen(","),1,file1);
    //  fwrite(s,strlen(s),1,file1);
    //  free(s);
    char *s = (char *)malloc(sizeof(char) * 150);
    sprintf(s, "memory_MB=%lf,nnzR=%d\n",cnt_memory,nnz);
    FILE *file1 = fopen("memory_cuSPARSE_nnz.csv", "a");
    if (file1 == NULL)
    {
        printf("open error!\n");
        return;
    }
    fwrite(filename, strlen(filename), 1, file1);
    fwrite(",", strlen(","), 1, file1);
    fwrite(s, strlen(s), 1, file1);
    free(s);
     cudaFree(k_b);
     cudaFree(k_x);
     cudaFree(k_r);
     cudaFree(k_d);
     cudaFree(k_q);
     cudaFree(k_jac);
     cudaFree(k_alpha);
     cudaFree(k_snew);
     cudaFree(k_sold);
     cudaFree(k_beta);
     cudaFree(k_s0);
     cudaFree(scratchpad);
     cudaFree(dBuffer);
     cudaFree(dA_csrOffsets);
     cudaFree(dA_columns);
     cudaFree(dA_values);
     cusparseDestroySpMat(matA);
     cusparseDestroyDnVec(vecX);
     cusparseDestroyDnVec(vecY);
     cusparseDestroy(handle);
     // End timing - call cudaThreadSynchronize so we know all computation is finished before we stop the clock.
     cudaThreadSynchronize();
 
     // Interesting information
 }
 __global__ void d_print_csr(int *row_ptr, int *col_idx, VALUE_TYPE_AX *val, int m)
 {
     printf("device: \n");
     printf("row ptr: ");
     for (size_t i = 0; i <= m; i++)
     {
         printf("%d,", row_ptr[i]);
     }
     printf("\ncol idx: ");
     for (size_t i = 0; i < row_ptr[m]; i++)
     {
         printf("%d,", col_idx[i]);
     }
     printf("\n val:");
     for (size_t i = 0; i < row_ptr[m]; i++)
     {
         printf("%.1f,", val[i]);
     }
     printf("\n");
 }
 
 int main(int argc, char **argv)
 {
     // Host problem definition
     int A_num_rows;
     int A_num_cols;
     int A_nnz;
     int isSymmetric;
     char *filename = argv[2];
     int device_id = atoi(argv[1]);
     mmio_info(&A_num_rows, &A_num_cols, &A_nnz, &isSymmetric, filename);
     ALPHA_TYPE alpha = 1.0f;
     ALPHA_TYPE beta = 0.0f;
     //cudaSetDevice(0);
     int isSymmetricA;
 
     int *hA_csrOffsets=(int *)malloc(sizeof(int)*(A_num_rows+1));
     int *hA_columns=(int *)malloc(sizeof(int)*A_nnz);
     double *hA_values_tmp=(double *)malloc(sizeof(double)*A_nnz);
    //  int memory_old=0;
    //  memory_old+=sizeof(int)*(A_num_rows+1);
    //  memory_old+=sizeof(int)*A_nnz;
    //  memory_old+=sizeof(double)*A_nnz;
    //  printf("memory_old=%lf\n",(double)memory_old/1048576);
     //read_Dmatrix(&A_num_rows, &A_num_cols, &A_nnz, &hA_csrOffsets, &hA_columns, &hA_values_tmp, &isSymmetricA, filename);
     mmio_data(hA_csrOffsets, hA_columns, hA_values_tmp, filename);
     VALUE_TYPE_AX *hA_values = (VALUE_TYPE_AX *)malloc(A_nnz * sizeof(VALUE_TYPE_AX));
     for (size_t i = 0; i < A_nnz; i++)
     {
         hA_values[i] = (VALUE_TYPE_AX)hA_values_tmp[i];
     }
    int n=A_num_rows;
    MAT_VAL_TYPE *X = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (n));
    MAT_VAL_TYPE *Y = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (n));
    MAT_VAL_TYPE *Y_golden = (MAT_VAL_TYPE *)malloc(sizeof(MAT_VAL_TYPE) * (n));
    float *y_golden_float= (float *)malloc(sizeof(float)*n);
    memset(X, 0, sizeof(MAT_VAL_TYPE) * (n));
    memset(Y, 0, sizeof(MAT_VAL_TYPE) * (n));
    memset(Y_golden, 0, sizeof(MAT_VAL_TYPE) * (n));
    for (int i = 0; i < n; i++)
    {
      X[i] = 1;
    }
    for (int i = 0; i < n; i++)
    {
      for (int j = hA_csrOffsets[i]; j < hA_csrOffsets[i + 1]; j++)
          Y_golden[i] += hA_values[j] * X[hA_columns[j]];
      y_golden_float[i]=(float)Y_golden[i];
    }
    gpucg_solve_(filename,hA_csrOffsets,hA_columns,hA_values_tmp,Y_golden,n,X,A_nnz);
    free(X);
    free(Y_golden);
    free(y_golden_float);
    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);
    free(hA_values_tmp);
    return EXIT_SUCCESS;
 }
 