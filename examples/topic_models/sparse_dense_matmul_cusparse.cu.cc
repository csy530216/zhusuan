#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cusparse.h>
#include <cublas_v2.h>
#include "sparse_dense_matmul_cusparse.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int tasks_per_thread = 8;
const int threads_per_block = 256;
const int shared_len = tasks_per_thread * threads_per_block;

__global__ void classifyIndices(long long nnz, const long long *indices,
                                int *rowIndices, int *colIndices)
{
    __shared__ int shared_row[shared_len];
    __shared__ int shared_col[shared_len];
    auto globalId = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    auto offset = threadIdx.x;
    for (auto i = 0; i < tasks_per_thread; ++i)
    {
        if (globalId >= nnz)
            break;
        auto j = globalId * 2;
        shared_row[offset] = (int)indices[j];
        shared_col[offset] = (int)indices[j + 1];
        offset += blockDim.x;
        globalId += stride;
    }
    __syncthreads();
    globalId = blockIdx.x * blockDim.x + threadIdx.x;
    /*if (globalId == 0)
    {
        for (auto i = 0; i < 10; ++i)
        {
            printf("%d, %d -- ", shared_row[i], shared_col[i]);
        }
        printf("\n");
    }*/
    for (auto i = threadIdx.x; i < offset; i += blockDim.x)
    {
        rowIndices[globalId] = shared_row[i];
        colIndices[globalId] = shared_col[i];
        globalId += stride;
    }
}

template <typename GPUDevice>
void SparseDenseMatmulCusparseFunctor<GPUDevice>::operator()(
    const GPUDevice &d, long long m, long long n, long long k,
    long long nnz, const float *sparse, const long long *indices,
    int *rowIndices, int *csrIndices, int *colIndices, const float *dense,
    float *dense_t, float *output, float *output_t, bool transpose_sparse)
{
    const int blocks = (nnz + shared_len - 1) / shared_len;
    classifyIndices<<<blocks, threads_per_block>>>(nnz, indices, rowIndices,
                                                   colIndices);

    /*int *rows = new int[nnz];
    int *cols = new int[nnz];
    float *vals = new float[nnz];
    cudaMemcpy(rows, rowIndices, nnz * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(cols, colIndices, nnz * sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(vals, sparse, nnz * sizeof(float), cudaMemcpyDefault);
    for (auto i = 0; i < nnz; ++i)
    {
        printf("%d %d %f -- ", rows[i], cols[i], vals[i]);
    }
    printf("\n");
    delete[] rows;
    delete[] cols;
    delete[] vals;

    printf("%d %d\n", n, k);
    float *dvals = new float[n * k];
    cudaMemcpy(dvals, dense, n * k * sizeof(float), cudaMemcpyDefault);
    for (auto i = 0; i < n * k; ++i)
    {
        printf("%f\n", dvals[i]);
    }
    delete[] dvals;*/

    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
    cublasHandle_t b_handle = 0;

    status = cusparseCreate(&handle);
    if (status > 0)
        printf("error occured!\n");
    status = cusparseCreateMatDescr(&descr);
    if (status > 0)
        printf("error occured!\n");
    status = cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    if (status > 0)
        printf("error occured!\n");
    status = cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    if (status > 0)
        printf("error occured!\n");

    status = cusparseXcoo2csr(handle, rowIndices, nnz, m, csrIndices,
                              CUSPARSE_INDEX_BASE_ZERO);
    //printf("coo to csr status %d\n", status);

    float alpha = 1.0f;
    float zero = 0.0f;
    cublasCreate(&b_handle);

    if (transpose_sparse)
    {
        cublasSgeam(b_handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, &alpha, dense,
                    k, &zero, dense, m, dense_t, m);
        status = cusparseScsrmm(handle, CUSPARSE_OPERATION_TRANSPOSE, m, k,
                                n, nnz, &alpha, descr, sparse, csrIndices, colIndices, dense_t, m, &zero, output_t, n);
        cublasSgeam(b_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, &alpha, output_t,
                    n, &zero, output_t, k, output, k);
        //printf("compute complete %d\n", status);
    }
    else
    {
        cublasSgeam(b_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, k, &alpha, dense,
                    k, &zero, dense, n, dense_t, n);
        /*float *dvalst = new float[n * k];
        cudaMemcpy(dvalst, dense_t, n * k * sizeof(float), cudaMemcpyDefault);
        for (auto i = 0; i < n * k; ++i)
        {
            printf("%f\n", dvalst[i]);
        }
        delete[] dvalst;*/

        status = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, k,
                                n, nnz, &alpha, descr, sparse, csrIndices,
                                colIndices, dense_t, n, &zero, output_t, m);
        //printf("non transpose compute complete %d\n", status);
        cublasSgeam(b_handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, &alpha, output_t,
                    m, &zero, output_t, k, output, k);
    }

    status = cusparseDestroyMatDescr(descr);
    status = cusparseDestroy(handle);
    cublasDestroy(b_handle);
}

template struct SparseDenseMatmulCusparseFunctor<GPUDevice>;

#endif