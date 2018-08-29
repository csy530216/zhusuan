#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cusparse.h>
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
    float *output)
{
    const int blocks = (nnz + shared_len - 1) / shared_len;
    classifyIndices<<<blocks, threads_per_block>>>(nnz, indices, rowIndices,
                                                   colIndices);

    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    status = cusparseCreate(&handle);
    status = cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    status = cusparseXcoo2csr(handle, rowIndices, nnz, m, csrIndices,
                              CUSPARSE_INDEX_BASE_ZERO);

    float alpha = 1.0f;
    float zero = 0.0f;
    status = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, k,
                            n, nnz, &alpha, descr, sparse, csrIndices, colIndices, dense, n, &zero, output, m);
}

template struct SparseDenseMatmulCusparseFunctor<GPUDevice>;

#endif