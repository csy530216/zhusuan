#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include <cusparse.h>
#include "sparse_dense_matmul_cusparse.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename GPUDevice>
void SparseDenseMatmulCusparseFunctor<GPUDevice> operator()(
    const GPUDevice &d, long long m, long long n, long long k,
    long long nnz, const float *sparse, const long long *indices,
    const float *dense)
{
    cusparseStatus_t status;
    cusparseHandle_t handle = 0;

    status = cusparseCreate(&handle);
}

#endif