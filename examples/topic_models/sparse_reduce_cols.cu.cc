#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reduce_cols.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

__global__ void SparseReduceColsKernel()
{
}

void SparseReduceFunctor<GPUDevice>::operator()(GPUDevice &d, int numvals,
                                                const float *values, const int64 *indices, const int64 *shape, float *sum_vec)
{
}

#endif //GOOGLE_CUDA