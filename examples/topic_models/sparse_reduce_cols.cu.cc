#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reduce_cols.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int sum_len = 256;
const int work_per_thread = 4;

__global__ void SparseReduceColsKernel(int numvals, const float *values,
                                       const float *indices, const int64 *shape,float *sum_vec)
{
    __shared__ float sum[sum_len];
    auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
    auto block_start_thread = threadId & 31;
    auto start_idx = threadId * work_per_thread;
    auto end_idx = min(start_idx + work_per_thread, numvals * 2 - 1);
    auto start_share = indices[block_start_thread * work_per_thread * 2];
    auto id = -1;
    auto val = 0.0f;
    for (auto i = start_idx; i < end_idx; ++i)
    {
        auto id_temp = indices[i * 2];
        auto val_temp = values[i];
        if (id == -1)
        {
            id = id_temp;
            val = val_temp;
        }
        else if (id_temp == id)
        {
            val += val_temp;
        }
        else
        {
            auto share_id = id - start_share;
            atomicAdd(sum + share_id, val);
            id = id_temp;
            val = val_temp;
        }
    }
    audo share_id = id - start_share;
    atomicAdd(&sum + share_id, val);
}

void SparseReduceFunctor<GPUDevice>::operator()(GPUDevice &d, int numvals,
                                                const float *values, const int64 *indices, const int64 *shape, float *sum_vec)
{
}

#endif //GOOGLE_CUDA