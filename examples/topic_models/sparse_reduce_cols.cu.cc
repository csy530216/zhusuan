#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reduce_cols.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int sum_len = 256;
const int work_per_thread = 4;

__global__ void SparseReduceColsKernel(int numvals, const float *values,
                                       const long long *indices, const long long *shape, float *sum_vec)
{
    //printf("compute start\n");
    /*__shared__ float sum[sum_len];
    auto block_start_thread = blockIdx.x * blockDim.x;
    auto threadId = block_start_thread + threadIdx.x;
    auto bound = numvals - 1;
    auto start_idx = min(threadId * work_per_thread, bound);
    auto end_idx = min(start_idx + work_per_thread, bound);
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
    auto share_id = id - start_share;
    atomicAdd(sum + share_id, val);
    //auto global_start = indices[block_start_thread * 2];
    auto global_start = start_share;
    auto global_end_idx = min((blockIdx.x + 1) * sum_len, numvals - 1);
    auto global_end = indices[global_end_idx * 2];
    for (auto i = threadIdx.x; i < global_end - global_start + 1;
         i += blockDim.x)
    {
        auto index = global_start + i;
        if (i == 0 || index == global_end)
            atomicAdd(sum_vec + index, sum[i]);
        else
            sum_vec[index] = sum[i];
    }*/
    //printf("compute start\n");
    //printf("%d, %d\n", gridDim.x, blockDim.x);
    __shared__ float sum[sum_len];
    for (auto i = threadIdx.x; i < sum_len; i += blockDim.x)
        sum[i] = 0;
    auto block_start_thread = blockIdx.x * blockDim.x;
    //printf("%d\n", block_start_thread);
    auto threadId = block_start_thread + threadIdx.x;
    auto bound = numvals;
    auto start_idx = min(threadId * work_per_thread, bound);
    auto end_idx = min(start_idx + work_per_thread, bound);
    //printf("%d, %d\n", start_idx, end_idx);
    auto start_share = indices[block_start_thread * work_per_thread * 2];
    auto id = -1;
    auto val = 0.0f;
    for (auto i = start_idx; i < end_idx; ++i)
    {
        auto id_temp = indices[i * 2];
        auto val_temp = values[i];
        //printf("%f\n", val_temp);
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
            __syncthreads();
            id = id_temp;
            val = val_temp;
        }
    }

    auto share_id = id - start_share;
    atomicAdd(sum + share_id, val);
    __syncthreads();
    //auto global_start = indices[block_start_thread * 2];
    auto global_start = start_share;
    auto global_end_idx = min((blockIdx.x + 1) * sum_len, numvals - 1);
    auto global_end = indices[global_end_idx * 2];
    for (auto i = threadIdx.x; i < global_end - global_start + 1;
         i += blockDim.x)
    {
        //printf("sum: %f, %d, %d, %d\n", sum[i], blockIdx.x, gridDim.x, blockDim.x);
        auto index = global_start + i;
        if (i == 0 || index == global_end)
            atomicAdd(sum_vec + index, sum[i]);
        else
            sum_vec[index] = sum[i];
    }
    if (threadId == 0)
        printf("sparse reduce cols complete.\n");
}

template <typename GPUDevice>
void SparseReduceColsFunctor<GPUDevice>::operator()(const GPUDevice &d,
                                                    int numvals, const float *values, const long long *indices, const long long *shape, float *sum_vec)
{
    auto threads_per_block = sum_len / work_per_thread;
    auto numblocks = (numvals + sum_len - 1) / sum_len;
    std::cout << "cuda kernel begin..." << std::endl;
    /*std::vector<long long> temp;
    for (auto i = 0; i < numvals; ++i)
    {
        temp.push_back(indices[i * 2]);
    }*/
    SparseReduceColsKernel<<<numblocks, threads_per_block>>>(numvals, values,
                                                             indices, shape, sum_vec);
    std::cout << "cuda kernel src complete" << std::endl;
}

template struct SparseReduceColsFunctor<GPUDevice>;

#endif //GOOGLE_CUDA