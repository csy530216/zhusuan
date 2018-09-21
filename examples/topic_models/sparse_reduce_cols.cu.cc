#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reduce_cols.h"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
//#include <iostream>
//#include <stdio.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int sum_len = 256;
const int work_per_thread = 4;

__global__ void SparseReduceColsKernel(int numvals, const float *values,
                                       const long long *indices, float *sum_vec)
{
    __shared__ float sum[sum_len];
    for (auto i = threadIdx.x; i < sum_len; i += blockDim.x)
        sum[i] = 0.0f;
    __syncthreads();
    auto block_start_offset = blockIdx.x * sum_len;
    auto block_end_offset = min(numvals, block_start_offset + sum_len);
    auto block_start_idx = indices[block_start_offset * 2];
    auto block_end_idx = indices[block_end_offset * 2 - 2];
    /*if (block_end_offset == numvals && threadIdx.x == 0)
        printf("%d is the end row index.\n", block_end_idx);*/
    auto thread_start_offset =
        block_start_offset + threadIdx.x * work_per_thread;
    auto id = -1;
    auto val = 0.0f;
    for (auto i = 0; i < work_per_thread; ++i)
    {
        auto offset = thread_start_offset + i;
        if (offset >= block_end_offset)
            break;
        auto id_temp = indices[offset * 2];
        auto val_temp = values[offset];
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
            auto id_offset = id - block_start_idx;
            atomicAdd(sum + id_offset, val);
            id = id_temp;
            val = val_temp;
        }
    }
    if (id > -1)
    {
        auto id_offset = id - block_start_idx;
        atomicAdd(sum + id_offset, val);
    }
    __syncthreads();
    auto bound = block_end_idx - block_start_idx + 1;
    /*if (threadIdx.x == 0)
        printf("%d\n", bound);*/
    for (auto i = threadIdx.x; i < bound; i += blockDim.x)
    {
        /*if (i == 0 || i == bound - 1)
        {
            atomicAdd(sum_vec + block_start_idx + i, sum[i]);
        }
        else
        {
            sum_vec[block_start_idx + i] = sum[i];
        }*/
        atomicAdd(sum_vec + block_start_idx + i, sum[i]);
    }
}

__global__ void zero(const long long num, float *sum)
{
    auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = gridDim.x * blockDim.x;
    for (auto i = threadId; i < num; i += stride)
    {
        sum[i] = 0.0f;
    }
}

template <typename GPUDevice>
void SparseReduceColsFunctor<GPUDevice>::operator()(const GPUDevice &d,
                                                    int numvals, const float *values, const long long *indices, const long long *shape, float *sum_vec)
{
    /*cudaDeviceSynchronize();
    auto error = cudaGetLastError();
    if (error)
    {
        printf("error occurred: %d\n", error);
    }
    else
        printf("OK!\n");*/
    /*auto tpb = 64;
    auto nb = (shape[0] + tpb - 1) / tpb;
    zero<<<nb, tpb>>>(shape[0], sum_vec);*/
    cudaMemset(sum_vec, 0, shape[0] * sizeof(float));
    auto threads_per_block = sum_len / work_per_thread;
    auto numblocks = (numvals + sum_len - 1) / sum_len;
    //std::cout << "cuda kernel begin..." << std::endl;
    /*std::vector<long long> temp;
    for (auto i = 0; i < numvals; ++i)
    {
        temp.push_back(indices[i * 2]);
    }*/
    /*std::cout << "in sparse reduce cols: " << shape[0] << " "
              << shape[1] << std::endl;*/
    //auto out_len = shape[0];
    SparseReduceColsKernel<<<numblocks, threads_per_block>>>(numvals, values,
                                                             indices, sum_vec);
    //std::cout << "cuda kernel src complete" << std::endl;
}

template struct SparseReduceColsFunctor<GPUDevice>;

#endif //GOOGLE_CUDA