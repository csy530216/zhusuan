#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "sparse_reduce_sum_cuda.h"
#include "util.cuh"
#include "cub/cub.cuh"
//#include "tensorflow/core/util/cuda_kernel_helper.h"
//#include <iostream>
//#include <stdio.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int sum_len = 256;
const int work_per_thread = 4;

__global__ void SparseReduceSumCudaKernel(int numvals, const float *values,
                                          int *indices, float *sum_vec)
{
    __shared__ float sum[sum_len];
    for (auto i = threadIdx.x; i < sum_len; i += blockDim.x)
        sum[i] = 0.0f;
    __syncthreads();
    auto block_start_offset = blockIdx.x * sum_len;
    auto block_end_offset = min(numvals, block_start_offset + sum_len);
    auto block_start_idx = indices[block_start_offset];
    auto block_end_idx = indices[block_end_offset - 1];
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
        auto id_temp = indices[offset];
        auto val_temp = values[offset];
        if (id < id_temp)
        {
            if (id > -1)
            {
                auto id_offset = id - block_start_idx;
                atomicAdd(sum + id_offset, val);
            }
            id = id_temp;
            val = val_temp;
        }
        else
        {
            val += val_temp;
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
        if (i == 0 || i == bound - 1)
        {
            atomicAdd(sum_vec + block_start_idx + i, sum[i]);
        }
        else
        {
            sum_vec[block_start_idx + i] = sum[i];
        }
        //atomicAdd(sum_vec + block_start_idx + i, sum[i]);
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
void SparseReduceSumCudaFunctor<GPUDevice>::operator()(const GPUDevice &d, 
                                                       int numvals, 
                                                       const float *values, const long long *indices,
                                                       const long long *shape, 
                                                       float *sum_vec, 
                                                       int *temp_buf, 
                                                       int axis)
{
    cudaMemset(sum_vec, 0, numvals * sizeof(float));
    int *rowIndices = temp_buf;
    int *colIndices = temp_buf + numvals;
    const int threads_per_block = 256;
    const int tasks_per_thread = 8;
    const int tasks_per_block = threads_per_block * tasks_per_thread;
    const int blocks = (numvals + tasks_per_block - 1) / tasks_per_block;
    extractIndices<<<blocks, threads_per_block>>>(numvals, indices,
                                                  rowIndices, colIndices, tasks_per_thread);
    auto numblocks = (numvals + sum_len - 1) / sum_len;
    if (axis == 1 || axis == -1)
    {
        /*auto tpb = 64;
        auto nb = (shape[0] + tpb - 1) / tpb;
        zero<<<nb, tpb>>>(shape[0], sum_vec);*/
        //auto threads_per_block = sum_len / work_per_thread;
        //std::cout << "cuda kernel begin..." << std::endl;
        /*std::vector<long long> temp;
        for (auto i = 0; i < numvals; ++i)
        {
            temp.push_back(indices[i * 2]);
        }*/
        /*std::cout << "in sparse reduce cols: " << shape[0] << " "
                << shape[1] << std::endl;*/
        //auto out_len = shape[0];
        SparseReduceSumCudaKernel<<<numblocks, sum_len / work_per_thread>>>(
            numvals, values, rowIndices, sum_vec);
        //std::cout << "cuda kernel src complete" << std::endl;
    }else
    {
        int *alt_col = colIndices + numvals;
        float *vals = reinterpret_cast<float *>(alt_col + numvals);
        cudaMemcpyAsync(vals, values, numvals * sizeof(float),
            cudaMemcpyDefault);
        float *alt_vals = vals + numvals;
        void *temp_store = NULL;
        size_t temp_store_byte = 0;

        cub::DoubleBuffer<int> keys(colIndices, alt_col);
        cub::DoubleBuffer<float> fvals(vals, alt_vals);
        cub::DeviceRadixSort::SortPairs(temp_store, temp_store_byte, keys,
                                        fvals, numvals);
        if (temp_store_byte <= numvals * sizeof(float))
            temp_store = (void *)(alt_vals + numvals);
        else
            cudaMalloc(&temp_store, temp_store_byte);
        cub::DeviceRadixSort::SortPairs(temp_store, temp_store_byte, keys,
                                        fvals, numvals);
        SparseReduceSumCudaKernel<<<numblocks, sum_len / work_per_thread>>>(
            numvals, fvals.Current(), keys.Current(), sum_vec);
    }
}

template struct SparseReduceSumCudaFunctor<GPUDevice>;

#endif //GOOGLE_CUDA