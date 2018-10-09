
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
//#include <cuda_runtime_api.h>
#include "sparse_dense_matmul.h"
#include "cub/cub.cuh"
#include "util.cuh"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int tasks_per_block = 8;
const int threads_per_block = 64;
//const int warps_per_block = 2;
//const int threads_per_warp = 32;

__global__ void matmul(long long m, long long n, long long k, long long nnz,
                       const float *sparse, const int *rowIndices,
                       const int *colIndices, const float *dense, float *output)
{
    //__shared__ float out_temp[threads_per_block];
    __shared__ int sparse_row[tasks_per_block];
    __shared__ int sparse_col[tasks_per_block];
    __shared__ float sparse_val[tasks_per_block];

    //init out temp
    //out_temp[threadIdx.x] = 0;
    //float value = 0.0f;

    //init sparse items
    auto sparse_start_id = blockIdx.x * tasks_per_block;
    const auto bound = min(tasks_per_block, int(nnz - sparse_start_id));
    if (threadIdx.x < bound)
    {
        auto j = sparse_start_id + threadIdx.x;
        sparse_val[threadIdx.x] = sparse[j];
        sparse_row[threadIdx.x] = rowIndices[j];
        sparse_col[threadIdx.x] = colIndices[j];
        /*if (rowIndices[j] >= m)
            printf("row error found!\n");
        if (colIndices[j] >= n)
            printf("col error found! %d, %d, %d\n", m, n, colIndices[j]);*/
    }
    __syncthreads();

    //k loop
    auto iters = int((k + blockDim.x - 1) / blockDim.x);
    //printf("%d\n", iters);
    for (auto i = 0; i < iters; ++i)
    {
        const int start_col = i * blockDim.x;
        auto end_col = min(blockDim.x, int(k - start_col));
        auto cur_row = -1;
        auto value = 0.0f;
        for (auto j = 0; j < bound; ++j)
        {
            if (cur_row < sparse_row[j])
            {
                if (cur_row >= 0)
                {
                    auto out_start = output + int(k) * cur_row + start_col;
                    /*if ((out_start - output) + threadIdx.x >= k * m)
                        printf("access mem bigger than bound!\n");*/
                    if (threadIdx.x < end_col)
                        atomicAdd(out_start + threadIdx.x, value);
                    value = 0.0f;
                }
                cur_row = sparse_row[j];
            }
            auto dense_start = dense + int(k) * sparse_col[j] + start_col;
            /*if ((dense_start - dense) + threadIdx.x >= n * k)
                printf("access dense matrix bigger than bound! %d\n", 
                    (dense_start - dense) + threadIdx.x);*/
            if (threadIdx.x < end_col)
                value += sparse_val[j] * dense_start[threadIdx.x];
        }
        if (cur_row > -1)
        {
            auto out_start = output + int(k) * cur_row + start_col;
            if (threadIdx.x < end_col)
                atomicAdd(out_start + threadIdx.x, value);
        }
    }
}

/*__global__ void matmul(long long m, long long n, long long k, long long nnz,
                       const float *sparse, const long long *indices,
                       const float *dense, float *output)
{
    __shared__ float c_rows[shared_per_warp * warps_per_block];
    const int warp_id = threadIdx.x / threads_per_warp;
    const int lane_id = threadIdx.x & 0x1f;
    float *shared_buf = c_rows + shared_per_warp * warp_id;
    long long sparse_idx =
        tasks_per_warp * (warps_per_block * blockIdx.x + warp_id);
    auto cur_row = -1;
    for (auto i = 0; i < tasks_per_warp; ++i)
    {
        if (sparse_idx >= nnz)
            break;
        auto sparse_val = sparse[sparse_idx];
        auto sparse_row = indices[sparse_idx * 2];
        auto sparse_col = indices[sparse_idx * 2 + 1];
        if (cur_row < sparse_row)
        {
            if (cur_row >= 0)
            {
                auto out_row = output + k * cur_row;
                for (auto j = lane_id; j < k; j += threads_per_warp)
                    atomicAdd(out_row + j, shared_buf[j]);
            }
            for (auto j = lane_id; j < shared_per_warp; j += threads_per_warp)
            {
                shared_buf[j] = 0.0f;
            }
            cur_row = sparse_row;
            if (threadIdx.x == 0)
                printf("%d\n", cur_row);
        }
        auto dense_row = dense + k * sparse_col;
        for (auto j = lane_id; j < k; j += threads_per_warp)
        {
            shared_buf[j] += sparse_val * dense_row[j];
        }
        ++sparse_idx;
    }
    if (cur_row >= 0)
    {
        auto out_row = output + k * cur_row;
        for (auto j = lane_id; j < k; j += threads_per_warp)
            atomicAdd(out_row + j, shared_buf[j]);
    }
}*/

template <typename GPUDevice>
void SparseDenseMatmulFunctor<GPUDevice>::operator()(
    const GPUDevice &d, long long m, long long n, long long k, long long nnz,
    const float *sparse, const long long *indices, const float *dense,
    float *output, bool transpose_sparse, int *tempbuf)
{
    //const auto j = transpose_sparse ? n : m;
    //cudaMemset(output, 0, j * k * sizeof(float));
    //printf("%d, %d, %d\n", m, n, k);
    int *rowIndices = tempbuf;
    int *colIndices = tempbuf + nnz;
    const int threads_pb = 256;
    const int tasks_pt = 8;
    const int tasks_pb = tasks_pt * threads_pb;
    const int blocks = (nnz + tasks_pb - 1) / tasks_pb;
    const auto num_blocks = (nnz + tasks_per_block - 1) / tasks_per_block;
    if (!transpose_sparse)
    {
        extractIndices<<<blocks, threads_pb>>>(nnz, indices, rowIndices,                                               colIndices);
        matmul<<<num_blocks, threads_per_block>>>(
            m, n, k, nnz, sparse, rowIndices, colIndices, dense, output);
        //cudaDeviceSynchronize();
        //printf("matmul without transpose complete, %d\n", cudaGetLastError());
    }
    else
    {
        extractIndices<<<blocks, threads_pb>>>(nnz, indices, colIndices,
                                               rowIndices);
        int *tempIdx = colIndices + nnz;
        initIndices<<<blocks, threads_pb>>>(nnz, tempIdx);
        int *alt_row = tempIdx + nnz;
        int *alt_idx = alt_row + nnz;
        cub::DoubleBuffer<int> keys(rowIndices, alt_row);
        cub::DoubleBuffer<int> idx(tempIdx, alt_idx);
        void *temp_store = NULL;
        size_t temp_store_byte = 0;
        cub::DeviceRadixSort::SortPairs(temp_store, temp_store_byte, keys,
                                        idx, nnz);
        if (temp_store_byte <= nnz * sizeof(int) * 2)
        {
            //printf("mem allocated by tf used!\n");
            temp_store = (void *)(alt_idx + nnz);
        }
        else
        {
            //printf("mem allocated by cuda used!\n");
            cudaMalloc(&temp_store, temp_store_byte);
        }
        cub::DeviceRadixSort::SortPairs(temp_store, temp_store_byte, keys,
                                        idx, nnz);
        //printf("sort complelte\n");
        int *col_cpy = alt_idx + nnz;
        float *val_cpy = reinterpret_cast<float *>(col_cpy + nnz);
        resort<<<blocks, threads_pb>>>(nnz, idx.Current(), colIndices, sparse,
                                       col_cpy, val_cpy);
        //cudaDeviceSynchronize();
        //printf("sort idx complete, %d\n", cudaGetLastError());
        matmul<<<num_blocks, threads_per_block>>>(
            n, m, k, nnz, val_cpy, keys.Current(), col_cpy, dense,
            output);
        //cudaDeviceSynchronize();
        //printf("matmul complelte, %d\n", cudaGetLastError());
    }
}

template struct SparseDenseMatmulFunctor<GPUDevice>;

#endif