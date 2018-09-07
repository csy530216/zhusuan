
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
//#include <cuda_runtime_api.h>
#include "sparse_dense_matmul.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

const int tasks_per_warp = 8;
const int shared_per_warp = 128;
const int warps_per_block = 1;
const int threads_per_warp = 32;

//currently I suppose that k <= 128
__global__ void matmul(long long m, long long n, long long k, long long nnz,
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
            /*if (threadIdx.x == 0)
                printf("%d\n", cur_row);*/
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
}

template <typename GPUDevice>
void SparseDenseMatmulFunctor<GPUDevice>::operator()(
    const GPUDevice &d, long long m, long long n, long long k, long long nnz,
    const float *sparse, const long long *indices, const float *dense,
    float *output, bool transpose_sparse)
{
    const auto j = transpose_sparse ? n : m;
    cudaMemset(output, 0, j * k * sizeof(float));

    const auto tasks_per_block = tasks_per_warp * warps_per_block;
    const auto num_blocks = (nnz + tasks_per_block - 1) / tasks_per_block;
    matmul<<<num_blocks, warps_per_block * threads_per_warp>>>(
        m, n, k, nnz, sparse, indices, dense, output);
}

template struct SparseDenseMatmulFunctor<GPUDevice>;

#endif