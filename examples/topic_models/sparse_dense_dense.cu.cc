/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
//#include "eigen3/unsupported/Eigen/CXX11/Tensor"

//__global__ void AddOneKernel(const int* in, const int N, int* out) {
//  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
//       i += blockDim.x * gridDim.x) {
//    out[i] = in[i] + 1;
//  }
//}
//
//void AddOneKernelLauncher(const int* in, const int N, int* out) {
//  AddOneKernel<<<32, 256>>>(in, N, out);
//}

// Define the CUDA kernel.
/*__global__ void SparseDenseDenseKernel(int ncols, int nnz,
                                       const float *A, const float *B,
                                       const long long *indices, float *P)
{
  int jstart = ((long long)blockIdx.x) * nnz / gridDim.x;
  int jend = ((long long)(blockIdx.x + 1)) * nnz / gridDim.x;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int stride = blockDim.x * blockDim.y;
  for (int j = jstart; j < jend; j++)
    if (tid == 0)
      P[j] = 0;
  __syncthreads();
  for (int j = jstart; j < jend; j++)
  {
    float result = 0;
    auto r = indices[j * 2];
    auto c = indices[j * 2 + 1];
    for (int i = tid; i < ncols; i += stride)
      result += A[r * ncols + i] * B[c * ncols + i];
    for (int i = 1; i < blockDim.x; i *= 2)
    {
      float tmp = __shfl_down(result, i);
      if (threadIdx.x + i < blockDim.x)
        result = result + tmp;
    }
    if (threadIdx.x == 0)
    {
      atomicAdd(&P[j], result);
      //P[j] = result;
    }
  }
  __syncthreads();
}*/

const int rows = 8;
const int items = 32;
const unsigned int full_mask = 0xffffffff;
const int warp_per_block = 8;
const int warp_size = 32;

__global__ void SparseDenseDenseKernel(int ncols, int nnz, const float *A,
                                       const float *B, const long long *indices,
                                       float *P)
{
  // may not copy same contents to shared memory, so do not init a_rows each
  // loop
  extern __shared__ float a_rows[];
  __shared__ float prdt[items];
  __shared__ int computed;
  __shared__ int start_row_idx;
  __shared__ int end_row_idx;
  __shared__ int num_rows;
  //auto threadId = blockIdx.x * blockDim.x + threadIdx.x;
  // use register while not shared memory
  const int to_be_computed = min(items, nnz - items * blockIdx.x);
  if (threadIdx.x == 0)
  {
    computed = 0;
    start_row_idx = indices[items * blockIdx.x * 2];
    end_row_idx = indices[(items * blockIdx.x + to_be_computed) * 2];
    num_rows = min(rows, end_row_idx - start_row_idx);
  }
  if (threadIdx.x < items)
    prdt[threadIdx.x] = -1.0f;
  do
  {
    // compute gap between indices
    if (computed && threadIdx.x == 0)
    {
      auto temp_idx = indices[(items * blockIdx.x + computed) * 2];
      if (temp_idx > start_row_idx)
      {
        start_row_idx = temp_idx;
        num_rows = min(rows, end_row_idx - start_row_idx);
      }
      else
      {
        num_rows = 0;
      }
    }
    // copy elements in A to shared memory
    if (num_rows)
    {
      auto start = start_row_idx * ncols;
      auto end = (min(start_row_idx + rows, end_row_idx) + 1) * ncols;
      if (threadIdx.x == 0)
      {
        num_rows = min(rows, end_row_idx - start_row_idx);
      }
      for (auto i = start + threadIdx.x; i < end; i += blockDim.x)
      {
        a_rows[i - start] = A[i];
      }
    }
    auto warpId = threadIdx.x / warp_size;
    auto laneId = threadIdx.x & 0x1f;
    for (auto m = computed + warpId; m < to_be_computed; m += warp_per_block)
    {
      auto offset = (items * blockIdx.x + m) * 2;
      auto j = indices[offset];
      auto k = indices[offset + 1];
      if (j >= start_row_idx + num_rows)
        break;
      const float *ma = a_rows + (j - start_row_idx) * ncols;
      const float *mb = B + k * ncols;
      float value = 0.0f;
      for (auto i = laneId; i < ncols; i += warp_size)
      {
        value += ma[i] * mb[i];
      }
      for (auto i = 16; i > 0; i /= 2)
      {
        value += __shfl_down_sync(full_mask, value, i);
      }
      if (laneId == 0)
      {
        prdt[m] = value;
      }
    }
    __syncthreads();
    /*auto read_items_num = 0;
    if (block_start_thread == threadId)
    {
      auto start_row_idx = indices[(items * blockIdx.x + computed) * 2];
      auto end_row_idx = indices[min(items * (blockIdx.x + 1), nnz) * 2];
      read_items_num = min(rows, end_row_idx - start_row_idx) * ncols;
    }*/
    if (threadIdx.x < items)
    {
      auto mask = __ballot_sync(full_mask, prdt[threadIdx.x] < 0);
      if (threadIdx.x == 0)
      {
        computed = __popc(mask);
      }
    }
  } while (computed < to_be_computed);
  if (threadIdx.x < computed)
  {
    P[items * blockIdx.x + threadIdx.x] = prdt[threadIdx.x];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
void SparseDenseDenseKernelLauncher(int ncols, int nnz,
                                    const float *A, const float *B,
                                    const long long *indices, float *P)
{
  //dim3 blockDims(min(32, ncols), min(32, 1 + (ncols - 1) / 64), 1);
  //dim3 blockDims(32, 1, 1);
  //int nblocks = min(16384, max(1, nnz / 128));
  auto threads_per_block = warp_per_block * warp_size;
  auto num_blocks = (nnz + items - 1) / items;
  SparseDenseDenseKernel<<<
      num_blocks, threads_per_block, ncols * rows * sizeof(float)>>>(
      ncols, nnz, A, B, indices, P);
  //cout << "Finished kernel" << endl;
}

#endif
