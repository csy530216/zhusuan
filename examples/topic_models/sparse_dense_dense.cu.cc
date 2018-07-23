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
#include "eigen3/unsupported/Eigen/CXX11/Tensor"

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
__global__ void SparseDenseDenseKernel(int ncols, int nnz, 
                const float *A, const float *B,
                const long long *indices, float *P) {
  int jstart = ((long long)blockIdx.x) * nnz / gridDim.x;
  int jend = ((long long)(blockIdx.x + 1)) * nnz / gridDim.x;
  int tid = threadIdx.x + blockDim.x * threadIdx.y;
  int stride = blockDim.x * blockDim.y;
  for (int j = jstart; j < jend ; j++) if (tid==0) P[j]=0;
  __syncthreads();
  for (int j = jstart; j < jend ; j++) {
    float result = 0;
    auto r = indices[j*2];
    auto c = indices[j*2+1];
    for (int i = tid; i < ncols; i += stride)
      result += A[r*ncols+i] * B[c*ncols+i];
    for (int i = 1; i < blockDim.x; i *= 2) {
      float tmp = __shfl_down(result, i);
      if (threadIdx.x + i < blockDim.x) result = result + tmp;
    } 
    if (threadIdx.x == 0) {
      atomicAdd(&P[j], result);
      //P[j] = result;
    }
  }
  __syncthreads();
}

// Define the GPU implementation that launches the CUDA kernel.
void SparseDenseDenseKernelLauncher(int ncols, int nnz, 
                const float *A, const float *B,
                const long long *indices, float *P) {
  dim3 blockDims(min(32,ncols), min(32, 1+(ncols-1)/64), 1);
  //dim3 blockDims(32, 1, 1);
  int nblocks = min(16384, max(1, nnz/128));
  SparseDenseDenseKernel<<<nblocks, blockDims>>>(ncols, nnz, A, B, indices, P);
  //cout << "Finished kernel" << endl;
}

#endif

