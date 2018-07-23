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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SparseDenseDense")
    .Input("a: float32")
    .Input("b: float32")
    .Input("indices: int64")
    .Output("p: float32")
    .Doc(R"doc(
Pj = <a[indices[j][0], :], b[indices[j][1], :]>
)doc");

void SparseDenseDenseKernelLauncher(int ncols, int nnz, 
                  const float *A, const float *B,
                  const int64 *indices, float *P);

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
class SparseDenseDenseOp: public OpKernel {
 public:
  explicit SparseDenseDenseOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& a    = context->input(0);
    const Tensor& b    = context->input(1);
    const Tensor& indices = context->input(2);
    const int64 nnz = indices.dim_size(0);

    // Create an output tensor
    Tensor* P = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(
                               0, TensorShape({nnz}), &P));

    const uint64 K = a.dim_size(1);

    auto am    = a.flat<float>();
    auto bm    = b.flat<float>();
    auto Pm    = P->flat<float>();
    auto indices_m = indices.flat<int64>();

    SparseDenseDenseKernelLauncher(
        K, nnz,
        am.data(), bm.data(), 
        indices_m.data(), Pm.data());
  }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("SparseDenseDense").Device(DEVICE_GPU), SparseDenseDenseOp);
#endif  // GOOGLE_CUDA
