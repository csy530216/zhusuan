#include "sparse_dense_matmul.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SparseDenseMatmul")
    .Input("sparse: float32")
    .Input("indices: int64")
    .Input("shape: int64")
    .Input("dense: float32")
    .Input("transpose_sparse: bool")
    .Output("matrix: float32");

template <typename Device>
class SparseDenseMatmulOp : public OpKernel
{
  public:
    explicit SparseDenseMatmulOp(OpKernelConstruction *c) : OpKernel(c)
    {
    }

    void Compute(OpKernelContext *context) override
    {
        const Tensor &sparse = context->input(0);
        const Tensor &indices = context->input(1);
        const Tensor &shape = context->input(2);
        const Tensor &dense = context->input(3);
        const Tensor &transpose_sparse = context->input(4);

        const bool *transpose = transpose_sparse.flat<bool>().data();

        const int64 nnz = sparse.dim_size(0);
        auto sparse_vec = sparse.flat<float>();
        auto indices_vec = indices.flat<int64>();
        auto shape_vec = shape.flat<int64>();
        auto dense_vec = dense.flat<float>();

        const int64 m = shape_vec(0);
        const int64 n = shape_vec(1);
        const int64 k = dense.dim_size(1);

        Tensor *C = NULL;
        auto j = *transpose ? n : m;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0,
                                                TensorShape({j, k}), &C));
        auto C_vec = C->flat<float>();
        Tensor temp;
        OP_REQUIRES_OK(context,
                       context->allocate_temp(DataType::DT_INT32,
                                              TensorShape({nnz * 7}),
                                              &temp));
        SparseDenseMatmulFunctor<Device>()(
            context->eigen_device<Device>(), m, n, k, nnz, sparse_vec.data(),
            indices_vec.data(), dense_vec.data(), C_vec.data(), *transpose,
            temp.flat<int>().data());
    }
};

#ifdef GOOGLE_CUDA
#define REGISTER_GPU()                                            \
    extern template struct SparseDenseMatmulFunctor<GPUDevice>;   \
    REGISTER_KERNEL_BUILDER(Name("SparseDenseMatmul")             \
                                 .Device(DEVICE_GPU)              \
                                 .HostMemory("shape")             \
                                 .HostMemory("transpose_sparse"), \
                            SparseDenseMatmulOp<GPUDevice>);
REGISTER_GPU();
#endif