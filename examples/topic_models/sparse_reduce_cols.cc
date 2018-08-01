#include "sparse_reduce_cols.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow; // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SparseReduceCols")
    .Input("values: float32")
    .Input("indices: int64")
    .Input("shape: int64")
    .Output("arr: float32");
/*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->Vector(c->Dim(c->input(2), 0)));
        return Status::OK();
    });*/

/*void SparseReduceCols(int numval, const float *values, const int64 *indices,
                      const int64 *shape, float *sum_vec);*/

template <typename Device>
class SparseReduceColsOp : public OpKernel
{
  public:
    explicit SparseReduceColsOp(OpKernelConstruction *context)
        : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor &vals = context->input(0);
        const Tensor &inds = context->input(1);
        const Tensor &shape_input = context->input(2);
        const int64 num_values = vals.dim_size(0);

        auto vec = shape_input.flat<int64>();
        TensorShape shape;
        OP_REQUIRES_OK(context,
                       TensorShapeUtils::MakeShape(vec.data(), vec.size(), &shape));

        Tensor *output = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, shape, &output));

        auto values = vals.flat<float>();
        auto indices = inds.flat<int64>();
        auto out = output->flat<float>();

        SparseReduceColsFunctor<Device>()(context->eigen_device<Device>(),
                                          num_values, values.data(),
                                          indices.data(), vec.data(), out.data());
    }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU()                                                   \
    /* It's recommended to add the code below, but not essential. */     \
    extern template struct SparseReduceColsFunctor<GPUDevice>;           \
    REGISTER_KERNEL_BUILDER(Name("SparseReduceCols").Device(DEVICE_GPU), \
                            SparseReduceColsOp<GPUDevice>);
REGISTER_GPU();
#endif // GOOGLE_CUDA