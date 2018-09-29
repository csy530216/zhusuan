#include "sparse_reduce_sum_cuda.h"
//#include "tensorflow/core/framework/shape_inference.h"
//#include "tensorflow/core/framework/tensor_shape.h"
//#include "tensorflow/core/kernels/fill_functor.h"

using namespace tensorflow; // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

REGISTER_OP("SparseReduceSumCuda")
    .Input("values: float32")
    .Input("indices: int64")
    .Input("shape: int64")
    .Input("axis: int32")
    .Output("arr: float32");
/*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->Vector(c->Dim(c->input(2), 0)));
        return Status::OK();
    });*/

/*void SparseReduceCols(int numval, const float *values, const int64 *indices,
                      const int64 *shape, float *sum_vec);*/

template <typename Device>
class SparseReduceSumCudaOp : public OpKernel
{
  public:
    explicit SparseReduceSumCudaOp(OpKernelConstruction *c) : OpKernel(c) {}

    void Compute(OpKernelContext *context) override
    {
        //std::cout << "begin call src" << std::endl;
        const Tensor &vals = context->input(0);
        const Tensor &inds = context->input(1);
        const Tensor &shape_input = context->input(2);
        const Tensor &axis = context->input(3);
        const int *axis_ptr = axis.flat<int>().data();
        const int64 num_values = vals.dim_size(0);
        /*std::cout << "vals num: " << num_values << " "
                  << vals.flat<float>().size() << " "
                  << vals.NumElements() << std::endl;
        std::cout << shape_input.NumElements() << std::endl;
        std::cout << inds.NumElements() << std::endl;*/

        //std::cout << "the shape: " << std::endl;
        auto vec = shape_input.flat<int64>();
        //auto vec = shape_input.vec<int64>();
        //printf("the shape: %d", vec(0));
        //std::cout << vec(0) << std::endl;
        //std::cout << "flat complete. the shape: " << std::endl;
        //std::cout << vec.data() << std::endl;
        //std::cout << vec(0) << std::endl;
        //printf("shape: %d, %d, %d\n", vec(0), vec(1), num_values);
        TensorShape shape;
        if (*axis_ptr == 1 || *axis_ptr == -1)
            OP_REQUIRES_OK(context,
                           TensorShapeUtils::MakeShape(vec.data(), 1, &shape));
        else
            OP_REQUIRES_OK(context,
                           TensorShapeUtils::MakeShape(vec.data() + 1, 1,
                                                       &shape));
        //std::cout << "shape creation complete." << std::endl;

        Tensor *output = NULL;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, shape, &output));
        //functor::SetZeroFunctor<Device, float> fill;
        //fill(context->eigen_device<Device>(), output->flat<float>());

        auto values = vals.flat<float>();
        auto indices = inds.flat<int64>();
        auto out = output->flat<float>();
        /*std::cout << out.data() << " " << values.data() << " "
                  << vec.data() << std::endl;*/
        int *temp_mem = NULL;
        //if (*axis_ptr == 0)
        //{
        Tensor temp_tensor;
        OP_REQUIRES_OK(context,
                        context->allocate_temp(DataType::DT_INT32,
                                                TensorShape({num_values * 6}),
                                                &temp_tensor));
        temp_mem = temp_tensor.flat<int>().data();
        //}

        //std::cout << "srsc initial complete" << std::endl;

        SparseReduceSumCudaFunctor<Device>()(context->eigen_device<Device>(),
                                          num_values, values.data(),
                                          indices.data(), vec.data(), 
                                          out.data(), temp_mem, *axis_ptr);
    }
};

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU()                                               \
    /* It's recommended to add the code below, but not essential. */ \
    extern template struct SparseReduceSumCudaFunctor<GPUDevice>;       \
    REGISTER_KERNEL_BUILDER(Name("SparseReduceSumCuda")                 \
                                .Device(DEVICE_GPU)                  \
                                .HostMemory("shape")                \
                                .HostMemory("axis"),                \
                            SparseReduceSumCudaOp<GPUDevice>);
REGISTER_GPU();
#endif // GOOGLE_CUDA