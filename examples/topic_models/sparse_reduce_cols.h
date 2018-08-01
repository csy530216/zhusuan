#ifndef SPARSE_REDUCE_COLS_H_
#define SPARSE_REDUCE_COLS_H_
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

template <typename Device>
struct SparseReduceColsFunctor
{
    void operator()(const Device &d, int numvals, const float *values,
                    const long long *indices, const long long *shape, float *sum_vec);
};

// the codes below should not add to header file!
/*#if GOOGLE_CUDA
//#define EIGEN_USE_GPU
template <>
struct SparseReduceColsFunctor<Eigen::GpuDevice>
{
    void operator()(const Eigen::GpuDevice &d, int numvals, const float *values,
                    const long long *indices, const long long *shape, float *sum_vec);
};
#endif*/

#endif //SPARSE_REDUCE_COLS_H_