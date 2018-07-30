#ifndef SPARSE_REDUCE_COLS_H_
#define SPARSE_REDUCE_COLS_H_

template <typename Device>
struct SparseReduceColsFunctor
{
    void operator()(const Device &d, int numvals, const float *values,
                    const int64 *indices, const int64 *shape, float *sum_vec);
};

#if GOOGLE_CUDA
template <typename Eigen::GpuDevice>
struct SparseReduceColsFuntor
{
    void operator()(const Eigen::GpuDevice &d, int numvals, const float *values,
                    const int64 *indices, const int64 *shape, float *sum_vec);
};
#endif

#endif SPARSE_REDUCE_COLS_H_