#ifndef SPARSE_DENSE_MATMUL_H
#define SPARSE_DENSE_MATMUL_H

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

template <typename Device>
struct SparseDenseMatmulFunctor
{
    void operator()(const Device &d, long long m, long long n, long long k,
                    long long nnz, const float *sparse,
                    const long long *indices, const float *dense, float *out,
                    bool transpose_sparse, int *temp_buf);
};

#endif // SPARSE_DENSE_MATMUL_H
