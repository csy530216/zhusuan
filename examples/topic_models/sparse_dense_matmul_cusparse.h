#ifndef SPARSE_DENSE_MATMUL_CUSPARSE_H
#define SPARSE_DENSE_MATMUL_CUSPARSE_H
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

template <typename Device>
struct SparseDenseMatmulCusparseFunctor
{
    void operator()(const Device &d, long long m, long long n, long long k,
                    long long nnz, const float *sparse,
                    const long long *indices, int *rowIndices, int *csrIndices,
                    int *colIndices, const float *dense, float *out);
};

#endif // !SPARSE_DENSE_MATMUL_CUSPARSE_H
