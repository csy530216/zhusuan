#ifndef SPARSE_DENSE_DENSE_H_
#define SPARSE_DENSE_DENSE_H_

template <typename Device>
struct SparseDenseDenseFunctor {
  void operator()(const Device& d, int ncols, int nnz, 
                  const float *A, const float *B,
                  const int *Cir, const int *cic, float *P);
};

#endif
