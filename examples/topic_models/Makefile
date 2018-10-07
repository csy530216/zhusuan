TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_FLAGS=-L$(CUDA_ROOT)/lib64

all: sparse_dense_dense.so sparse_reduce_sum_cuda.so sparse_dense_matmul_cusparse.so sparse_dense_matmul.so

%.cu.o: %.cu.cc
	nvcc -std=c++11 -c -o $@ $^ -I/usr/local/cuda/include  -I/workspace/tsinghua/cub-1.8.0 $(TF_CFLAGS)  -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG

%.so: %.cc %.cu.o
	g++ -std=c++11 -shared -o $@ $^ $(TF_CFLAGS) -fPIC $(CUDA_FLAGS) -lcudart -lcusparse -lcublas $(TF_LFLAGS) -D GOOGLE_CUDA=1

clean:
	#-rm *.o
	-rm *.so
