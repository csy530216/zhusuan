TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_FLAGS=-L$(CUDA_ROOT)/lib64

all: sparse_dense_dense.so sparse_reduce_cols.so sparse_dense_matmul_cusparse.so

%.cu.o: %.cu.cc
	nvcc -std=c++11 -c -o $@ $^ $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

%.so: %.cc %.cu.o
	g++ -std=c++11 -shared -o $@ $^ $(TF_CFLAGS) -fPIC $(CUDA_FLAGS) -lcudart $(TF_LFLAGS) -D GOOGLE_CUDA=1

clean:
	#-rm *.o
	-rm *.so
