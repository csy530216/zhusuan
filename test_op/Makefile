TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CUDA_FLAGS=-L$(CUDA_ROOT)/lib64

all: zero_out.so cuda_op_kernel.so sparse_dense_dense.so

zero_out.so: zero_out.cc
	g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

%.cu.o: %.cu.cc
	nvcc -std=c++11 -c -o $@ $^ $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

%.so: %.cc %.cu.o
	g++ -std=c++11 -shared -o $@ $^ $(TF_CFLAGS) -fPIC $(CUDA_FLAGS) -lcudart $(TF_LFLAGS) -D GOOGLE_CUDA=1

clean:
	-rm *.o
	-rm *.so
