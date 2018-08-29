import tensorflow as tf

sdmc_module = tf.load_op_library('./sparse_dense_matmul_cusparse.so')

def sdmc(sparse, dense):
    return sdmc_module.sparse_dense_matmul_cusparse(sparse.values,
        sparse.indices, sparse.dense_shape, dense)