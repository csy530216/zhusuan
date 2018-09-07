import tensorflow as tf

sdm_module = tf.load_op_library('./sparse_dense_matmul.so')


def sdm(sparse, dense, adjoint_a=False):
    return sdm_module.sparse_dense_matmul(
        sparse.values, sparse.indices, sparse.dense_shape, dense, adjoint_a)
