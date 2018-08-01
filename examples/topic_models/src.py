import tensorflow as tf

src_module = tf.load_op_library('./sparse_reduce_cols.so')


def src(values, indices, shape):
    return src_module.sparse_reduce_cols(values, indices, shape)
