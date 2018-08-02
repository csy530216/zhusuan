import tensorflow as tf
from tensorflow.python.framework import ops

src_module = tf.load_op_library('./sparse_reduce_cols.so')


def src(values, indices, shape):
    return src_module.sparse_reduce_cols(values, indices, shape)

@ops.RegisterGradient("SparseReduceCols")
def _sparse_reduce_cols_grad(op, grad):
    val = op.inputs[0]
    idx = op.inputs[1]
    shp = op.inputs[2]
    