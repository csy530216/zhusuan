from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


sdd_module = tf.load_op_library('./sparse_dense_dense.so')

# indices: int64
def sdd(a, b, indices):
    print(indices)
    return sdd_module.sparse_dense_dense(a, tf.transpose(b), indices)


@ops.RegisterGradient("SparseDenseDense")
def _sparse_dense_dense_grad(op, grad):
  a    = op.inputs[0]
  b    = op.inputs[1]
  indices = op.inputs[2]
  print(a.get_shape(), b.get_shape(), grad)
  result_shape = tf.cast(tf.stack([tf.shape(a)[0], tf.shape(b)[0]]), dtype=dtypes.int64)
  grad = tf.SparseTensor(indices=indices, values=grad, dense_shape=result_shape)

  grad_T      = sparse_ops.sparse_transpose(grad)
  grad_a      = sparse_ops.sparse_tensor_dense_matmul(grad, b)
  grad_b      = sparse_ops.sparse_tensor_dense_matmul(grad_T, a)
  return [grad_a, grad_b, None] 


def logsdd(a, b, indices):
    c = sdd(a, b, indices)
    #c = tf.Print(c, [tf.reduce_min(c), tf.reduce_min(a), tf.reduce_min(b)], 'c')
    return tf.log(c)
