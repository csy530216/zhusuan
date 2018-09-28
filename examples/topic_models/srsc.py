import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops


src_module = tf.load_op_library('./sparse_reduce_sum_cuda.so')


def srsc(values, indices, shape, axis=1):
    return src_module.sparse_reduce_sum_cuda(values, indices, shape, axis)


@ops.RegisterGradient("SparseReduceSumCuda")
def _sparse_reduce_cols_grad(op, grad):
    values = op.inputs[0]
    indices = op.inputs[1]
    print(values.get_shape(), indices.get_shape(), grad)
#  shape = op.inputs[2]
#  output_shape_kept_dims = math_ops.reduced_shape(shape, -1)
#  grad_reshaped = array_ops.reshape(grad, output_shape_kept_dims)
#  scale = shape // math_ops.to_int64(output_shape_kept_dims)
#  grad_values = array_ops.gather_nd(grad_reshaped, indices // scale)

#    reduced_indices = indices[:, :-1]
#    grad_values = tf.gather_nd(grad, reduced_indices)
# Or:
    reduced_indices = indices[:, 0]
#    grad_values = array_ops.gather(grad, reduced_indices)
    grad_values = tf.gather(grad, reduced_indices)

    return [grad_values, None, None]
