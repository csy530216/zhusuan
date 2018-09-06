import tensorflow as tf

from sdmc import *

mat_4 = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2], [3, 3]],
                        values=[1.5, 2.5, 3.5, 4.5], dense_shape=[4, 4])

sess = tf.Session()

# print(sess.run(mat_4))

mat_42 = tf.constant([[1.5, 0.5], [2.5, 3.5], [4.5, 5.5], [7.5, 6.5]])

mul1 = tf.sparse_tensor_dense_matmul(mat_4, mat_42)

print(sess.run(mul1))

mul2 = sdmc(mat_4, mat_42)

print(sess.run(mul2))
