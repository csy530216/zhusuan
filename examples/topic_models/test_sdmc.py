import tensorflow as tf

from sdmc import *

from sdm import *

mat_4 = tf.SparseTensor(indices=[[0, 1], [1, 2], [2, 3], [3, 0]],
                        values=[1.5, 2.5, 3.5, 4.5], dense_shape=[4, 4])

sess = tf.Session()

# print(sess.run(mat_4))

mat_42 = tf.constant([[1.5, 0.5], [2.5, 3.5], [4.5, 5.5], [7.5, 6.5]])

mul1 = tf.sparse_tensor_dense_matmul(mat_4, mat_42)

print(sess.run(mul1))

mul2 = sdmc(mat_4, mat_42)

print(sess.run(mul2))

mul3 = tf.sparse_tensor_dense_matmul(mat_4, mat_42, adjoint_a=True)

print(sess.run(mul3))

mul4 = sdmc(mat_4, mat_42, adjoint_a=True)

print(sess.run(mul4))

mul5 = sdm(mat_4, mat_42)

print(sess.run(mul5))
