import tensorflow as tf
from src import *

N = 100
K = 12419

a = tf.random_uniform((N, K))
mask = tf.random_uniform((N, K)) < 0.04
indices = tf.where(mask)
values = tf.gather_nd(a, indices)
sparse = tf.SparseTensor(indices=indices, values=values, dense_shape=[N, K])

sess = tf.Session()
print(sess.run(src(sparse.values, sparse.indices, sparse.dense_shape)))
print(sess.run(tf.reduce_sum(tf.scatter_nd(sparse.indices, sparse.values,
                                           sparse.dense_shape), -1)))
print(sess.run(tf.sparse_reduce_sum(sparse, -1)))
