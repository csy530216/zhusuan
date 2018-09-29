import numpy as np
import tensorflow as tf
from srsc import *

N = 100
K = 12419

a = np.random.random((N, K)).astype(np.float32)
mask = a < 0.04
indices = tf.where(mask)
values = tf.gather_nd(a, indices)
sparse = tf.SparseTensor(indices=indices, values=values, dense_shape=[N, K])

sess = tf.Session()
print(sess.run(srsc(sparse.values, sparse.indices, sparse.dense_shape, 1)))
#print(sess.run(tf.reduce_sum(tf.scatter_nd(sparse.indices, sparse.values,
#                                           sparse.dense_shape), -1)))
print(sess.run(tf.sparse_reduce_sum(sparse, 1)))
