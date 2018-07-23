import tensorflow as tf
from sdd import *


N = 5
K = 3
M = 10

a = tf.random_uniform((N, K)) + 0.5
b = tf.random_uniform((K, M)) + 0.5
mask = tf.random_uniform((N, M)) < 0.1
# Dense
fmask = tf.cast(mask, tf.float32)
y = tf.random_uniform((N, M))
masked_y = y * fmask
# Sparse
indices = tf.where(mask)
values = tf.gather_nd(y, indices)


c = logsdd(a, b, indices)
# Sparse
L = tf.reduce_sum(c * values)
ga, gb = tf.gradients(L, [a, b])
# Dense
Ld = tf.reduce_sum(masked_y * tf.log(tf.matmul(a, b)))
gad, gbd = tf.gradients(Ld, [a, b])

with tf.Session() as sess:
    _a, _b, _mask, _ga, _gb, _gad, _gbd, _L, _Ld, _c = sess.run([a, b, mask, ga, gb, gad, gbd, L, Ld, c])
    print(_a)
    print(_b)
    print(_mask)
    print(_c)
    print(_L, _Ld)
    print(_ga)
    print(_gad)
    print(_gb)
    print(_gbd)
