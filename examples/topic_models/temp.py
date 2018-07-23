import tensorflow as tf

def sparse_tile(indices, values, dense_shape, n_rep):
    # HACK
    # Convert to dense
    dense_t = tf.sparse_to_dense(indices, dense_shape, values)
    # Tile
    dense_t = tf.tile(dense_t, [n_rep, 1])
    # Convert back
    tiled_indices = tf.where(dense_t > 0)
    tiled_values = tf.tile(values, [n_rep])
    return tiled_indices, tiled_values

indices = tf.constant([[0, 1], [1, 3], [2, 2]], dtype=tf.int64)
values  = tf.constant([1, 2, 3], dtype=tf.float32)
dense_shape = tf.constant([4, 4], dtype=tf.int64)

t_indices, t_values = sparse_tile(indices, values, dense_shape, 3)

with tf.Session() as sess:
    print(sess.run(t_indices))
    print(sess.run(t_values))
