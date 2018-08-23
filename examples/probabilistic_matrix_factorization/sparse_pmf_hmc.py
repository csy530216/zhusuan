#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

from six.moves import range
import tensorflow as tf
import numpy as np
import zhusuan as zs
from sdd import *
from src import *
from scipy.sparse import csr_matrix

from examples import conf
from examples.utils import dataset

def pmf(observed, N, M, D, K, alpha_u, alpha_v, alpha_pred):
    with zs.BayesianNet(observed=observed) as model:
        mu_u = tf.zeros(shape=[N, D])
        u = zs.Normal('u', mu_u, std=alpha_u,
                      n_samples=K, group_ndims=1)
        u = tf.reshape(u, [-1, D])
        u = tf.nn.softmax(u)
        mu_v = tf.zeros(shape=[M, D])
        v = zs.Normal('v', mu_v, std=alpha_v,
                      n_samples=K, group_ndims=1)
        v = tf.reshape(v, [-1, D])
        v = tf.nn.softmax(v)
        r_pred = sdd(u, tf.transpose(v), observed['r_indices'])
        r = zs.Normal('r_values', r_pred, std=alpha_pred)
        # Should be: no `u = tf.nn.softmax(u)`, no `v = tf.nn.softmax(v)`,
        # r should be predicted as tf.sigmoid(r_pred)
    return model, r_pred


def get_indices_and_values(data):
    data = data.tocoo()
    indices = np.transpose(np.array([data.row, data.col])).astype(np.int64)
    values = data.data
    p = np.lexsort((indices[:, 1], indices[:, 0]))
    return indices[p, :], values[p].astype(np.float32)


def main():
    np.random.seed(1234)
    tf.set_random_seed(1237)
    M, N, train_data, valid_data, test_data = dataset.load_movielens1m(
        os.path.join(conf.data_dir, 'ml-1m.zip'))
    train = csr_matrix((train_data[:, -1], (train_data[:, 0], train_data[:, 1])))
    valid = csr_matrix((valid_data[:, -1], (valid_data[:, 0], valid_data[:, 1])))
    test = csr_matrix((test_data[:, -1], (test_data[:, 0], test_data[:, 1])))

    R_train_indices, R_train_values = get_indices_and_values(train)
    R_valid_indices, R_valid_values = get_indices_and_values(valid)
    R_test_indices, R_test_values = get_indices_and_values(test)

    r_indices = tf.placeholder(tf.int64, shape=[None, 2], name='r_indices')
    r_values = tf.placeholder(tf.float32, shape=[None], name='r_values')

    # set configurations and hyper parameters
    D = 30
    K = 1
    epochs = 500
    valid_freq = 10
    test_freq = 10
    alpha_u = 1.0
    alpha_v = 1.0
    alpha_pred = 0.2 / 4.0

    U = tf.get_variable('U', shape=[K, N, D],
                        initializer=tf.random_normal_initializer(0, 0.1),
                        trainable=False)

    V = tf.get_variable('V', shape=[K, M, D],
                        initializer=tf.random_normal_initializer(0, 0.1),
                        trainable=False)

    # Define models for prediction
    normalized_rating = (r_values - 1.0) / 4.0
    _, pred_rating = pmf({'u': U, 'v': V, 'r_indices': r_indices},
                         N, M, D, K, alpha_u, alpha_v, alpha_pred)
    pred_rating = tf.reduce_mean(pred_rating, axis=0)
    error = pred_rating - normalized_rating
    rmse = tf.sqrt(tf.reduce_mean(error * error)) * 4

    def log_joint(observed):
        model, _ = pmf(observed, N, M, D, K, alpha_u, alpha_v, alpha_pred)
        log_pu, log_pv, log_pr = model.local_log_prob(['u', 'v', 'r_values'])
        r_indices = observed['r_indices']
        dense_shape = [K*N, K*M]
        log_pr_dense = tf.scatter_nd(r_indices, log_pr, dense_shape)
        log_pr_u = tf.reduce_sum(log_pr_dense, -1)
        log_pr_u = tf.reshape(log_pr_u, [K, N])
        log_pr_v = tf.reduce_sum(log_pr_dense, 0)
        log_pr_v = tf.reshape(log_pr_v, [K, M])

        return log_pu + log_pr_u, log_pv + log_pr_v

    def e_obj_u(observed):
        log_pu, _ = log_joint(observed)
        return log_pu

    def e_obj_v(observed):
        _, log_pv = log_joint(observed)
        return log_pv

    hmc_u = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)
    hmc_v = zs.HMC(step_size=1e-3, n_leapfrogs=10, adapt_step_size=None,
                   target_acceptance_rate=0.9)

    def sparse_tile(indices, dense_shape, n_rep):
        tiled_indices = indices
        i = tf.constant(1, dtype=tf.int64)

        def c(idx, i): return i < tf.cast(n_rep, tf.int64)

        def b(idx, i):
            nrow = [dense_shape[0] * i, dense_shape[1] * i]
            tempidx = indices + nrow
            return (tf.concat([idx, tempidx], 0), i + 1)
        tiled_indices, _ = tf.while_loop(c, b, [tiled_indices, i])
        return tiled_indices
    
    tiled_r_indices = sparse_tile(r_indices, [N, M], K)
    tiled_r_values = tf.tile(r_values, [K])
    sample_u_op, sample_u_info = hmc_u.sample(e_obj_u,
        observed={'v': V, 'r_indices': tiled_r_indices, 'r_values': tiled_r_values},
        latent={'u': U})
    sample_v_op, sample_v_info = hmc_v.sample(e_obj_v,
        observed={'u': U, 'r_indices': tiled_r_indices, 'r_values': tiled_r_values},
        latent={'v': V})

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            epoch_time = -time.time()
            feed_dict = {r_indices: R_train_indices,
                         r_values: R_train_values}
            _, acc_u = sess.run([sample_u_op, sample_u_info.acceptance_rate],
                                feed_dict=feed_dict)
            _, acc_v = sess.run([sample_v_op, sample_v_info.acceptance_rate],
                                feed_dict=feed_dict)
            epoch_time += time.time()
            time_train = -time.time()
            train_rmse = sess.run(rmse, feed_dict=feed_dict)
            time_train += time.time()

            print('Epoch {}({:.1f}s): rmse ({:.1f}s) = {}'
                  .format(epoch, epoch_time, time_train, train_rmse))

            if epoch % valid_freq == 0:
                valid_rmse = sess.run(rmse, feed_dict={R_valid_indices, R_valid_values})
                print('>>> VALID')
                print('>> Valid rmse = {}'.format(valid_rmse))

            if epoch % test_freq == 0:
                valid_rmse = sess.run(rmse, feed_dict={R_test_indices, R_test_values})
                print('>>> TEST')
                print('>> Test rmse = {}'.format(test_rmse))


if __name__ == "__main__":
    main()
