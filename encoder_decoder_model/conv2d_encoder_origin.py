# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : rcl_encoder_origin.py
# @IDE     : PyCharm Community Edition
"""
    直接将models.py文件中的conv2d单独移出来
"""
import tensorflow as tf


class conv2d(object):
    def __init__(self, input, weight_size, strides=[1, 1, 1, 1], padding='SAME', pool=None, pool_size=4,
                 activation_func=None,
                 use_dropout=True, keep_prob=1.0, use_batchnorm=True, std=0.01, offset=1e-10, scale=1, epsilon=1e-10,
                 name='conv2d_default'):
        with tf.variable_scope(name):
            self.weight = tf.Variable(tf.random_normal(weight_size, stddev=std, dtype=tf.float32))
            self.bias = tf.Variable(tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32))
            network = tf.nn.bias_add(tf.nn.conv2d(input=input, filter=self.weight, strides=strides, padding=padding),
                                     self.bias, name=name)
            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])  # ,1,2])
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset=offset, scale=scale,
                                                    variance_epsilon=epsilon, name=name)
            if activation_func != None:
                network = activation_func(network, name=name)
            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)
            if pool == 'p':
                network = tf.nn.max_pool(value=network,
                                         ksize=[1, 1, pool_size, 1],
                                         strides=[1, 1, pool_size, 1],
                                         padding='SAME')
            self.result = network

    def get_layer(self):
        return self.result

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias
