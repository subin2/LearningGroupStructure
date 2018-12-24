# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : rcl_encoder_origin.py
# @IDE     : PyCharm Community Edition
"""
    直接将models.py文件中的RCL单独移出来
"""
import tensorflow as tf
from encoder_decoder_model import conv2d_encoder_origin

class RCL(object):
    def __init__(self, input, weight_size, weight=None, biases=None, strides=[1, 1, 1, 1],
                 padding='SAME', pool='max_pool', pool_size=[2, 2],
                 activation_func=None, use_dropout=True, keep_prob=1.0, use_batchnorm=True,
                 std=0.01, offset=1e-10, scale=1, epsilon=1e-10, name='RCL_default'):
        """
			conv2d
			与cnn-EM中的RCL相比，删除了本次训练中用不到的参数和结构，如num_iter, 及参数pool='c'时的情况
		"""
        self.pool = pool
        with tf.variable_scope(name):
            self.weight = tf.Variable(
                tf.random_normal(weight_size, stddev=std, dtype=tf.float32)) if weight is None else weight
            self.biases = tf.Variable(
                tf.random_normal([weight_size[-1]], stddev=std, dtype=tf.float32)) if biases is None else biases

            network = input

            network = tf.nn.bias_add(
                tf.nn.conv2d(input=network, filter=self.weight, strides=strides, padding=padding),
                self.biases
                )

            if use_batchnorm:
                batch_mean, batch_var = tf.nn.moments(network, [0])  # [0,1,2]
                network = tf.nn.batch_normalization(network, batch_mean, batch_var, offset, scale, epsilon,
                                                    name=name)
            if activation_func != None:
                network = activation_func(network, name=name)

            if use_dropout:
                network = tf.nn.dropout(network, keep_prob=keep_prob, name=name)

            if pool == 'max_pool':
                network = tf.nn.max_pool(value=network,
                                         ksize=[1, pool_size[0], pool_size[1], 1],
                                         strides=[1, pool_size[0], pool_size[1], 1],
                                         padding='SAME')
            self.result = network

    def get_layer(self):
        return self.result

    def get_conv_layer(self):
        if self.pool != 'c':
            raise ValueError('No conv layer is used for pooling.')
        return self.pool

    def get_weight(self):
        return self.weight

    def get_biases(self):
        return self.biases
