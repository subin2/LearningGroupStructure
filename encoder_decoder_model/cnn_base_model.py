# -*- coding: utf-8 -*-
# @Time    : 18-12-05
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : cnn_base_model.py
# @IDE     : PyCharm Community Edition

"""
the base convolution neural network mainly implements some useful cnn functions
"""

import tensorflow as tf
import numpy as np


class CNNBaseModel(object):
    """
    base model for other specific cnn models, such as vgg, fcn, denseNet
    """

    def __init__(self):
        pass

    @staticmethod
    def conv2d(input_data, out_channel, kernel_size,
              padding='SAME',  stride=1, w_init=None, b_init=None,
              split=1, use_bias=True, data_format='NHWC', name=None):
        """
        Packing the tensorflow conv2d function

        :param out_channel:     number of output channels
        :param kernel_size:     list or int, if it's int, the kernel shape will be [kernel_size, kernel_size]
        :param padding:         'VALID' or 'SAME'
        :param stride:          list or int, if it's int, the stride shape will be [stride, stride]
        :param w_init:          initializer for convolution weights
        :param b_init:          initializer for bias
        :param split:           split channels as used in Alexnet mainly group for GPU memory save.
        :param use_bias:        whether to use bias.
        :param data_format:     'NHWC' or 'NCHW', default set to 'NHWC' according tensorflow
        :param name:            operation name

        :return:                tf.Tensor named 'output'
        """
        with tf.variable_scope(name):
            in_shape = input_data.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]

            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
            assert in_channel % split == 0, "param split is not satisfied for in_channel"
            assert out_channel % split == 0, "param split is not satisfied for out_channel"

            padding = padding.upper()

            if isinstance(kernel_size, list):
                filter_shape = [kernel_size[0], kernel_size[1]] + [in_channel/split, out_channel]
            else:
                filter_shape = [kernel_size, kernel_size] + [in_channel/split, out_channel]

            if isinstance(stride, list):
                strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                    else [1, 1, stride[0], stride[1]]
            else:
                strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                    else [1, 1, stride, stride]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_channel], initializer=b_init)

            if split == 1:
                conv = tf.nn.conv2d(input_data, w, strides, padding, data_format=data_format)
            else:
                inputs = tf.split(input_data, split, channel_axis)
                kernels = tf.split(w, split, 3)
                outputs = [tf.nn.conv2d(i, k, strides, padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            ret = tf.identity(tf.nn.bias_add(conv, b, data_format=data_format)
                              if use_bias else conv, name=name)

        return ret

    @staticmethod
    def relu(input_data, name=None):
        """

        :param input_data:
        :param name:
        :return:
        """
        return tf.nn.relu(features=input_data, name=name)

    @staticmethod
    def sigmoid(input_data, name=None):
        """

        :param input_data:
        :param name:
        :return:
        """
        return tf.nn.sigmoid(x=input_data, name=name)

    @staticmethod
    def max_pooling(input_data, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """

        :param input_data:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :param name:
        :return:
        """
        padding = padding.upper()

        if stride is None:
            stride = kernel_size

        if isinstance(kernel_size, list):
            kernel = [1, kernel_size[0], kernel_size[1], 1] if data_format == 'NHWC' else \
                [1, 1, kernel_size[0], kernel_size[1]]
        else:
            kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
                else [1, 1, kernel_size, kernel_size]

        if isinstance(stride, list):
            strides = [1, stride[0], stride[1], 1] if data_format == 'NHWC' \
                else [1, 1, stride[0], stride[1]]
        else:
            strides = [1, stride, stride, 1] if data_format == 'NHWC' \
                else [1, 1, stride, stride]

        return tf.nn.max_pool(value=input_data, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def avg_pooling(input_data, kernel_size, stride=None, padding='VALID',
                   data_format='NHWC', name=None):
        """


        :param name:
        :param input_data:
        :param kernel_size:
        :param stride:
        :param padding:
        :param data_format:
        :return:
        """
        if stride is None:
            stride = kernel_size

        kernel = [1, kernel_size, kernel_size, 1] if data_format == 'NHWC' \
            else [1, 1, kernel_size, kernel_size]

        strides = [1, stride, stride, 1] if data_format == 'NHWC' else [1, 1, stride, stride]

        return tf.nn.avg_pool(value=input_data, ksize=kernel, strides=strides, padding=padding,
                              data_format=data_format, name=name)

    @staticmethod
    def global_avg_pooling(input_data, data_format='NHWC', name=None):
        """
        :param name:
        :param input_data:
        :param data_format:
        :return:
        """
        assert input_data.shape.ndims == 4
        assert data_format in ['NHWC', 'NCHW']

        axis = [1, 2] if data_format == 'NHWC' else [2, 3]

        return tf.reduce_mean(input_tensor=input_data, axis=axis, name=name)

    @staticmethod
    def layer_norm(input_data, epsilon=1e-5, use_bias=True, use_scale=True,
                  data_format='NHWC', name=None):
        """

        :param input_data:
        :param epsilon:         epsilon to avoid divide-by-zero.
        :param use_bias:        whether to use the extra affine transformation or not.
        :param use_scale:       whether to use the extra affine transformation or not.
        :param data_format:
        :param name:
        :return:
        """
        shape = input_data.get_shape().as_list()
        ndims = len(shape)
        assert ndims in [2, 4]

        mean, var = tf.nn.moments(input_data, list(range(1, len(shape))), keep_dims=True)

        if data_format == 'NCHW':
            channnel = shape[1]
            new_shape = [1, channnel, 1, 1]
        else:
            channnel = shape[-1]
            new_shape = [1, 1, 1, channnel]
        if ndims == 2:
            new_shape = [1, channnel]

        if use_bias:
            beta = tf.get_variable('beta', [channnel], initializer=tf.constant_initializer())
            beta = tf.reshape(beta, new_shape)
        else:
            beta = tf.zeros([1] * ndims, name='beta')
        if use_scale:
            gamma = tf.get_variable('gamma', [channnel], initializer=tf.constant_initializer(1.0))
            gamma = tf.reshape(gamma, new_shape)
        else:
            gamma = tf.ones([1] * ndims, name='gamma')

        return tf.nn.batch_normalization(input_data, mean, var, beta, gamma, epsilon, name=name)

    @staticmethod
    def dropout(input_data, keep_prob, noise_shape=None, name=None):
        """

        :param input_data:
        :param keep_prob:A scalar Tensor with the same type as x. The probability that each element is kept.
        :param noise_shape:A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags
        :param name:
        :return: A Tensor of the same shape of x
        """
        return tf.nn.dropout(input_data, keep_prob=keep_prob, noise_shape=noise_shape, name=name)

    @staticmethod
    def fully_connect(input_data, out_dim, w_init=None, b_init=None,
                     use_bias=True, name=None):
        """
        Fully-Connected layer, takes a N>1D tensor and returns a 2D tensor.
        It is an equivalent of `tf.layers.dense` except for naming conventions.
        Dense layer implements the operation: outputs = activation(inputs * kernel + bias)

        :param input_data:
        :param out_dim:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        shape = input_data.get_shape().as_list()[1:]
        if None not in shape:
            input_data = tf.reshape(input_data, [-1, int(np.prod(shape))])
        else:
            input_data = tf.reshape(input_data, tf.stack([tf.shape(input_data)[0], -1]))

        if w_init is None:
            w_init = tf.contrib.layers.variance_scaling_initializer()
        if b_init is None:
            b_init = tf.constant_initializer()

        ret = tf.layers.dense(inputs=input_data, activation=lambda x: tf.identity(x, name='output'),# activation???
                              use_bias=use_bias, name=name,
                              kernel_initializer=w_init, bias_initializer=b_init,
                              trainable=True, units=out_dim)
        return ret

    @staticmethod
    def layer_bn(input_data, is_training, name):
        """

        :param input_data:
        :param is_training:
        :param name:
        :return:
        """
        return tf.layers.batch_normalization(inputs=input_data, training=is_training, name=name)

    @staticmethod
    def deconv2d(input_data, out_channel, kernel_size, padding='SAME',
                 stride=1, w_init=None, b_init=None,
                 use_bias=True, activation=None, data_format='channels_last',
                 trainable=True, name=None):
        """
        Packing the tensorflow conv2d function.

        :param input_data:
        :param out_channel:
        :param kernel_size:
        :param padding:
        :param stride:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param activation:
        :param data_format:
        :param trainable:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_data.get_shape().as_list()
            channel_axis = 3 if data_format == 'channels_last' else 1
            in_channel = in_shape[channel_axis]
            assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            ret = tf.layers.conv2d_transpose(inputs=input_data, filters=out_channel,
                                             kernel_size=kernel_size,
                                             strides=stride, padding=padding,
                                             data_format=data_format,
                                             activation=activation, use_bias=use_bias,
                                             kernel_initializer=w_init,
                                             bias_initializer=b_init, trainable=trainable,
                                             name=name)
        return ret

    @staticmethod
    def dilation_conv(input_data, k_size, out_dims, rate, padding='SAME',
                      w_init=None, b_init=None, use_bias=False, name=None):
        """

        :param input_data:
        :param k_size:
        :param out_dims:
        :param rate:
        :param padding:
        :param w_init:
        :param b_init:
        :param use_bias:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            in_shape = input_data.get_shape().as_list()
            in_channel = in_shape[3]
            assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"

            padding = padding.upper()

            if isinstance(k_size, list):
                filter_shape = [k_size[0], k_size[1]] + [in_channel, out_dims]
            else:
                filter_shape = [k_size, k_size] + [in_channel, out_dims]

            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            if b_init is None:
                b_init = tf.constant_initializer()

            w = tf.get_variable('W', filter_shape, initializer=w_init)
            b = None

            if use_bias:
                b = tf.get_variable('b', [out_dims], initializer=b_init)

            conv = tf.nn.atrous_conv2d(value=input_data, filters=w, rate=rate,
                                       padding=padding, name='dilation_conv')

            if use_bias:
                ret = tf.add(conv, b)
            else:
                ret = conv

        return ret
