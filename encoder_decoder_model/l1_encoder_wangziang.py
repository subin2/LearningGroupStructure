# -*- coding: utf-8 -*-
# @Time    : 18-12-06
# @Author  : Wang ZiAng
# @Site    : http://github.com/gongmm
# @File    : l1_encoder.py
# @IDE     : PyCharm Community Edition
"""
packing CNN-l1
"""
from collections import OrderedDict

import tensorflow as tf
import cnn_base_model


class L1Encoder(cnn_base_model.CNNBaseModel):
    """
        packing CNN-l1
    """

    def __init__(self, phase, weight_decay_rate=0.5, use_bn=False, loss_func=None):
        """

        :param phase:
        """
        super(L1Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._use_bn = use_bn
        self._labels = tf.placeholder(tf.float32, shape=[1, 2], name='y') # [batch_size, num]
        self._weight_decay_rate = weight_decay_rate
        self._is_training = self._init_phase()
        self._loss_func = loss_func

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name, stride=1, padding='SAME'):
        """
        packing convolution function and activation function

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param padding:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(input_data=input_tensor, out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=padding, name='conv')

            if self._use_bn:
                bn = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')
                relu = self.relu(input_data=bn, name='relu')
            else:
                relu = self.relu(input_data=conv, name='relu')

        return relu

    def _full_connected_stage(self, input_tensor, out_dims, name, use_bias=False):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            fc = self.fully_connect(input_data=input_tensor, out_dim=out_dims, name='fc',
                                    use_bias=use_bias)

            if self._use_bn:
                bn = self.layer_bn(input_data=fc, is_training=self._is_training, name='bn')
                relu = self.relu(input_data=bn, name='relu')
            else:
                relu = self.relu(input_data=fc, name='relu')

        return relu

    # def build_model(self):

    def encode(self, input_tensor, name, num_iter=3):
        """
        initialize VGG16 network structure

        :param num_iter:
        :param input_tensor:
        :param name:
        :return:
        """
        print('    {:{length}} : {}'.format('x', input_tensor, length=12))
        layer_count = 0

        ret = OrderedDict()
        # conv_1
        with tf.variable_scope(name + str(layer_count + 1)):
            if num_iter == 0:
                # conv stage
                network = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                           out_dims=64, name='conv_'+str(layer_count))
            else:
                for i in range(num_iter):
                    network = input_tensor
                    network = self._conv_stage(input_tensor=network, k_size=3,
                                               out_dims=64, name='conv'+str(layer_count)+'_'+str(i))
                # pool stage
                pool = self.max_pooling(input_data=network, kernel_size=2,
                                         stride=2, name='pool'+str(layer_count))
                ret['pool1'] = dict()
                ret['pool1']['data'] = pool
                ret['pool1']['shape'] = pool.get_shape().as_list()

            print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
            layer_count += 1
        # conv_2
        with tf.variable_scope(name + str(layer_count + 1)):
            if num_iter == 0:
                # conv stage
                network = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                           out_dims=64, name='conv_'+str(layer_count))
            else:
                for i in range(num_iter):
                    network = input_tensor
                    network = self._conv_stage(input_tensor=network, k_size=3,
                                               out_dims=64, name='conv'+str(layer_count)+'_'+str(i))
                # pool stage
                pool = self.max_pooling(input_data=network, kernel_size=2,
                                         stride=2, name='pool'+str(layer_count))
                ret['pool2'] = dict()
                ret['pool2']['data'] = pool
                ret['pool2']['shape'] = pool.get_shape().as_list()

            print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
            layer_count += 1
        # conv_3
        with tf.variable_scope(name + str(layer_count + 1)):
            if num_iter == 0:
                # conv stage
                network = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                           out_dims=64, name='conv_'+str(layer_count))
            else:
                for i in range(num_iter):
                    network = input_tensor
                    network = self._conv_stage(input_tensor=network, k_size=3,
                                               out_dims=64, name='conv'+str(layer_count)+'_'+str(i))
                # pool stage
                pool = self.max_pooling(input_data=network, kernel_size=2,
                                         stride=2, name='pool'+str(layer_count))
                ret['pool3'] = dict()
                ret['pool3']['data'] = pool
                ret['pool3']['shape'] = pool.get_shape().as_list()

            print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
            layer_count += 1
        # conv_4
        with tf.variable_scope(name + str(layer_count + 1)):
            if num_iter == 0:
                # conv stage
                network = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                           out_dims=64, name='conv_'+str(layer_count))
            else:
                for i in range(num_iter):
                    network = input_tensor
                    network = self._conv_stage(input_tensor=network, k_size=3,
                                               out_dims=64, name='conv'+str(layer_count)+'_'+str(i))
                # pool stage
                pool = self.max_pooling(input_data=network, kernel_size=2,
                                         stride=2, name='pool'+str(layer_count))
                ret['pool4'] = dict()
                ret['pool4']['data'] = pool
                ret['pool4']['shape'] = pool.get_shape().as_list()

            print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
            layer_count += 1


        with tf.variable_scope('logit'):
            self._output = network  # .get_layer()
            self._output_layer = network
            # self._output_layer = self._full_connected_stage(input_tensor, self.hps.num_classes)
            # self._predictions = tf.nn.softmax(self._output_layer)
            # fc6 = self._full_connected_stage(input_tensor=pool1, out_dims=4096, name='fc6', use_bias=False)

        with tf.variable_scope('loss'):
            # 构建损失函数
            if self._loss_func is None:
                self._cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(self._output_layer, self._output_layer)))
                # L1 正则化
                self._cost += self._decay()
                # self.err = tf.sqrt(tf.losses.mean_squared_error(self.output_layer, self.y))
            else:
                # self.err =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.y))
                cost = self._loss_func(logits=self._output_layer, labels=self._labels)
                self._cost = tf.reduce_mean(cost)
                # L1 正则化
                self._cost += self._decay()
        return ret

    # 权重衰减，L2正则loss
    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(self._weight_decay_rate, tf.add_n(costs))


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input')
    encoder = L1Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')
    for layer_name, layer_info in ret.items():
        print("layer name: {:s} shape: {}".format(layer_name, layer_info['shape']))
