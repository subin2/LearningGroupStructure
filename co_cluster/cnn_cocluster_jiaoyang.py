# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : cnn_cocluster_jiaoyang.py
# @IDE     : PyCharm Community Edition
"""
    将em中的模型转化成co-cluster的综合版本
"""
import tensorflow as tf
import numpy as np
from encoder_decoder_model import cnn_base_model
from encoder_decoder_model import rcl_encoder_origin

class CNNCocluster(cnn_base_model.CNNBaseModel):
    """
        对CNN中的权重参数矩阵使用 cocluster 进行优化
    """
    def __init__(self):
        super(CNNCocluster, self).__init__()

    def _input_cocluster(self, input_tensor):
        """
        根据权重参数w的聚类规则调整input的排列顺序，使其对应w1_cocluster做出相同的变化
        :param input_tensor:
        :return:
        """
        input_tensor_cocluster = input_tensor
        return input_tensor_cocluster

    def _co_cluster(self, W):
        """
        对权重参数W进行聚类

        :param W:
        :return:
        """
        w_cocluster = W
        return w_cocluster

    def build_model(self,
                    input_tensor,
                    name='fp_model',
                    data_format='NHWC',
                    out_channel=128):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        # 搭建第一层卷积网络
        with tf.name_scope(name):
            # layer1 = rcl_encoder_origin.RCL(input=input_tensor,
            #                                 weight_size=[3, 3, input_tensor.shape[3], 128],
            #                                 pool='max_pool',
            #                                 activation_func=tf.nn.relu,
            #                                 use_dropout = False,
            #                                 use_batchnorm=False,
            #                                 std=0.05)
            # conv_layers.append(layer1)

            # 获取输入数据的shape
            in_shape = input_tensor.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]
            # 初始化权重参数w1
            w_init = tf.contrib.layers.variance_scaling_initializer()
            w1 = tf.get_variable('W1', shape=[3,3, in_channel, out_channel], initializer=w_init)
            # 对权重参数进行聚类，得到w1_cocluster
            w1_cocluster = self._co_cluster(w1)
            # 根据wi_cocluster的聚类规则调整神经元的顺序，使其对应w1_cocluster做出相同的变化
            input_tensor_cocluster = self._input_cocluster(input_tensor)
            # 搭建神经网络，去除不重要的连接
            conv_1 = self.conv2d(input_data=input_tensor, out_channel=out_channel,
                                                  kernel_size=3, use_bias=False, name='conv_1')
            relu_1 = self.relu(input_data=conv_1, name='relu_1')

