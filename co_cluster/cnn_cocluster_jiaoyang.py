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


class CNNCocluster(cnn_base_model.CNNBaseModel):
    """
        对CNN中的权重参数矩阵使用 cocluster 进行优化
    """
    def __init__(self):
        super(CNNCocluster, self).__init__()

    def _init_w(self, input_tensor, use_cocluster=False, std=0.05, data_format='NHWC', out_channel=128, name='init_w'):
        """

        :param input_tensor:
        :param data_format:
        :param out_channel:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            # 获取输入数据的shape
            in_shape = input_tensor.get_shape().as_list()
            channel_axis = 3 if data_format == 'NHWC' else 1
            in_channel = in_shape[channel_axis]

            # 初始化权重参数w
            w = tf.Variable(tf.random_normal([3, 3, in_channel, out_channel], stddev=std, dtype=tf.float32))

            if use_cocluster:
                # 对权重参数进行聚类，得到w_cocluster，内容为01矩阵，1代表关键元素，0代表不重要元素
                w_cocluster = self._co_cluster(w)

                # 将w_cocluster_T与w逐元素相乘，w中不重要的参数全部转化为0
                w = tf.multiply(w, w_cocluster)

            return w

    def _co_cluster(self, w):
        """
        对权重参数W进行聚类

        :param w: 为tensorflow中的矩阵
        :return w_cocluster: 对w聚类过后的矩阵，为tensorflow中的0 1矩阵，1代表关键元素，0代表不重要元素
        """
        w_cocluster = w

        return w_cocluster

    def build_model(self,
                    input_tensor,
                    name='cnn_cocluster_model'):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            # 搭建第一层卷积网络
            w1 = self._init_w(input_tensor=input_tensor, out_channel=128, name='init_w1')
            # 在计算时(搭建神经网络)，根据wi_cocluster的聚类规则进行计算，去除不重要的连接，减少参数量
            conv_1 = self.conv2d(input_data=input_tensor,
                                 kernel_size=3,
                                 out_channel=128,
                                 w_init=w1,
                                 name='conv2d_cosluter1')
            relu_1 = self.relu(input_data=conv_1, name='relu_1')
            pool1 = self.max_pooling(input_data=relu_1, kernel_size=2, stride=2)

            # 搭建第二层卷积网络
            w2 = self._init_w(input_tensor=pool1, use_cocluster=True, out_channel=128, name='init_w2')
            # 在计算时(搭建神经网络)，根据wi_cocluster的聚类规则进行计算，去除不重要的连接，减少参数量
            conv_2 = self.conv2d(input_data=pool1,
                                 kernel_size=3,
                                 out_channel=128,
                                 w_init=w2,
                                 name='conv2d_cosluter2')
            relu_2 = self.relu(input_data=conv_2, name='relu_2')
            pool2 = self.max_pooling(input_data=relu_2, kernel_size=2, stride=2)

            # 搭建第三层卷积网络
            w3 = self._init_w(input_tensor=pool2, use_cocluster=True, out_channel=128, name='init_w3')
            # 在计算时(搭建神经网络)，根据wi_cocluster的聚类规则进行计算，去除不重要的连接，减少参数量
            conv_3 = self.conv2d(input_data=pool2,
                                 kernel_size=3,
                                 out_channel=128,
                                 w_init=w3,
                                 name='conv2d_cosluter3')
            relu_3 = self.relu(input_data=conv_3, name='relu_3')
            pool3 = self.max_pooling(input_data=relu_3, kernel_size=2, stride=2)

            # 搭建第四层卷积网络
            w4 = self._init_w(input_tensor=pool3, use_cocluster=True, out_channel=128, name='init_w4')
            # 在计算时(搭建神经网络)，根据wi_cocluster的聚类规则进行计算，去除不重要的连接，减少参数量
            conv_4 = self.conv2d(input_data=pool3,
                                 kernel_size=3,
                                 out_channel=128,
                                 w_init=w4,
                                 name='conv2d_cosluter4')
            relu_4 = self.relu(input_data=conv_4, name='relu_4')
            pool4 = self.max_pooling(input_data=relu_4, kernel_size=2, stride=2)

            # 搭建第五层卷积网络
            w5 = self._init_w(input_tensor=pool4, use_cocluster=True, out_channel=128, name='init_w5')
            # 在计算时(搭建神经网络)，根据wi_cocluster的聚类规则进行计算，去除不重要的连接，减少参数量
            conv_5 = self.conv2d(input_data=pool4,
                                 kernel_size=3,
                                 out_channel=128,
                                 w_init=w5,
                                 name='conv2d_cosluter5')
            relu_5 = self.relu(input_data=conv_5, name='relu_5')
            pool5 = self.max_pooling(input_data=relu_5, kernel_size=2, stride=2)

