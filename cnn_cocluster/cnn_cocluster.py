# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : cnn_cocluster.py
# @IDE     : PyCharm Community Edition
"""
    将em中的模型转化成co-cluster的综合版本
"""
import tensorflow as tf
from encoder_decoder_model import cnn_base_model
from co_cluster import co_cluster_class_wuqiao
from tools import show_result


class CNNCocluster(cnn_base_model.CNNBaseModel):
    """
        对CNN中的权重参数矩阵使用 cocluster 进行优化
    """
    def __init__(self, phase):
        super(CNNCocluster, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def init_w(self, input_tensor, use_cocluster=False, std=0.05,
                data_format='NHWC', out_channel=128, name='init_w'):
        """
        初始化w权重参数，可以选择是否通过cocluster进行初始化

        :param input_tensor:
        :param use_cocluster:
        :param std:
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
                # 先将w初始化，否则无法转为numpy数组
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                w_numpy = sess.run(w)    # 将tensor转为numpy中的数组

                # 对权重参数进行聚类，得到w_cocluster，内容为01矩阵，1代表关键元素，0代表不重要元素
                w_cocluster = self._co_cluster(w_numpy)

                # 将w_cocluster_T与w逐元素相乘，w中不重要的参数全部转化为0, 并将矩阵转回为tensor
                w_numpy = w_numpy * w_cocluster
                w = tf.convert_to_tensor(w_numpy, dtype=tf.float32)

            return w

    def _co_cluster(self, w):
        """
        对权重参数W进行聚类

        :param w: 为tensorflow中的矩阵
        :return w_cocluster: 对w聚类过后的矩阵，为tensorflow中的0 1矩阵，1代表关键元素，0代表不重要元素
        """
        co_cluster = co_cluster_class_wuqiao.BaseCoCluster(w=w)
        w_cocluster = co_cluster.co_cluster()
        return w_cocluster

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, padding='SAME', w=None, use_bn=False,
                    std=0.05, use_relu=True):
        """
        卷积、标准化和激活函数综合层，形成一个完整的卷积操作

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param padding:
        :param w:
        :return:
        """
        with tf.variable_scope(name):
            b_init = tf.Variable(tf.random_normal([out_dims], stddev=std, dtype=tf.float32))
            conv = self.conv2d(input_data=input_tensor,
                               kernel_size=k_size,
                               out_channel=out_dims,
                               w_init=w,
                               b_init=b_init,
                               stride=stride,
                               padding=padding,
                               name='conv2d')
            if use_bn:
                conv = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')
            if use_relu:
                conv = self.relu(input_data=conv, name='relu')

        return conv

    def encode(self,
                    input_tensor,
                    name='cnn_cocluster_encode'):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.name_scope(name):
            # 第一次卷积与池化操作，此处不对w进行聚类，只改变通道数，改为128
            conv1 = self._conv_stage(input_tensor=input_tensor, k_size=3, out_dims=128, name="conv1")
            max_pool1 = self.max_pooling(input_data=conv1, kernel_size=2, stride=2, name='max_pool1', padding='SAME')

            # 第二次卷积与池化操作，此时对 w 进行聚类
            w2 = self.init_w(input_tensor=max_pool1, use_cocluster=True, name='init_w2_cocluster')
            conv2 = self._conv_stage(input_tensor=max_pool1, k_size=3, w=w2, out_dims=128, name='conv2_cocluster')
            max_pool2 = self.max_pooling(input_data=conv2, kernel_size=2, stride=2, name='max_pool2', padding='SAME')

            # 第三次卷积与池化操作，此时对 w 进行聚类
            w3 = self.init_w(input_tensor=max_pool2, use_cocluster=True, name='init_w3')
            conv3 = self._conv_stage(input_tensor=max_pool2, k_size=3, w=w3, out_dims=128, name='conv3_cocluster')
            max_pool3 = self.max_pooling(input_data=conv3, kernel_size=2, stride=2, name='max_pool3', padding='SAME')

            # 第四次卷积与池化操作，此时对 w 进行聚类
            w4 = self.init_w(input_tensor=max_pool3, use_cocluster=True, name='init_w4')
            conv4 = self._conv_stage(input_tensor=max_pool3, k_size=3, w=w4, out_dims=128, name='conv4_cocluster')
            max_pool4 = self.max_pooling(input_data=conv4, kernel_size=2, stride=2, name='max_pool4', padding='SAME')

            # Reshape 操作，输出为 [batch_size, 512]
            reshape = tf.reshape(max_pool4, shape=[-1, 512])

            # 全连接操作，输出为 [batch_size, 128]
            # todo cnn_em中原作者在此对w进行了聚类，测试时注意这里可能还需要对w进行聚类
            # todo cnn_em中原作者在此对使用的bias为 tf.random_normal(stddev=0.05)，cnn_base_model中random-normal 没指定std
            fc = self.fully_connect(input_data=reshape, out_dim=128)

            # 分类操作，输出为 [batch_size, 10]
            net = self.fully_connect(input_data=fc, out_dim=10)

        return net

    def optimize(self, input_tensor, labels, l_rate=0.0001,
                 err_func=tf.nn.softmax_cross_entropy_with_logits):
        if err_func is None:
            err = tf.sqrt(tf.reduce_mean(tf.squared_difference(input_tensor, labels)))
        else:
            err_temp = err_func(logits=input_tensor, labels=labels)
            err = tf.reduce_mean(err_temp)
        optimizer = tf.train.GradientDescentOptimizer(l_rate)
        optimizer.minimize(err, global_step=tf.Variable(0, trainable=False))


if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[64, 32, 32, 3], name='input')
    cnn_cocluster = CNNCocluster(phase=tf.constant('train', dtype=tf.string))

    # 对 w_cocluster的结果进行测试 success
    # w1 = cnn_cocluster.init_w(input_tensor=a, use_cocluster=False, out_channel=128)
    # w2 = cnn_cocluster.init_w(input_tensor=a, use_cocluster=True, out_channel=128)
    # show_result.show_w(w=w2, name='test')

    # 对网络结构进行测试 success
    # net = cnn_cocluster.encode(input_tensor=a)
    # print(net.get_shape().as_list())  # [64, 10]

