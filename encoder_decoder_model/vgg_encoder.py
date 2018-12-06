# -*- coding: utf-8 -*-
# @Time    : 18-12-06
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : vgg_encoder.py
# @IDE     : PyCharm Community Edition
"""
packing VGG16 neural network model
"""
from collections import OrderedDict
import tensorflow as tf
from encoder_decoder_model import cnn_base_model


class VGG16Encoder(cnn_base_model.CNNBaseModel):
    """
    packing VGG16 neural network model
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(VGG16Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

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

            bn = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')

            relu = self.relu(input_data=bn, name='relu')

        return relu

    def _fc_stage(self, input_tensor, out_dims, name, use_bias=False):
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

            bn = self.layer_bn(input_data=fc, is_training=self._is_training, name='bn')

            relu = self.relu(input_data=bn, name='relu')

        return relu

    def encode(self, input_tensor, name):
        """
        initialize VGG16 network structure

        :param input_tensor:
        :param name:
        :return:
        """
        ret = OrderedDict()

        with tf.variable_scope(name):
            # conv stage 1_1
            conv_1_1 = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                        out_dims=64, name='conv_1_1')

            # conv stage 1_2
            conv_1_2 = self._conv_stage(input_tensor=input_tensor, k_size=3,
                                        out_dims=64, name='conv_1_2')

            # pool stage 1
            pool1 = self.max_pooling(input_data=conv_1_2, kernel_size=2,
                                     stride=2, name='pool1')

            # conv stage 2_1
            conv_2_1 = self._conv_stage(input_tensor=pool1, k_size=3,
                                        out_dims=128, name='conv_2_1')

            # conv stage 2_2
            conv_2_2 = self._conv_stage(input_tensor=conv_2_1, k_size=3,
                                        out_dims=128, name='conv_2_2')

            # pool stage 2
            pool2 = self.max_pooling(input_data=conv_2_2, kernel_size=2,
                                     stride=2, name='pool2')

            # conv stage 3_1
            conv_3_1 = self._conv_stage(input_tensor=pool2, k_size=3,
                                        out_dims=256, name='conv3_1')

            # conv_stage 3_2
            conv_3_2 = self._conv_stage(input_tensor=conv_3_1, k_size=3,
                                        out_dims=256, name='conv3_2')

            # conv stage 3_3
            conv_3_3 = self._conv_stage(input_tensor=conv_3_2, k_size=3,
                                        out_dims=256, name='conv3_3')

            # pool stage 3
            pool3 = self.max_pooling(input_data=conv_3_3, kernel_size=2,
                                     stride=2, name='pool3')

            ret['pool3'] = dict()
            ret['pool3']['data'] = pool3
            ret['pool3']['shape'] = pool3.get_shape().as_list()

            # conv stage 4_1
            conv_4_1 = self._conv_stage(input_tensor=pool3, k_size=3,
                                        out_dims=512, name='conv4_1')

            # conv stage 4_2
            conv_4_2 = self._conv_stage(input_tensor=conv_4_1, k_size=3,
                                        out_dims=512, name='conv4_2')

            # conv stage 4_3
            conv_4_3 = self._conv_stage(input_tensor=conv_4_2, k_size=3,
                                        out_dims=512, name='conv4_3')

            # pool stage 4
            pool4 = self.max_pooling(input_data=conv_4_3, kernel_size=2,
                                    stride=2, name='pool4')

            ret['pool4'] = dict()
            ret['pool4']['data'] = pool4
            ret['pool4']['shape'] = pool4.get_shape().as_list()

            # conv stage 5_1
            conv_5_1 = self._conv_stage(input_tensor=pool4, k_size=3,
                                        out_dims=512, name='conv5_1')

            # conv stage 5_2
            conv_5_2 = self._conv_stage(input_tensor=conv_5_1, k_size=3,
                                        out_dims=512, name='conv5_2')

            # conv stage 5_3
            conv_5_3 = self._conv_stage(input_tensor=conv_5_2, k_size=3,
                                        out_dims=512, name='conv5_3')

            # pool stage 5
            pool5 = self.max_pooling(input_data=conv_5_3, kernel_size=2,
                                    stride=2, name='pool5')

            ret['pool5'] = dict()
            ret['pool5']['data'] = pool5
            ret['pool5']['shape'] = pool5.get_shape().as_list()

            # fc stage1
            # fc6 = self._fc_stage(input_tensor=pool5, out_dims=4096, name='fc6', use_bias=False)

            # fc stage2
            # fc7 = self._fc_stage(input_tensor=fc6, out_dims=4096, name='fc7', use_bias=False)

        return ret

if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input')
    encoder = VGG16Encoder(phase=tf.constant('train', dtype=tf.string))
    ret = encoder.encode(a, name='encode')
    for layer_name, layer_info in ret.items():
        print("layer name: {:s} shape: {}".format(layer_name, layer_info['shape']))