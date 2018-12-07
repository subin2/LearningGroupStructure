# -*- coding: utf-8 -*-
# @Time    : 18-12-06
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : rcl_encoder.py
# @IDE     : PyCharm Community Edition
"""
this class is used to repeat extracting features according an assigned iter_num
it is as same as conv2d when the iter_num == 1
"""
from collections import OrderedDict
import tensorflow as tf
from encoder_decoder_model import cnn_base_model


class RCLEncoder(cnn_base_model.CNNBaseModel):
    """
    repeat extracting features according an assigned iter_num
    it is as same as conv2d when the iter_num == 1
    """
    def __init__(self, phase, filter_shape, iter_num=3, strides=1, padding='SAME',
                 bias=None, use_batch_norm=True):

        super(RCLEncoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()
        self._iter_num = iter_num
        self._strides = strides
        self._padding = padding
        self._b_init = bias
        self._out_channel = filter_shape[-1]
        self._kernel_size = [filter_shape[0], filter_shape[1]]
        self._use_batch_norm = use_batch_norm

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._train_phase, self._phase)

    def encode(self, input_data, name='RCL_default'):
        """

        :param input_data:
        :param name:
        :return:
        """
        ret = OrderedDict()

        with tf.variable_scope(name):
            if self._iter_num == 0:
                conv = self.conv2d(input_data=input_data, stride=self._strides,
                                   padding=self._padding, b_init=self._b_init,
                                   out_channel=self._out_channel, kernel_size=self._kernel_size)
                if self._use_batch_norm:
                    bn = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')



