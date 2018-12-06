# -*- coding: utf-8 -*-
# @Time    : 18-12-06
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : rcl_encoder.py
# @IDE     : PyCharm Community Edition
"""
repeat extracting features according an assigned iter_num
it is same as conv2d when the iter_num == 1
"""
from collections import OrderedDict
import tensorflow as tf
from encoder_decoder_model import cnn_base_model


class RCLEncoder(cnn_base_model.CNNBaseModel):
    """
    repeat extracting features according an assigned iter_num
    it is same as conv2d when the iter_num == 1
    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(RCLEncoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._train_phase, self._phase)

    def encode(self):
        return None