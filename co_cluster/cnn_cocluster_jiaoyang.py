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
from encoder_decoder_model import rcl_encoder_origin

class CNNCocluster():
    """

    """
    def __init__(self):
        pass


    def _build_fp_model(self,
                        input_tensor,
                        weight_size,
                        use_dropout = False,
                        name='fp_model'):
        with tf.name_scope(name):
            layer = rcl_encoder_origin.RCL(input=input_tensor,
                                           weight_size=weight_size)


    def _co_cluster(self):
        pass

    def build_model(self):
        pass
