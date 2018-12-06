# -*- coding: utf-8 -*-
# @Time    : 18-12-06
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : feedforward_encoder.py
# @IDE     : PyCharm Community Edition
"""
packing feedforward neural network model
"""
from collections import OrderedDict
import tensorflow as tf
from encoder_decoder_model import cnn_base_model


class FeedforwardEncoder(cnn_base_model.CNNBaseModel):
    """
    packing feedforward neural network model
    """
    def __init__(self):
        super(FeedforwardEncoder, self).__init__()
