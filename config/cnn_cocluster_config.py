# -*- coding: utf-8 -*-
# @Time    : 18-12-21
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : cnn_cocluster_config.py
# @IDE     : PyCharm Community Edition
"""
    cnn_cocluster_config
"""
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Train options
__C.TRAIN = edict()

__C.TRAIN.EPOCH = 1

__C.TRAIN.BATCH_SIZE = 64

# Test options
__C.TEST = edict()

__C.TEST.BATCH_SIZE = 64


