# -*- coding: utf-8 -*-
# @Time    : 18-12-21
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : cnn_cocluster_config.py
# @IDE     : PyCharm Community Edition
"""
    设置全局变量，由cnn-Em.py中的参数调整而来, 578行开始
    以下配置中未包含的参数：
    inputs                  585行
    weight_size             590行
    outputs                 592行
    l_rate = 0.0001         601
    std = 0.05              602
    train_batch_num         606
    test_batch_num          608

    l_step = 300*train_batch_num    610-618
    l_decay=0.1
    cluster_num = 2
    em_layers=[2,3,4,5]
    q_param = 1e-6

    已删除的参数，这些参数在编写神经网络时已经指定，不需要再进行全局配置
    pool
    pool_size
    conv
    NONLINEARITY
    keep_probs
    USE_BATCHNORM
    USE_DROPOUT
    iter
"""
from easydict import EasyDict as edict

__C = edict()

cfg = __C

# Train options
__C.TRAIN = edict()

__C.TRAIN.EPOCH = 400

__C.TRAIN.BATCH_SIZE = 64

__C.TRAIN.FEED_FORWARDS = [512, 128, 10]

__C.TRAIN.STD = 0.05

#__C.TRAIN.ITER=[0,0,0,0]

# __C.TRAIN.POOL=['p','p','p','p']

# __C.TRAIN.POOL_SIZE = [[2,2], [2,2], [2,2], [2,2]]

# __C.TRAIN.CONV = [128,128, 128, 128]

# __C.TRAIN.NONLINEARITY = tf.nn.relu

# __C.TRAIN.KEEP_PROBS = None

# __C.TRAIN.USE_BATCHNORM = False

# __C.TRAIN.USE_DROPOUT = not (__C.TRAIN.KEEP_PROBS == None
#                             or __C.TRAIN.KEEP_PROBS == [1.0 for i in range(len(__C.TRAIN.KEEP_PROBS))])

# __C.TRAIN.OPTIMIZER = tf.train.RMSPropOptimizer

# __C.TRAIN.ERR_FUNC = tf.nn.softmax_cross_entropy_with_logits

