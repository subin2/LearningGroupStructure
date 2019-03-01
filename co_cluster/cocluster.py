# -*- coding: utf-8 -*-
# @Time    : 19-1-16
# @Author  : Yang Jiao
# @Site    : https://github.com/mrjiao2018
# @File    : cocluster.py
# @IDE     : PyCharm Community Edition

"""
    在陈武桥实现的版本上将底层的numpy实现转为tensorflow实现
"""
import tensorflow as tf

class BaseCocluster:
    def __init__(self, w, threshold_factor=0.00001, row_cluster_num=2, column_cluster_num=2):
        '''
        init the co-cluster algorithm class
        :param w: the matrix to be co-clustered，height * width
        :param threshold_factor:
        :param row_cluster_num:
        :param column_cluster_num:
        '''
        if isinstance(w, np.ndarray) and len(w.shape) == 4:
            row_num, col_num, in_channel, out_channel = w.shape
            self.w = w
            self.row_num = row_num
            self.col_num = col_num
            self.in_channel = in_channel
            self.out_channel = out_channel
            self.threshold_facotor = threshold_factor
            self.k = row_cluster_num
            self.l = column_cluster_num
            self.run_time = 0
        else:
            raise Exception("wrong array")


    def co_cluster(self):
        pass
