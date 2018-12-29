# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : main_cnn_cocluster.py
# @IDE     : PyCharm Community Edition
from main import train_cnn_cocluster
from main import test_cnn_cocluster

if __name__ == '__main__':
    cnn_cocluster_model = train_cnn_cocluster.train()
    accuracy = test_cnn_cocluster.test(cnn_cocluster_model)