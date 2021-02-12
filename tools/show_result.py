# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : show_result.py
# @IDE     : PyCharm Community Edition
import tensorflow as tf
"""
    打印结果
"""


def show_w(w, name):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print('{:s} shape:{:s}'.format(name, str(w.shape)))
        print('{:s} : '.format(name))
        print(sess.run(w))


def show_cnn_cocluster_result():
    pass
