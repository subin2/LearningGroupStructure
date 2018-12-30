# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : test_cnn_cocluster.py
# @IDE     : PyCharm Community Edition
"""
    test
"""
import time
from data_provider import data_provider
import tensorflow as tf
from config import cnn_cocluster_config
from cnn_cocluster import cnn_cocluster


def test(cnn_cocluster_model):
    start_time = time.time()

    # get cnn_cocluster_config
    CFG = cnn_cocluster_config.cfg

    # get test data
    test_data, test_label = data_provider.get_test_data()
    input_shape = [CFG.TEST.BATCH_SIZE, test_data.shape[1], test_data.shape[2], test_data.shape[3]]
    test_batch_num = test_data.shape[0] / CFG.TEST.BATCH_SIZE
    input_data = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
    input_label = tf.placeholder(tf.float32, shape=[input_shape[0], test_label.shape[1]], name='input_label')
    print("input_data:batch, in_height, in_width, in_channels : {:s}".format(str(input_shape)))
    print("finish unzipping test data. {:.3f}s taken.".format(time.time() - start_time))

    # biuld model
    print("start to build model")
    # todo 此处目前是通过参数将模型传递，但最好应该改成：将已经训练好的模型的参数保存，test时直接加载这些参数建立网络
    # cnn_cocluster_model = cnn_cocluster.CNNCocluster(phase=tf.constant('test', dtype=tf.string))
    cnn_cocluster_net = cnn_cocluster_model.encode(input_tensor=input_data)
    print("finish building model. {:.3f}s taken.".format(time.time() - start_time))

    # start test
    test_start_time = time.time()
    accuracy = 0.0
    batch_index = 0
    for test_input, test_target in data_provider.iterate_minibatches(inputs=test_data,
                                                                     targets=test_label,
                                                                     batchsize=CFG.TEST.BATCH_SIZE):
        print('{}/{} batch test start'.format(batch_index + 1, test_batch_num))
        batch_index = batch_index + 1
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            predict = tf.argmax(cnn_cocluster_net, 1)
            truth = tf.argmax(test_target, 1)
            temp_accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, truth), tf.float32))
            accuracy += sess.run(temp_accuracy, feed_dict={input_data: test_input, input_label: test_target})

    print("finish test. {:.3f}s token".format(time.time() - test_start_time))
    print("accuracy: {}".format(accuracy / test_batch_num))
    return accuracy / test_batch_num


if __name__ == '__main__':
    # test(None)
    pass