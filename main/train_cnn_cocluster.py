# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : train_cnn_cocluster.py
# @IDE     : PyCharm Community Edition
import time
import tensorflow as tf
from data_provider import data_provider
from config import cnn_cocluster_config
from cnn_cocluster import cnn_cocluster


def train():
    start_time = time.time()

    # get cnn_cocluster_config
    CFG = cnn_cocluster_config.cfg

    # get training data
    train_data, train_label = data_provider.get_train_data()
    input_shape = [CFG.TRAIN.BATCH_SIZE, train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    train_batch_num = train_data.shape[0]/CFG.TRAIN.BATCH_SIZE
    input_data = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
    input_label = tf.placeholder(tf.float32, shape=[input_shape[0], train_label.shape[1]], name='input_label')
    print("input_data:batch, in_height, in_width, in_channels : {:s}".format(str(input_shape)))
    print("finish unzipping training data. {:.3f}s taken.".format(time.time()-start_time))

    # build model
    print("start to build model")
    cnn_cocluster_model = cnn_cocluster.CNNCocluster(phase=tf.constant('train', dtype=tf.string))
    cnn_cocluster_net = cnn_cocluster_model.encode(input_tensor=input_data)
    # print(cnn_cocluster_net.shape) (64, 2, 2, 128)
    print("finish building model. {:.3f}s taken.".format(time.time() - start_time))

    # start training
    print("start training")
    train_start_time = time.time()
    train_loss = []
    for epoch in range(CFG.TRAIN.EPOCH):
        print('start train epoch')
        start_time_per_epoch = time.time()
        loss = 0.0
        batch_index = 0
        for train_input, train_target in data_provider.iterate_minibatches(inputs=train_data,
                                                                           targets=train_label,
                                                                           batchsize=CFG.TRAIN.BATCH_SIZE):
            print('{}/{} batch train start'.format(batch_index+1, train_batch_num))
            batch_index = batch_index + 1
            train_feed_dict = {input_data: train_input, input_label: train_target}
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                err_temp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_cocluster_net,
                                                                                  labels=input_label))
                # todo 在此如何加入global_step来调整学习率
                # https://blog.csdn.net/uestc_c2_403/article/details/72403833
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
                train_step = optimizer.minimize(err_temp, name="optimizer")
                err_temp, opt = sess.run((err_temp, train_step), feed_dict=train_feed_dict)

                loss += err_temp
                train_loss.append(loss/train_batch_num)

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, CFG.TRAIN.EPOCH, time.time() - start_time_per_epoch))
        print("  training loss:    {:.6f}".format(train_loss[-1]))

        # TODO save model
        # TODO draw figures

    print("finish training. Training time {:.3f}s ".format(time.time() - train_start_time))
    return cnn_cocluster_model


if __name__ == '__main__':
    train()