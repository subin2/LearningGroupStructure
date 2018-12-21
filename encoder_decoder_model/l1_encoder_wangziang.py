# -*- coding: utf-8 -*-
# @Time    : 18-12-20
# @Author  : Wang ZiAng
# @Site    : http://github.com/gongmm
# @File    : l1_encoder.py
# @IDE     : PyCharm Community Edition
"""
packing CNN-l1
"""
from collections import OrderedDict

import tensorflow as tf
import time
from sklearn import preprocessing
import numpy as np
import pandas as pd
from tensorflow.python.client import timeline

import cnn_base_model


class L1Encoder(cnn_base_model.CNNBaseModel):
    """
        packing CNN-l1
    """

    def __init__(self, weight_size, pool_size, inputs, conv, feed_forwards, optimizer=None,
                 l_rate=0.001, l_step=1e15, l_decay=1.0, weight_decay_rate=0.5,
                 use_bn=False,
                 loss_func=None,
                 keep_probs=None
                 ):
        """
        :param loss_func: loss function
        :param use_bn: weather to use batch normalization
        :param weight_decay_rate: L1 regularization ratio
        :param inputs:
        """
        super(L1Encoder, self).__init__()

        self._inputs = inputs
        self._frozen = False
        self._weight_size = weight_size
        self._pool_size = pool_size
        self._conv = conv
        self.feed_forwards = feed_forwards
        self._use_bn = use_bn
        self._weight_decay_rate = weight_decay_rate
        self._loss_func = loss_func
        self._global_step = tf.Variable(0, trainable=False)
        self._is_training = True
        self._l_rate = tf.train.exponential_decay(l_rate, self._global_step, l_step, l_decay, staircase=True)
        self._x = tf.placeholder(tf.float32, shape=inputs, name='x')  # [batch_size, width, height, depth]
        self._y = tf.placeholder(tf.float32, shape=[inputs[0], feed_forwards[-1]], name='y')  # [batch_size, num]
        self.keep_probs_values = keep_probs
        self.output = None
        self.output_layer = None

        if keep_probs == None:
            self.keep_probs_values = [1.0 for i in range(len(conv)+len(feed_forwards)-1)]
        self.keep_probs = tf.placeholder(tf.float32, [len(self.keep_probs_values)], name='keep_probs')

        self.session_conf = tf.ConfigProto()
        self.session_conf.gpu_options.allow_growth = False
        self.sess = tf.InteractiveSession(config=self.session_conf)

        self.global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, self.global_step, l_step, l_decay, staircase=True)
        self.summaries = tf.summary.merge_all()

        print('  Start building...')
        self.build_model()
        print('  Done.')

        self.sess.run(tf.global_variables_initializer())

        # self.saver = tf.train.Saver(max_to_keep=10000)
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        print('Done L1Encoder.')

    def _conv_stage(self, input_tensor, k_size, out_dims, name, layer_count,
                    stride=1, padding='SAME', weight=None, biases=None):
        """
        packing convolution function and activation function


        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param padding:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(input_data=input_tensor,
                               w_init=weight, b_init=biases,
                               out_channel=out_dims,
                               kernel_size=k_size, stride=stride,
                               use_bias=False, padding=padding, name='conv')

            if self._use_bn:
                conv = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')

            conv = self.relu(input_data=conv, name='relu')
            # pool stage
            pool = self.max_pooling(input_data=conv, kernel_size=2,
                                    stride=2, name='pool' + str(layer_count))

        return pool

    def _full_connected_stage(self, input_tensor, out_dims, name, use_bias=False, use_relu=True):
        """

        :param input_tensor:
        :param out_dims:
        :param name:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name):
            network = self.fully_connect(input_data=input_tensor, out_dim=out_dims, name='fc',
                                    use_bias=use_bias)

            if use_bias:
                network = self.layer_bn(input_data=network, is_training=self._is_training, name='bn')
            if use_relu:
                network = self.relu(input_data=network, name='relu')

        return network

    def build_model(self):

        print('    {:{length}} : {}'.format('x', self._x, length=12))
        layer_count = 0
        self.convs = []
        with tf.name_scope('conv' + str(layer_count + 1)):
            layer = self._conv_stage(input_tensor=self._x, k_size=self._weight_size[layer_count],
                                     out_dims=self._weight_size[layer_count][-1],
                                     name='conv_' + str(layer_count), layer_count=layer_count)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), layer, length=12))
            layer_count += 1

        for i in range(layer_count, len(self._conv)):
            layer = self._conv_stage(input_tensor=self.convs[layer_count - 1], k_size=self._weight_size[layer_count],
                                     out_dims=self._weight_size[layer_count][-1],
                                     name='conv_' + str(layer_count), layer_count=layer_count)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), layer, length=12))
            layer_count += 1

        # 64*512
        network = tf.reshape(layer, shape=[-1, self.feed_forwards[0]])
        self.flatten = network
        print('    {:{length}} : {}'.format('flatten', self.flatten, length=12))

        self.forwards = []
        for f in range(len(self.feed_forwards) - 1):
            if layer_count == 4:
                # full connected 1
                network = self._full_connected_stage(input_tensor=network, out_dims=feed_forwards[-2],
                                                     name="full-connected-1", use_bias=True)
                self.forwards.append(network)
                layer_count += 1
                print('    {:{length}} : {}'.format('feed_forward' + str(f + 1), network, length=12))

            else:
                # full connected 2
                network = self._full_connected_stage(input_tensor=network, out_dims=feed_forwards[-1],
                                                     name="full-connected-2", use_bias=True, use_relu=False)

                self.output = network
                self.output_layer = network
                print('    {:{length}} : {}'.format('feed_forward' + str(f + 2), self.output_layer, length=12))

        if self._loss_func is None:
            self._cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.output, self._y)))
        else:
            err = self._loss_func(logits=self.output, labels=self._y)
            self._cost = tf.reduce_mean(err)

        if optimizer is None:
            self.optimizer = tf.train.GradientDescentOptimizer(self._l_rate)
        else:
            self.optimizer = optimizer(self._l_rate)
        self.optimize = self.optimizer.minimize(self._cost, global_step=self._global_step)

    def freeze(self):
        if not self._frozen:
            self.optimizer = tf.train.GradientDescentOptimizer(self.l_rate)
            self.optimize = self.optimizer.minimize(self._cost, global_step=self.global_step)
            self._frozen = True

    def train(self, data, target, profile=False):
        train_feed_dict = {self._x: data}
        train_feed_dict.update({self._y: target})
        train_feed_dict.update({self.keep_probs: self.keep_probs_values})

        if profile:
            sums, opt, cost, err = self.sess.run((self.summaries, self.optimize, self._cost, self._cost),
                                                 feed_dict=train_feed_dict,
                                                 options=self.options,
                                                 run_metadata=self.run_metadata
                                                 )
            return sums, cost, err
        # sums = self.sess.run(self.summaries,feed_dict=train_feed_dict)
        opt = self.sess.run(self.optimize, feed_dict=train_feed_dict)
        cost = self.sess.run(self._cost,feed_dict=train_feed_dict)

        # sums, opt, cost = self.sess.run((self.summaries, self.optimize, self._cost),
        #                                      feed_dict=train_feed_dict
        #                                      )
        return cost

    def test(self, data, target):
        test_feed_dict = {self._x: data}
        test_feed_dict.update({self._y: target})
        keep_probs_values = [1.0 for i in range(len(self.keep_probs_values))]
        test_feed_dict.update({self.keep_probs: keep_probs_values})
        cost, err = self.sess.run((self._cost, self._cost),
                                  feed_dict=test_feed_dict
                                  )
        return cost, err

    # def encode(self, input_tensor, name, num_iter=3):
    #     """
    #     initialize L1 Encoder structure
    #
    #     :param num_iter:
    #     :param input_tensor:
    #     :param name:
    #     :return:
    #     """
    #     print('    {:{length}} : {}'.format('x', input_tensor, length=12))
    #     layer_count = 0
    #
    #     ret = OrderedDict()
    #     # conv_1
    #     with tf.variable_scope(name + str(layer_count + 1)):
    #         if num_iter == 0:
    #             # conv stage
    #             network = self._conv_stage(k_size=3, out_dims=64, name='conv_' + str(layer_count))
    #         else:
    #             for i in range(num_iter):
    #                 network = input_tensor
    #                 network = self._conv_stage(k_size=3, out_dims=64, name='conv' + str(layer_count) + '_' + str(i))
    #             # pool stage
    #             pool = self.max_pooling(input_data=network, kernel_size=2,
    #                                     stride=2, name='pool' + str(layer_count))
    #             ret['pool1'] = dict()
    #             ret['pool1']['data'] = pool
    #             ret['pool1']['shape'] = pool.get_shape().as_list()
    #
    #         print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
    #         layer_count += 1
    #     # conv_2
    #     with tf.variable_scope(name + str(layer_count + 1)):
    #         if num_iter == 0:
    #             # conv stage
    #             network = self._conv_stage(input_tensor=input_tensor, k_size=3,
    #                                        out_dims=64, name='conv_' + str(layer_count))
    #         else:
    #             for i in range(num_iter):
    #                 network = input_tensor
    #                 network = self._conv_stage(input_tensor=network, k_size=3,
    #                                            out_dims=64, name='conv' + str(layer_count) + '_' + str(i))
    #             # pool stage
    #             pool = self.max_pooling(input_data=network, kernel_size=2,
    #                                     stride=2, name='pool' + str(layer_count))
    #             ret['pool2'] = dict()
    #             ret['pool2']['data'] = pool
    #             ret['pool2']['shape'] = pool.get_shape().as_list()
    #
    #         print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
    #         layer_count += 1
    #     # conv_3
    #     with tf.variable_scope(name + str(layer_count + 1)):
    #         if num_iter == 0:
    #             # conv stage
    #             network = self._conv_stage(k_size=3, out_dims=64, name='conv_' + str(layer_count))
    #         else:
    #             for i in range(num_iter):
    #                 network = input_tensor
    #                 network = self._conv_stage(k_size=3, out_dims=64, name='conv' + str(layer_count) + '_' + str(i))
    #             # pool stage
    #             pool = self.max_pooling(input_data=network, kernel_size=2,
    #                                     stride=2, name='pool' + str(layer_count))
    #             ret['pool3'] = dict()
    #             ret['pool3']['data'] = pool
    #             ret['pool3']['shape'] = pool.get_shape().as_list()
    #
    #         print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
    #         layer_count += 1
    #     # conv_4
    #     with tf.variable_scope(name + str(layer_count + 1)):
    #         if num_iter == 0:
    #             # conv stage
    #             network = self._conv_stage(input_tensor=input_tensor, k_size=3,
    #                                        out_dims=64, name='conv_' + str(layer_count))
    #         else:
    #             for i in range(num_iter):
    #                 network = input_tensor
    #                 network = self._conv_stage(input_tensor=network, k_size=3,
    #                                            out_dims=64, name='conv' + str(layer_count) + '_' + str(i))
    #             # pool stage
    #             pool = self.max_pooling(input_data=network, kernel_size=2,
    #                                     stride=2, name='pool' + str(layer_count))
    #             ret['pool4'] = dict()
    #             ret['pool4']['data'] = pool
    #             ret['pool4']['shape'] = pool.get_shape().as_list()
    #
    #         print('    {:{length}} : {}'.format('conv' + str(layer_count + 1), network, length=12))
    #         layer_count += 1
    #
    #     with tf.variable_scope('logit'):
    #         self._output = network  # .get_layer()
    #         self._output_layer = network
    #         # self._output_layer = self._full_connected_stage(input_tensor, self.hps.num_classes)
    #         # self._predictions = tf.nn.softmax(self._output_layer)
    #         # fc6 = self._full_connected_stage(input_tensor=pool1, out_dims=4096, name='fc6', use_bias=False)
    #
    #     with tf.variable_scope('loss'):
    #         # 构建损失函数
    #         if self._loss_func is None:
    #             self._cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(self._output_layer, self._output_layer)))
    #             # L1 正则化
    #             self._cost += self._decay()
    #             # self.err = tf.sqrt(tf.losses.mean_squared_error(self.output_layer, self.y))
    #         else:
    #             # self.err =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.y))
    #             cost = self._loss_func(logits=self._output_layer, labels=self._y)
    #             self._cost = tf.reduce_mean(cost)
    #             # L1 正则化
    #             self._cost += self._decay()
    #     return ret

    # 权重衰减，L2正则loss

    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(self._weight_decay_rate, tf.add_n(costs))


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        # dict = pickle.load(fo)
        dict = pickle.load(fo, encoding='latin1')
    return dict

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

if __name__ == '__main__':
    tf.reset_default_graph()
    """
    Load dataset
    """

    data_path = '../CIFAR-10'

    one_hot_enc = preprocessing.OneHotEncoder(n_values=10, sparse=False)
    # one_hot_enc = preprocessing.OneHotEncoder(categories=[range(10)], sparse=False)

    train_data = []
    train_label = []
    for i in range(5):
        tmp = unpickle('../cifar-10-batches-py/data_batch_' + str(i + 1))
        # print(tmp)
        train_data.append(tmp["data"])
        train_label.append(tmp["labels"])
    train_data = np.concatenate(train_data).reshape([-1, 32, 32, 3], order='F')
    train_label = np.concatenate(train_label)
    train_label = one_hot_enc.fit_transform(train_label.reshape([-1, 1]))
    print("train data: {}, {}".format(train_data.shape, train_label.shape))

    test_data = unpickle('../cifar-10-batches-py/test_batch')['data'].reshape([-1, 32, 32, 3], order='F')
    test_label = unpickle('../cifar-10-batches-py/test_batch')['labels']
    test_label = np.array(test_label)
    test_label = one_hot_enc.fit_transform(test_label.reshape([-1, 1]))
    print("test data: {}, {}".format(test_label.shape, test_label.shape))

    """
    Set Parameters
    """

    batch_size = 64
    inputs = [batch_size, train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    conv = [128, 128, 128, 128]  # conv_base, conv, conv
    pool_size = [[2, 2], [2, 2], [2, 2], [2, 2]]
    weight_size = [[3, 3, inputs[-1], conv[0]], [3, 3, conv[0], conv[1]], [3, 3, conv[1], conv[2]],
                   [3, 3, conv[2], conv[3]]]
    feed_forwards = [512, 128, 10]
    outputs = [batch_size, feed_forwards[-1]]
    err_func = tf.nn.softmax_cross_entropy_with_logits

    optimizer = tf.train.RMSPropOptimizer
    l_rate = 0.0001
    keep_probs = None

    num_epochs = 2
    # num_epochs = 2
    train_batch_num = train_data.shape[0] / batch_size
    print("train_batch_num: %f", train_batch_num)
    # valid_batch_num = valid_data.shape[0] / batch_size
    test_batch_num = test_data.shape[0] / batch_size

    l_step = 300 * train_batch_num
    l_decay = 0.1

    a = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input')

    """
    Train
    """
    start_time = time.time()

    model = L1Encoder(weight_size=weight_size, pool_size=pool_size,
                      inputs=inputs,
                      conv=conv,
                      feed_forwards=feed_forwards,
                      loss_func=err_func,
                      l_rate=l_rate,
                      optimizer=optimizer,
                      )
    print('Done model. {:.3f}s taken.'.format(time.time() - start_time))

    valid_freq = 10
    save_freq = 50
    frozen_epoch = 0  # 350
    test_epoch = [frozen_epoch, 300, num_epochs]

    train_history = pd.DataFrame(index=np.arange(0, num_epochs),
                                 columns=['epoch', 'loss', 'err', 'timestamp'])
    valid_history = pd.DataFrame(index=np.arange(0, num_epochs / valid_freq),
                                 columns=['epoch', 'loss', 'err', 'timestamp'])
    test_history = pd.DataFrame(index=np.arange(0, len(test_epoch)),
                                columns=['epoch', 'train accuracy', 'test accuracy', 'timestamp'])

    param_history = pd.DataFrame(index=np.arange(0, num_epochs / save_freq))

    train_loss = []
    train_err = []
    valid_loss = []
    valid_err = []
    train_accuracy = []
    test_accuracy = []

    # ret = model.encode(a, name='encode')
    # for layer_name, layer_info in ret.items():
    #     print("layer name: {:s} shape: {}".format(layer_name, layer_info['shape']))

    def test(test_data, test_labels, batch_size, model, test_batch_num):
        accuracy = 0.0
        keep_probs_values = [1.0 for i in range(len(model.keep_probs_values))]
        for batch in iterate_minibatches(inputs=test_data, targets=test_labels, batchsize=batch_size):
            test_in, test_target = batch
            # test_in = test_in[:,np.newaxis,:,np.newaxis]
            # print model.sess.run(tf.reduce_sum(tf.equal(tf.argmax(model.output_layer,1), tf.argmax(model.y, 1))) ,
            #                            feed_dict={model.x:test_in, model.y:test_target})
            accuracy += model.sess.run(
                tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.output_layer, 1), tf.argmax(model._y, 1)), tf.float32)),
                feed_dict={model._x: test_in, model._y: test_target, model.keep_probs: keep_probs_values})
        # print'accuracy: {}'.format(accuracy/test_batch_num)
        return accuracy / test_batch_num


    profile = False
    first = True
    test_count = 0
    Q_conv_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        loss = 0
        err = 0
        for batch in iterate_minibatches(inputs=train_data, targets=train_label, batchsize=batch_size):
            train_in, train_target = batch
            # train_in = train_in[:,np.newaxis,:,np.newaxis]
            loss_ = model.train(train_in, train_target, profile)
            print(loss_)
            profile = False
            loss += loss_

        train_loss.append(loss / train_batch_num)
        train_err.append(err / train_batch_num)
        train_accuracy.append(test(train_data, train_label, batch_size, model, train_batch_num))
        train_history.loc[epoch] = [epoch + 1, train_loss[-1], train_err[-1],
                                    time.strftime("%Y-%m-%d-%H:%M", time.localtime())]

        if (epoch + 1) % save_freq == 0:

            w_mask_pass = []

            param_history.loc[epoch / save_freq] = [epoch + 1] + w_mask_pass + [
                time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:    {:.6f}".format(train_loss[-1]))
        print("  training acc:     {:.6f}".format(train_accuracy[-1]))

        if (epoch + 1) in test_epoch:
            test_accuracy.append(test(test_data, test_label, batch_size, model, test_batch_num))
            train_accuracy.append(test(train_data, train_label, batch_size, model, train_batch_num))
            test_history.loc[test_count] = [epoch + 1, train_accuracy[-1], test_accuracy[-1],
                                            time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
            test_count += 1
