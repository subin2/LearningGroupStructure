# -*- coding: utf-8 -*-
# @Project : LearningGroupStructure
# @Author  : Jinglin Chen
# @File    : res_encoder_jinglin.py
# @Time    : 2018/12/11 11:33
# @IDE     : PyCharm
import pandas as pd
from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import time

from tensorflow.python.training import moving_averages
from sklearn import preprocessing
from encoder_decoder_model import cnn_base_model

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer')


class ResEncoder(cnn_base_model.CNNBaseModel):
    """
    packing Residual network model
    """

    def __init__(self, weight_size, pool_size, inputs, conv, feed_forwards, optimizer=None,
                 std=0.05, l_rate=0.001, l_step=1e15, l_decay=1.0, weight_decay_rate=0.5, relu_leakiness=0.1,
                 keep_probs=None, use_batchnorm=False):
        """ResNet packing.
        Args:

        """
        super(ResEncoder, self).__init__()
        self._is_training = True
        self._weight_size = weight_size
        self._pool_size = pool_size
        self._inputs = inputs
        self._conv = conv
        self._feed_forwards = feed_forwards
        self._optimizer = optimizer
        self._std = std
        self._global_step = tf.Variable(0, trainable=False)
        self._l_rate = tf.train.exponential_decay(l_rate, self._global_step, l_step, l_decay, staircase=True)
        self.use_batchnorm = use_batchnorm
        self._l_step = l_step
        self._l_decay = l_decay
        self._weight_decay_rate = weight_decay_rate
        self._relu_leakiness = relu_leakiness

        self._x = tf.placeholder(tf.float32, shape=inputs, name='x')  # [batch_size, width, height, depth]
        self._y = tf.placeholder(tf.float32, shape=[inputs[0], feed_forwards[-1]], name='y')  # [batch_size, num]
        print('  Start building...')
        self._build_model()
        print('  Done.')

        if keep_probs is None:
            self.keep_probs_values = [1.0 for i in range(len(conv) + len(feed_forwards) - 1)]
        if pool is None:
            self.pool = ['p' for i in range(len(conv))]
        if optimizer is None:
            self._optimizer = tf.train.GradientDescentOptimizer(self._l_rate)
        else:
            self._optimizer = optimizer(self._l_rate)
        self.global_step = tf.Variable(0, trainable=False)
        self.optimize = self._optimizer.minimize(self.cost, global_step=self.global_step)
        self.summaries = tf.summary.merge_all()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        print('Done Res')

    # 构建模型
    def _build_model(self):
        print('    {:{length}} : {}'.format('x', self._x, length=12))
        layer_count = 0
        self.convs = []
        with tf.name_scope('conv' + str(layer_count)):
            x = self
            # 第一层卷积
            layer = self._conv_stage(input_tensor=self._x, k_size=self._weight_size[layer_count],
                                     out_dims=self._weight_size[layer_count][-1],
                                     name='conv_' + str(layer_count), layer_count=layer_count)
        self.convs.append(layer)
        print('    {:{length}} : {}'.format('conv' + str(layer_count), layer, length=12))
        layer_count += 1

        # 残差网络参数
        strides = [1, 1, 1, 1]
        # 标准残差单元模块
        res_func = self._residual

        # 第一层
        with tf.variable_scope('unit_1_0'):
            x = res_func(x, 128, 128,
                         self._stride_arr(strides[0]),
                         16)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv' + str(layer_count), layer, length=12))
            layer_count += 1

        # 第二层
        with tf.variable_scope('unit_2_0'):
            x = res_func(x, 128, 128,
                         self._stride_arr(strides[1]),
                         8)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv' + str(layer_count), layer, length=12))
            layer_count += 1

        # 第三层
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, 128, 128, self._stride_arr(strides[2]),
                         4)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv' + str(layer_count), layer, length=12))
            layer_count += 1

        # 第四层
        with tf.variable_scope('unit_3_0'):
            x = res_func(x, 128, 128, self._stride_arr(strides[3]),
                         2)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv' + str(layer_count), layer, length=12))
            layer_count += 1

        # reshape
        network = tf.reshape(layer, shape=[-1, self._feed_forwards[0]])
        self.flatten = network
        print('    {:{length}} : {}'.format('flatten', self.flatten, length=12))

        # 全连接层
        with tf.variable_scope('logit'):
            logits = self._fully_connected(x, 10)
            self.predictions = tf.nn.softmax(logits)

        # 构建损失函数
        with tf.variable_scope('costs'):
            # 交叉熵
            xent = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=self._y)
            # 加和
            self.cost = tf.reduce_mean(xent, name='xent')
            # L2正则，权重衰减
            self.cost += self._decay()
            # 添加cost总结，用于Tensorborad显示
            tf.summary.scalar('cost', self.cost)

    def _conv_stage(self, input_tensor, k_size, out_dims, name, layer_count,
                    stride=1, padding='SAME', weight=None, biases=None, regularizer=None):
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
            if regularizer is None:
                conv = self.conv2d(input_data=input_tensor,
                                   w_init=weight, b_init=biases,
                                   out_channel=out_dims,
                                   kernel_size=k_size, stride=stride,
                                   use_bias=False, padding=padding, name='conv')
            if self.use_batchnorm:
                conv = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')
            conv = self.relu(input_data=conv, name='relu')
            # pool stage
            conv = self.max_pooling(input_data=conv, kernel_size=2,
                                    stride=2, name='pool' + str(layer_count))
        return conv

    def _fully_connected(self, x, out_dim):
        # 输入转换成2D tensor，尺寸为[N,-1]
        x = tf.reshape(x, [x[0], -1])
        # 参数w，平均随机初始化，[-sqrt(3/dim), sqrt(3/dim)]*factor
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        # 参数b，0值初始化
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer())
        # 计算x*w+b
        return tf.nn.xw_plus_b(x, w, b)

    # 把步长值转换成tf.nn.conv2d需要的步长数组
    def _stride_arr(self, stride):
        return [1, stride, stride, 1]

    # 残差单元模块
    def _residual(self, x, in_filter, out_filter, stride, shape):
        # 前置激活(取残差直连之前进行BN和ReLU）
        with tf.variable_scope('shared_activation'):
            # 先做BN和ReLU激活
            x = self.layer_bn(x, is_training=self._is_training, name='bn')
            x = self._relu(x, self._relu_leakiness)
            # 获取残差直连
            orig_x = x

        # 第1子层
        with tf.variable_scope('sub1'):
            # x卷积，使用输入步长，通道数(in_filter -> out_filter)
            x = self.conv('conv1', x, shape, in_filter, out_filter, stride)

        # 第2子层
        with tf.variable_scope('sub2'):
            # BN和ReLU激活
            x = self.layer_bn(input_data=x, is_training=self._is_training, name='bn')
            x = self._relu(x, self._relu_leakiness)
            # x卷积，步长为1，通道数不变(out_filter)
            x = self.conv('conv2', x, shape, out_filter, out_filter, [1, 1, 1, 1])

        # 合并残差层
        with tf.variable_scope('sub_add'):
            # 当通道数有变化时
            if in_filter != out_filter:
                # 均值池化，无补零
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                # 通道补零(第4维前后对称补零)
                orig_x = tf.pad(orig_x,
                                [[0, 0],
                                 [0, 0],
                                 [0, 0],
                                 [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]
                                 ])
            # 合并残差
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    # 2D卷积
    def conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            # 获取或新建卷积核，正态随机初始化
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
            # 计算卷积
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    # 权重衰减，L2正则loss
    def _decay(self):
        costs = []
        # 遍历所有可训练变量
        for var in tf.trainable_variables():
            # 只计算标有“DW”的变量
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        # 加和，并乘以衰减因子
        return tf.multiply(self._weight_decay_rate, tf.add_n(costs))

    # leaky ReLU激活函数，泄漏参数leakiness为0就是标准ReLU
    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    # 全局均值池化
    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        # 在第2&3维度上计算均值，尺寸由WxH收缩为1x1
        return tf.reduce_mean(x, [1, 2])


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


# Main


if __name__ == '__main__':
    """
    Load dataset
    """

    data_path = './CIFAR-10'

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

    # CIFAR-10 [32, 32, 3]
    # [64,32,32,3] - [50, 1, 112, 112] -- [50, 1, 28, 112] - [50, 1, 7, 112] - [50, 784]

    batch_size = 64
    inputs = [batch_size, train_data.shape[1], train_data.shape[2], train_data.shape[3]]
    conv = [128, 128, 128, 128]  # conv_base, conv, conv
    iter = [0, 0, 0, 0]
    pool = ['p', 'p', 'p', 'p']
    pool_size = [[2, 2], [2, 2], [2, 2], [2, 2]]
    weight_size = [[3, 3, inputs[-1], conv[0]], [3, 3, conv[0], conv[1]], [3, 3, conv[1], conv[2]],
                   [3, 3, conv[2], conv[3]]]
    feed_forwards = [512, 128, 10]
    outputs = [batch_size, feed_forwards[-1]]
    nonlinearity = tf.nn.relu
    err_func = tf.nn.softmax_cross_entropy_with_logits

    keep_probs = None
    use_dropout = not (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))])
    use_batchnorm = False

    optimizer = tf.train.RMSPropOptimizer
    l_rate = 0.0001
    std = 0.05

    num_epochs = 400
    # num_epochs = 2
    train_batch_num = train_data.shape[0] / batch_size
    # valid_batch_num = valid_data.shape[0] / batch_size
    test_batch_num = test_data.shape[0] / batch_size

    l_step = 300 * train_batch_num
    l_decay = 0.1

    input = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input')

    """
    Train
    """

    start_time = time.time()
    model = ResEncoder(weight_size=weight_size,
                       pool_size=pool_size,
                       inputs=inputs,
                       conv=conv,
                       feed_forwards=feed_forwards,
                       std=std)
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
        print('accuracy: {}'.format(accuracy / test_batch_num))
        return accuracy / test_batch_num


    profile = False
    first = True
    test_count = 0
    Q_conv_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()
        loss = 0
        accuracy = 0
        train_test_data = None
        train_test_label = None
        for batch in iterate_minibatches(inputs=train_data, targets=train_label, batchsize=batch_size):
            train_in, train_target = batch
            train_test_data = train_in
            train_test_label = train_target
            # train_in = train_in[:,np.newaxis,:,np.newaxis]
            loss_, accuracy = model.train(train_in, train_target, profile)
            # loss_ = model.train(train_in, train_target, profile)
            # prediction = model.sess.run(model.prediction, feed_dict=train_feed_dict)
            # accuracy = model.get_accuracy(prediction, target)
            # print(loss_)
            profile = False
            loss += loss_

        train_loss.append(loss / train_batch_num)
        # train_accuracy.append(test(train_data, train_label, batch_size, model, train_batch_num))
        train_accuracy.append(accuracy)
        # train_accuracy.append(test(train_data, train_label, batch_size, model, train_batch_num))
        # train_history.loc[epoch] = [epoch + 1, train_loss[-1], train_err[-1],
        #                             time.strftime("%Y-%m-%d-%H:%M", time.localtime())]

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
            # test_history.loc[test_count] = [epoch + 1, train_accuracy[-1], test_accuracy[-1],
            #                                 time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
            test_count += 1
            print("test accuracy:   {:.2f}%".format(test_accuracy[-1] * 100))
