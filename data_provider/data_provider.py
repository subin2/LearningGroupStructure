# -*- coding: utf-8 -*-
# @Time    : 18-12-21
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : data_provider.py
# @IDE     : PyCharm Community Edition
"""
    对训练数据进行预处理
"""
import numpy as np
from sklearn import preprocessing
import pickle


def get_train_data():
    """
    获取cifar中的训练数据

    :return: train_data, train_label
    """
    one_hot_enc = preprocessing.OneHotEncoder(n_values=10, sparse=False)

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

    return train_data, train_label


def get_test_data():
    """
    获取cifar中的测试数据

    :return: test_data, test_label
    :return:
    """
    one_hot_enc = preprocessing.OneHotEncoder(n_values=10, sparse=False)
    test_data = unpickle('../cifar-10-batches-py/test_batch')['data'].reshape([-1, 32, 32, 3], order='F')
    test_label = unpickle('../cifar-10-batches-py/test_batch')['labels']
    test_label = np.array(test_label)
    test_label = one_hot_enc.fit_transform(test_label.reshape([-1, 1]))
    print("test data: {}, {}".format(test_label.shape, test_label.shape))

    return test_data, test_label


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """
    分批返回训练数据与标签

    :param inputs:
    :param targets:
    :param batchsize:
    :param shuffle:
    :return:
    """
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


def unpickle(file):
    """
    反序列化

    :param file:
    :return:
    """
    with open(file, 'rb') as fo:
        # dict = pickle.load(fo)
        dict = pickle.load(fo, encoding='latin1')
    return dict


if __name__ == '__main__':
    get_train_data()
    get_test_data()