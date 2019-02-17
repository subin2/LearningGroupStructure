#!/usr/bin/env python
# coding=utf-8


import os
import csv
import time
import random
# import cPickle
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib import colors
from sklearn import preprocessing

import numpy as np
import tensorflow as tf
from tensorflow.contrib import graph_editor as ge
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

import utils
from utils import iterate_minibatches as iterate_minibatches
import models


# N
def gmm_pdf_log(x, mu=[], sigma=[], sess=None):
    #check shape
    if type(mu) is list:
        multi=True
        if len(mu) != len(sigma):
            raise ValueError('mu and sigma number not matched.')
    if multi:
        #tf.split(split_dim, num_split, value, name='split')
        x = [tf.reshape(t, [-1]) for t in tf.split(axis=0, num_or_size_splits=x.get_shape()[0].value, value=x)] #[[100],[100],[100]]
        # D
        dim=mu[-1].get_shape()[-1].value #100
    else:
        if len(x.get_shape()) != len(mu.get_shape()) or x.get_shape()[0].value!=mu.get_shape()[0].value:
            raise ValueError('x shape error')
        dim=mu.get_shape()[-1].value
    # return pdf. Watch it is not log(pdf).
    if multi:
        output=[]
        for i in range(len(mu)): #3
            flag=False
            # make covariance matrix positive diagonal matrix with some noise added
            # 当前类的协方差矩阵
            tmp_sigma = tf.abs(sigma[i])
            # 协方差矩阵对角线元素构成对角矩阵
            tmp_sigma = tf.matrix_diag(tf.matrix_diag_part(tmp_sigma))
            # 当前类的均值向量
            tmp_mu = mu[i]
            # 降维加法，log|协方差矩阵|
            log_det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tmp_sigma) + 1e-40))  # 加噪声

            # 协方差矩阵^(-1)
            inv = tf.clip_by_value(tf.matrix_diag(tf.div(tf.cast(1.0, dtype=tf.float64),
                                                                         tf.matrix_diag_part(tmp_sigma))),
                                           clip_value_min=-1e30, clip_value_max=1e30)
            # e前参数部分的log
            tmp1 = -(dim*0.5*tf.log(tf.cast(2*np.pi, dtype=tf.float64))) - (0.5*log_det)

            tmp2_1 = tf.matmul(tf.matmul(tf.reshape(tf.cast(x[i], dtype=tf.float64) - tmp_mu,[1,-1]),
                                         inv),#tf.matrix_inverse(sigma)),
                               tf.reshape(tf.cast(x[i], dtype=tf.float64)-tmp_mu,[-1,1]))
            # e的指数部分
            tmp2 = -0.5*tmp2_1#tf.matmul(tmp2_1, -0.5)
            #pdf = tf.exp(tf.clip_by_value(tmp1+tmp2, clip_value_min=-100.0, clip_value_max=85.0)) #remove log
            # 去掉log
            pdf = tf.exp(tmp1+tmp2) #remove log
            # N
            output.append(pdf)
        output = tf.reshape(tf.concat(values=output, axis=0),[-1])
        return output
    else:
        #det = tf.matrix_determinant(self.sigma[i])
        # 行列式是否为0
        det = tf.cond(tf.equal(tf.matrix_determinant(sigma),0),
                    lambda:  tf.constant(1e-30),
                      # 取方阵行列式
                    lambda: tf.matrix_determinant(sigma))
        tmp1_1 = tf.matmul(tf.pow((2*np.pi),dim), det)
        tmp1 = tf.pow(tmp1_1, -1.0/2)
        tmp2 = tf.exp(-tf.matmul(tf.matmul(tf.reshape(x-mu,[1,-1]),
                                            tf.matrix_inverse(sigma)),
                                tf.reshape(x-mu,[-1,1])
                                )/2)
        output = tf.matmul(tmp1,tmp2)
        return output




class grcnn(object):
    def __init__(self, weight_size, pool_size, inputs, conv, feed_forwards, outputs, iter=[], cluster_num=4, em_layers=[2], pool=None,
                 nonlinearity=None, keep_probs=None, use_dropout=False, use_batchnorm=False, err_func=None,
                 optimizer=None, l_rate=0.001, l_step=1e15, l_decay=1.0, q_param=1e-5,
                 std=0.05, offset=1e-10, scale=1, epsilon=1e-10, summary_path='./'):
        print('Start constructing...')
        self.init = True
        self.frozen = False
        self.weight_size = weight_size
        self.pool_size = pool_size
        self.pool = pool
        self.conv = conv
        self.feed_forwards = feed_forwards
        self.iter = iter
        self.cluster_num = cluster_num
        self.em_layers = em_layers
        self.nonlinearity = nonlinearity
        self.err_func = err_func
        self.keep_probs_values = keep_probs
        self.use_batchnorm = use_batchnorm
        self.use_dropout=use_dropout
        if keep_probs == None:
            self.keep_probs_values = [1.0 for i in range(len(conv)+len(feed_forwards)-1)]
        if pool == None:
            self.pool = ['p' for i in range(len(conv))]
        self.keep_probs = tf.placeholder(tf.float32, [len(self.keep_probs_values)], name='keep_probs')
        self.std = std # random_normal
        self.q_param = q_param
        self.offset=offset
        self.scale=scale
        self.epsilon=epsilon
        
        self.session_conf = tf.ConfigProto()
        self.session_conf.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=self.session_conf)
        
        self.global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, self.global_step, l_step, l_decay, staircase=True)
        
        self.x = tf.placeholder(tf.float32, shape=inputs, name='x')  # [batch_size, width, height, depth]
        self.y = tf.placeholder(tf.float32, shape=[inputs[0],feed_forwards[-1]], name='y') # [batch_size, num]
        print('  Start building...')
        self.build_model()
        print('  Done.')
        if err_func is None:
            self.err = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.output_layer, self.y)))
            #self.err = tf.sqrt(tf.losses.mean_squared_error(self.output_layer, self.y))
        else:
            #self.err = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.y))
            err = self.err_func(logits=self.output_layer, labels=self.y)
            self.err = tf.reduce_mean(err)
        self.err_summary = tf.summary.scalar("err", self.err)
        self.cost = self.err - self.q_param*tf.cast(self.Q, tf.float32)
        if optimizer is None:
            self.optimizer = tf.train.GradientDescentOptimizer(self.l_rate)
        else:
            self.optimizer = optimizer(self.l_rate)
        self.optimize = self.optimizer.minimize(self.cost, global_step=self.global_step)
        
        self.summaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(summary_path, "train_log"), self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        print('Done.')
        
    def build_model(self): 
        print('    {:{length}} : {}'.format('x', self.x, length=12))
        layer_count=0
        self.convs = []
        with tf.name_scope('conv'+str(layer_count+1)):
            layer = models.RCL(input=self.x, 
                               weight_size=self.weight_size[layer_count],
                               pool=self.pool[layer_count],
                               pool_size=self.pool_size[layer_count], 
                               num_iter=self.iter[layer_count], 
                               nonlinearity=self.nonlinearity, 
                               use_dropout=self.use_dropout,
                               keep_prob=self.keep_probs[layer_count], 
                               use_batchnorm=self.use_batchnorm, 
                               std=self.std)
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv'+str(layer_count+1), layer.get_layer(), length=12))
            layer_count += 1
        # 输出层的节点数
        length = self.weight_size[layer_count][-1]#layer.get_layer().get_shape()[2].value
        # 均值向量矩阵
        mu_init_value = np.zeros([self.cluster_num, length])
        # 协方差矩阵
        sigma_init_value = np.zeros([self.cluster_num, length, length])
        # 混合模型的混合权重或混合系数
        pi_init_value = np.ones([cluster_num]) / self.cluster_num
        # 数组 大小：cluster_num聚类量，float
        self.mu = [tf.Variable(tf.random_normal([length], dtype=tf.float64), name='mu'+str(t)) for t in range(self.cluster_num)]
        self.sigma = [tf.Variable(tf.random_normal([length,length], dtype=tf.float64), name='sigma'+str(t)) for t in range(self.cluster_num)]
        # cluster_num * cluster_num 矩阵
        self.pi = tf.Variable(tf.multiply(tf.ones([1, self.cluster_num], tf.float64), pi_init_value),
                            trainable=True,name='pi')
        # force the sum of elements of pi vector to be 1.
        self.pi_normed = tf.div(tf.maximum(self.pi, 0.0), tf.reduce_sum(tf.maximum(self.pi, 0.0))) 
        
        ### convs before em
        # 根据作者的参数设定，该for循环不执行
        for i in range(layer_count, self.em_layers[0]-1):
            layer = models.RCL(input=layer.get_layer(),
                               weight_size=self.weight_size[layer_count],
                               weight=self.w_masked,
                               biases=self.b,
                               pool=self.pool[layer_count],
                               pool_size=self.pool_size[layer_count], 
                               num_iter=self.iter[layer_count], 
                               nonlinearity=self.nonlinearity, 
                               use_dropout=self.use_dropout,
                               keep_prob=self.keep_probs[layer_count], 
                               use_batchnorm=self.use_batchnorm, 
                               std=self.std,
                               name='conv'+str(layer_count+1))
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv'+str(layer_count+1), layer.get_layer(), length=12))
            layer_count += 1
        #        
        ### em
        self.em_w=[]
        self.w_mask=[]
        self.w_masked=[]    # 为二维矩阵，每一维都是一个处理后的权重参数矩阵，矩阵中较小值全部为0，较大值保留
        self.cluster = []
        self.max_idx = []
        for em in range(len(self.em_layers)):
            with tf.name_scope('conv'+str(layer_count+1)+'em'):
                # 以输入节点个数和输出节点个数建立矩阵
                self.em_w.append(tf.Variable( tf.random_normal( self.weight_size[layer_count], stddev=self.std, dtype=tf.float32), name='w' ))                
                if em == 0:
                    # 第k个高斯分量 对于“解释”观测值 x_i 的“责任”
                    gamma_elem = []
                    # 对数似然函数的期望
                    Q_elem = []
                    self.x_batch = tf.reduce_max(self.em_w[-1], axis=[0,1]) - tf.reduce_min(self.em_w[-1], axis=[0,1])
                    # 输入节点循环
                    for w in range(self.weight_size[layer_count][-2]):
                        # 高斯混合模型对数似然函数N
                        x_pdf = gmm_pdf_log(mu=self.mu, 
                                                        sigma=self.sigma, 
                                                        x=tf.reshape(tf.tile(self.x_batch[w,:],[self.cluster_num]),[self.cluster_num,-1]), #[3, 100]
                                                       sess=self.sess)                
                        # 乘以系数的 N
                        pi_pdf = tf.multiply(self.pi_normed, x_pdf)

                        # 可以理解为分量 k 对于“解释”观测值 x 的“责任”
                        gamma_tmp = tf.reshape(tf.div(pi_pdf,
                                                      tf.maximum(tf.reduce_sum(pi_pdf),1e-30)),
                                               [-1])
                        # 不要算梯度
                        gamma_tmp = tf.stop_gradient(gamma_tmp) # fix the value. do not calculate the gradient of this term.
                        # 2维
                        gamma_elem.append(gamma_tmp)
                        tmp = tf.reduce_sum(tf.multiply(gamma_tmp, 
                                                                    tf.log(pi_pdf+1e-30)))
                        Q_elem.append(tmp)
                    # 对数似然函数的期望
                    self.Q = tf.reduce_sum(Q_elem)
                    self.Q_summary = tf.summary.scalar("Q", self.Q)

                    self.gamma = tf.stack(gamma_elem)
                    # 返回行中最大值所在的下标
                    # cluster 1维
                    self.cluster.append(tf.cast(tf.argmax(self.gamma, axis=1), dtype=tf.int32))
                    print('      {:{length}} : {}'.format('cluster', self.cluster[-1], length=12))
                    # 行聚类
                    i = tf.constant(0)
                    # 数组
                    w_mean = tf.TensorArray(dtype=tf.float32, size=self.cluster_num)#tf.constant(0.0, shape=tf.TensorShape([]))

                    cond = lambda i,w_mean : i<self.cluster_num
                    # rep value
                    x_batch = tf.reduce_max(self.em_w[-1], axis=[0,1]) - tf.reduce_min(self.em_w[-1], axis=[0,1])
                    # 求每个组的均值向量
                    def func(i,w_mean):
                        mean = tf.reduce_mean(tf.boolean_mask(x_batch, tf.equal(self.cluster[-1],i)), axis=[0])
                        w_mean = w_mean.write(i, mean)
                        return i+1, w_mean
                    # 求均值向量
                    i, w_mean = tf.while_loop(cond, func, [i,w_mean])
                    self.max_idx.append(tf.cast(tf.argmax(w_mean.stack(), axis=0), tf.int32))
                    print('      {:{length}} : {}'.format('max_idx', self.max_idx[-1], length=12))
                else:
                    # em!= 0
                    self.cluster.append(self.max_idx[-1])
                    print('      {:{length}} : {}'.format('cluster', self.cluster[-1], length=12))
                    #
                    i = tf.constant(0)
                    w_mean = tf.TensorArray(dtype=tf.float32, size=self.cluster_num)#tf.constant(0.0, shape=tf.TensorShape([]))
                    cond = lambda i,w_mean : i<self.cluster_num
                    x_batch = tf.reduce_max(self.em_w[-1], axis=[0,1]) - tf.reduce_min(self.em_w[-1], axis=[0,1])
                    def func(i,w_mean):
                        mean = tf.reduce_mean(tf.boolean_mask(x_batch, tf.equal(self.cluster[-1],i)), axis=[0])
                        w_mean = w_mean.write(i, mean)
                        return i+1, w_mean
                    # 循环每一个输入节点
                    i, w_mean_ = tf.while_loop(cond, func, [i,w_mean])
                    self.max_idx.append(tf.cast(tf.argmax(w_mean_.stack(), axis=0), tf.int32))
                    print('      {:{length}} : {}'.format('max_idx', self.max_idx[-1], length=12))
                # 列聚类
                i = tf.constant(0)
                w_mask_array = tf.TensorArray(dtype=tf.float32, size=self.weight_size[layer_count][-1])
                cond2 = lambda i,w_mask_array : i<self.weight_size[layer_count][-1]
                def func2(i, w_mask_array):
                    w_mask_array_column = tf.cast(tf.equal(self.cluster[-1], self.max_idx[-1][i]), dtype=tf.float32)
                    w_mask_array = w_mask_array.write(i, w_mask_array_column)
                    return i+1, w_mask_array
                i, w_mask_array = tf.while_loop(cond2, func2, [i, w_mask_array])
                w_mask_pack = tf.transpose(w_mask_array.stack())
                self.w_mask.append(tf.expand_dims(tf.stack([w_mask_pack for i in range(self.weight_size[layer_count][1])]), 0))
                self.w_masked.append(tf.multiply(self.em_w[-1], self.w_mask[-1]))
            # end if-else
            layer = models.RCL(input=layer.get_layer(),
                               weight_size=self.weight_size[layer_count],
                               weight=self.w_masked[-1],
                               pool=self.pool[layer_count],
                               pool_size=self.pool_size[layer_count], 
                               num_iter=self.iter[layer_count], 
                               nonlinearity=self.nonlinearity, 
                               use_dropout=self.use_dropout,
                               keep_prob=self.keep_probs[layer_count], 
                               use_batchnorm=self.use_batchnorm, 
                               std=self.std,
                              name='conv'+str(layer_count+1))
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv'+str(layer_count+1), layer.get_layer(), length=12))
            layer_count += 1
            # 第一次执行至此时，根据作者的参数设定，layer_count = 2, len(self.conv) = 4
            # 所以该 for 循环一共执行3次，分别为layer_count = 2、3、4时
            # 退出该 for 循环时，layer_count = 4
            if layer_count>=len(self.conv):
                break
        # end for

        ### left conv layers
        # 根据作者的参数设定，该for循环不执行
        for i in range(layer_count, len(self.conv)):
            layer = models.RCL(input=layer.get_layer(),
                               weight_size=self.weight_size[layer_count],
                               pool=self.pool[layer_count],
                               pool_size=self.pool_size[layer_count], 
                               num_iter=self.iter[layer_count], 
                               nonlinearity=self.nonlinearity, 
                               use_dropout=self.use_dropout,
                               keep_prob=self.keep_probs[layer_count], 
                               use_batchnorm=self.use_batchnorm, 
                               std=self.std,
                               name='conv'+str(layer_count+1))
            self.convs.append(layer)
            print('    {:{length}} : {}'.format('conv'+str(layer_count+1), layer.get_layer(), length=12))
            layer_count += 1
        
        network = tf.reshape(layer.get_layer(), shape=[-1, self.feed_forwards[0]])# * self.keep_probs[1]]) ###
        self.flatten = network
        print('    {:{length}} : {}'.format('flatten', self.flatten, length=12))

        # 该 if 不执行
        if len(self.feed_forwards) == 2:
            network = models.feedforward(input = network,
                                         weight_size=[self.feed_forwards[0], self.feed_forwards[1]],
                                         nonlinearity=None,
                                         use_dropout = False, 
                                         use_batchnorm = False,
                                         std=self.std,
                                         offset=self.offset,
                                         scale=self.scale,
                                         epsilon=self.epsilon, 
                                         name='output')
            self.output = network#.get_layer()
            self.output_layer = network.get_layer()
            print('    {:{length}} : {}'.format('feedforward'+str(1), self.output_layer, length=12))
        # 该 else 执行
        else:
            self.forwards=[]
            # 根据作者的参数设定，该for循环执行一次：len(self.feed_forwards)-1 -1 = 1
            for f in range(len(self.feed_forwards)-1 -1):
                #该 if 执行
                if layer_count+1+f in self.em_layers:
                    with tf.name_scope('feedforward'+str(f+1)+'em'):
                        self.em_w.append(tf.Variable( tf.random_normal( [self.feed_forwards[f], self.feed_forwards[f+1]], stddev=self.std, dtype=tf.float32), name='w' ))
                        conv_len1 = self.convs[-1].get_layer().get_shape()[1].value
                        conv_len2 = self.convs[-1].get_layer().get_shape()[2].value
                        if (f==0) and (conv_len1 > 1 or conv_len2>1):
                            self.cluster.append(tf.tile(self.max_idx[-1], [conv_len1*conv_len2]))
                        else:
                            self.cluster.append(self.max_idx[-1])
                        print('      {:{length}} : {}'.format('cluster', self.cluster[-1], length=12))
                        #
                        i = tf.constant(0)
                        w_mean = tf.TensorArray(dtype=tf.float32, size=self.cluster_num)#tf.constant(0.0, shape=tf.TensorShape([]))
                        cond = lambda i,w_mean : i<self.cluster_num
                        x_batch = self.em_w[-1]
                        def func(i,w_mean):
                            mean = tf.reduce_mean(tf.boolean_mask(x_batch, tf.equal(self.cluster[-1],i)), axis=[0])
                            w_mean = w_mean.write(i, mean)
                            return i+1, w_mean
                        i, w_mean = tf.while_loop(cond, func, [i,w_mean])
                        self.max_idx.append(tf.cast(tf.argmax(w_mean.stack(), axis=0), tf.int32))
                        print('      {:{length}} : {}'.format('max_idx', self.max_idx[-1], length=12))
                        #
                        i = tf.constant(0)
                        w_mask_array = tf.TensorArray(dtype=tf.float32, size=self.feed_forwards[f+1])
                        cond2 = lambda i,w_mask_array : i<self.feed_forwards[f+1]
                        def func2(i, w_mask_array):
                            w_mask_array_column = tf.cast(tf.equal(self.cluster[-1], self.max_idx[-1][i]), dtype=tf.float32)
                            w_mask_array = w_mask_array.write(i, w_mask_array_column)
                            return i+1, w_mask_array
                        i, w_mask_array = tf.while_loop(cond2, func2, [i, w_mask_array])
                        w_mask_pack = tf.transpose(w_mask_array.stack())
                        self.w_mask.append(w_mask_pack)
                        self.w_masked.append(tf.multiply(self.em_w[-1], self.w_mask[-1]))

                    # 以上代码的目的均为求 w_masked
                    network  = models.feedforward(input = network, 
                                                  weight_size=[self.feed_forwards[f], self.feed_forwards[f+1]],
                                                  weight=self.w_masked[-1],
                                                  nonlinearity=self.nonlinearity, 
                                                  use_dropout = self.use_dropout, 
                                                  keep_prob = self.keep_probs[len(self.conv)+f], 
                                                  use_batchnorm = self.use_batchnorm,
                                                  std=self.std,
                                                  offset=self.offset,
                                                  scale=self.scale,
                                                  epsilon=self.epsilon, 
                                                  name='forward'+str(f+1))
                    self.forwards.append(network)
                    network = network.get_layer()
                    # 此处 layer_count = 4, +1后为5
                    layer_count += 1
                    print('    {:{length}} : {}'.format('feedforward'+str(f+1), network, length=12))
                else:
                    network  = models.feedforward(input = network, 
                                                  weight_size=[self.feed_forwards[f], self.feed_forwards[f+1]],
                                                  nonlinearity=self.nonlinearity, 
                                                  use_dropout = self.use_dropout, 
                                                  keep_prob = self.keep_probs[len(self.conv)+f], 
                                                  use_batchnorm = self.use_batchnorm,
                                                  std=self.std,
                                                  offset=self.offset,
                                                  scale=self.scale,
                                                  epsilon=self.epsilon, 
                                                  name='forward'+str(f+1))
                    self.forwards.append(network)
                    network = network.get_layer()
                    layer_count += 1
                    print('    {:{length}} : {}'.format('feedforward'+str(f+1), network, length=12))
                #
            network =  models.feedforward(input = network,
                                         weight_size=[self.feed_forwards[-2], self.feed_forwards[-1]],
                                         nonlinearity=None,
                                         use_dropout = False, 
                                         use_batchnorm = False,
                                         std=self.std,
                                         offset=self.offset,
                                         scale=self.scale,
                                         epsilon=self.epsilon, 
                                         name='output')
            self.output = network#.get_layer()
            self.output_layer = network.get_layer()
            print('    {:{length}} : {}'.format('feedforward'+str(f+2), self.output_layer, length=12))
        
    def freeze(self):
        if not self.frozen:
            self.optimizer = tf.train.GradientDescentOptimizer(self.l_rate)
            self.optimize = self.optimizer.minimize(self.err, global_step=self.global_step)
            self.frozen=True
            #print('Frozen.')
            
    def train(self, data, target, profile=False):
        train_feed_dict = {self.x:data}
        train_feed_dict.update({self.y:target})
        train_feed_dict.update({self.keep_probs:self.keep_probs_values})
        if self.init:
            x_batch = self.x_batch.eval(feed_dict=train_feed_dict) #[timeseries length, channel]
            ch = x_batch.shape[0]
            length = x_batch.shape[1]
            cluster = np.random.randint(0, self.cluster_num, ch)
            mu_init = []
            sigma_init = []
            for i in range(model.cluster_num):
                X = x_batch[(cluster==i),:]
                print('X'),
                print(X.shape)
                mu_init = np.mean(X, axis=0)
                print('mu_init'),
                print(mu_init.shape)
                sigma_init = np.cov(X.T)
                print('sigma_init'),
                print(sigma_init.shape)
                self.sess.run(tf.assign(self.mu[i], 
                                                tf.multiply(tf.ones([length], tf.float64), mu_init)))
                self.sess.run(tf.assign(self.sigma[i], 
                                                tf.multiply(tf.ones([length, length], tf.float64), sigma_init)))
            self.init = False
        if profile:
            sums, opt, cost, err, Q = self.sess.run((self.summaries, self.optimize, self.cost, self.err, self.Q), 
                                                    feed_dict=train_feed_dict,
                                                    options=self.options,
                                                    run_metadata=self.run_metadata
                                                   )
            return sums, cost, err, Q
        sums, opt, cost, err, Q = self.sess.run((self.summaries, self.optimize, self.cost, self.err, self.Q), 
                                  feed_dict=train_feed_dict
                                 )
        return sums, cost, err, Q
    
    def test(self, data, target):
        test_feed_dict = {self.x:data}
        test_feed_dict.update({self.y:target})
        keep_probs_values = [1.0 for i in range(len(self.keep_probs_values))]
        test_feed_dict.update({self.keep_probs:keep_probs_values})
        cost, err, Q = self.sess.run((self.cost, self.err, self.Q),
                             feed_dict=test_feed_dict
                            )
        return cost, err, Q
    
    def get_cluster(self, data):
        return self.cluster[0].eval(feed_dict={self.x:data})
    
    def reconstruct(self, data):
        recon_feed_dict = {self.x:data}
        keep_probs_values = [1.0 for i in range(len(self.keep_probs_values))]
        recon_feed_dict.update({self.keep_probs:keep_probs_values})
        return self.sess.run(tf.nn.softmax(self.output), 
                             feed_dict=recon_feed_dict
                            )
    
    def save(self, save_path='./model.ckpt'):
        saved_path = self.saver.save(self.sess, save_path)
        print("Model saved in file: %s"%saved_path)
        
    def load(self, load_path = './model.ckpt'):
        self.saver = tf.train.import_meta_graph(load_path+'.meta')
        self.saver.restore(self.sess, load_path)
        self.init = False
        print("Model restored")
    
    def terminate(self):
        self.sess.close()
        tf.reset_default_graph()       


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        # dict = pickle.load(fo)
        dict = pickle.load(fo, encoding='latin1')
    return dict




# Main


"""
Load dataset
"""

data_path = './CIFAR-10'

one_hot_enc = preprocessing.OneHotEncoder(n_values=10, sparse=False)
# one_hot_enc = preprocessing.OneHotEncoder(categories=[range(10)], sparse=False)

train_data=[]
train_label=[]
for i in range(5):
    tmp = unpickle('./cifar-10-batches-py/data_batch_'+str(i+1))
    # print(tmp)
    train_data.append(tmp["data"])
    train_label.append(tmp["labels"])
train_data = np.concatenate(train_data).reshape([-1, 32,32,3], order='F')
train_label = np.concatenate(train_label)
train_label = one_hot_enc.fit_transform(train_label.reshape([-1,1]))
print("train data: {}, {}".format(train_data.shape, train_label.shape))

test_data = unpickle('./cifar-10-batches-py/test_batch')['data'].reshape([-1,32,32,3], order='F')
test_label = unpickle('./cifar-10-batches-py/test_batch')['labels']
test_label = np.array(test_label)
test_label = one_hot_enc.fit_transform(test_label.reshape([-1,1]))
print("test data: {}, {}" .format(test_label.shape, test_label.shape))


"""
Set Parameters
"""

# CIFAR-10 [32, 32, 3]
#[64,32,32,3] - [50, 1, 112, 112] -- [50, 1, 28, 112] - [50, 1, 7, 112] - [50, 784]

batch_size = 64
inputs = [batch_size, train_data.shape[1], train_data.shape[2], train_data.shape[3]]
conv = [128,128, 128, 128] #conv_base, conv, conv
iter=[0,0,0,0]
pool=['p','p','p','p']
pool_size = [[2,2], [2,2], [2,2], [2,2]]
weight_size = [[3, 3, inputs[-1], conv[0]], [3, 3, conv[0], conv[1]], [3, 3, conv[1], conv[2]], [3, 3, conv[2], conv[3]]]
feed_forwards = [512, 128, 10]
outputs = [batch_size, feed_forwards[-1]]
nonlinearity = tf.nn.relu
err_func = tf.nn.softmax_cross_entropy_with_logits

keep_probs = None
use_dropout = not (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))])   # 为false
use_batchnorm = False

optimizer = tf.train.RMSPropOptimizer
l_rate = 0.0001
std = 0.05

num_epochs = 400
# num_epochs = 2
train_batch_num = train_data.shape[0] / batch_size
#valid_batch_num = valid_data.shape[0] / batch_size
test_batch_num = test_data.shape[0] / batch_size

l_step = 300*train_batch_num
l_decay=0.1



cluster_num = 2
em_layers=[2,3,4,5]
q_param = 1e-6



"""
Set Path
"""

data_path = './CIFAR-10'
# data_save_path = os.path.join('/data2/subin/regularize', data_path[2:])
data_save_path = os.path.join('./data2/subin/regularize', data_path[2:])

# todo 初始化 data_save_path
if not os.path.exists(data_save_path):
    print( 'creating difectory {}'.format(data_save_path))
    os.mkdir(os.path.join(data_save_path))


if not os.path.exists(data_path):
    print( 'creating difectory {}'.format(data_path))
    os.mkdir(os.path.join(data_path))

save_path = os.path.join(data_path, 'grcnn-clvar-freeze')
if not os.path.exists(save_path):
    print( 'creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'gradientdescentopt') if optimizer is None else os.path.join(save_path, str(optimizer).split('.')[-1][:-3])
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))
    
save_path = os.path.join(save_path, 'use_batch_norm') if use_batchnorm else os.path.join(save_path, 'no_batch_norm')
if not os.path.exists(save_path):
    print ('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'no_dropout') if not use_dropout else os.path.join(save_path, 'use_dropout')
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, str(nonlinearity).split(' ')[1]) if nonlinearity is not None else os.path.join(save_path, 'None')
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))


model_name = str(inputs[1:]).replace(', ', '_')+'-['+str(inputs[1]/ pool_size[0][0])+'_'+\
             str(inputs[2]/ pool_size[0][1])+'_'+str(conv[0])+']'+\
             '-['+str(inputs[1]/pool_size[0][0]/pool_size[1][0])+'_'+\
             str(inputs[2]/pool_size[0][1]/pool_size[1][1])+'_'+str(conv[1])+']'+\
             '-['+str(inputs[1]/pool_size[0][0]/pool_size[1][0]/pool_size[2][0])+'_'+\
             str(inputs[2]/pool_size[0][1]/pool_size[1][1]/pool_size[2][0])+'_'+str(conv[2])+']'+\
             '-['+str(inputs[1]/pool_size[0][0]/pool_size[1][0]/pool_size[2][0]/pool_size[3][0])+'_'+\
             str(inputs[2]/pool_size[0][1]/pool_size[1][1]/pool_size[2][0]/pool_size[3][0])+'_'+\
             str(conv[2])+']'+\
             '-em'+str(em_layers).replace(', ','_')[1:-1]+\
             '-cl'+str(cluster_num)+'-q'+str(q_param)+'-iter'+str(iter).replace(', ','_')[1:-1]+\
             '-feedforward'+str(feed_forwards).replace(', ','_')[1:-1]+\
             '-weightsize'+str(np.array(weight_size)[:,0:2]).replace('\n ','_').replace(' ',',')[1:-1]+\
             '-batch'+str(batch_size)+'-l_rate'+str(l_rate)+'-std'+str(std)+\
             '-l_step'+str(l_step/train_batch_num)+'-l_decay'+str(l_decay)

model_name = "history"

if use_dropout:
    model_name = model_name + '-keep'+str(keep_probs).replace(', ','_')[1:-1]
    print(os.path.join(save_path, model_name))
print(os.path.join(save_path, model_name))


if not os.path.exists(os.path.join(save_path, model_name)):
    save_path_origin = save_path
    save_path = os.path.join(save_path, model_name)
    if not os.path.exists(save_path):
        print('creating difectory {}'.format(save_path))
        os.mkdir(os.path.join(save_path))

    save_path = save_path_origin
    print(os.path.join(save_path, model_name))




"""
Train
"""

start_time = time.time()
if 'model' in globals():
    model = globals()["model"]
    model.terminate()
model = grcnn(weight_size=weight_size, 
              pool=pool,
              pool_size=pool_size, 
              inputs=inputs, 
              conv=conv, 
              em_layers=em_layers,
              feed_forwards=feed_forwards, 
              outputs=outputs, 
              iter=iter, 
              cluster_num=cluster_num,
              nonlinearity=nonlinearity, 
              err_func=err_func,
              keep_probs=keep_probs, 
              use_dropout=use_dropout, 
              l_rate=l_rate, 
              optimizer=optimizer,
              q_param=q_param,
              std=std, 
              use_batchnorm=use_batchnorm, 
              offset=1e-10, scale=1, epsilon=1e-10,
              summary_path=os.path.join(save_path,model_name))
print('Done. {:.3f}s taken.'.format(time.time()-start_time))



valid_freq = 10
save_freq = 50
frozen_epoch = 0#350 
test_epoch = [frozen_epoch, 300, num_epochs]

train_history = pd.DataFrame(index=np.arange(0, num_epochs), 
                             columns=['epoch', 'loss', 'err', 'Q', 'timestamp'])
valid_history = pd.DataFrame(index=np.arange(0, num_epochs/valid_freq),
                             columns=['epoch', 'loss', 'err', 'Q', 'timestamp'])
test_history = pd.DataFrame(index=np.arange(0, len(test_epoch)),
                             columns=['epoch', 'train accuracy', 'test accuracy', 'timestamp'])
cluster_history = pd.DataFrame(index=np.arange(0, num_epochs),
                             columns=['epoch']+ [x for x in range(0,model.cluster[0].get_shape()[0].value)]+['timestamp'])
col = ['epoch'] + ['layer'+str(i)+'-w_mask_pass' for i in em_layers] +['timestamp']
param_history = pd.DataFrame(index=np.arange(0, num_epochs/save_freq), columns=col)

train_loss=[]
train_err=[]
train_Q=[]
valid_loss=[]
valid_err=[]
valid_Q=[]
train_accuracy=[]
test_accuracy=[]



def test(test_data, test_labels, batch_size, model, test_batch_num):
    accuracy=0.0
    keep_probs_values = [1.0 for i in range(len(model.keep_probs_values))]
    for batch in iterate_minibatches(inputs=test_data, targets=test_labels, batchsize=batch_size):
        test_in, test_target = batch
        #test_in = test_in[:,np.newaxis,:,np.newaxis]
        #print model.sess.run(tf.reduce_sum(tf.equal(tf.argmax(model.output_layer,1), tf.argmax(model.y, 1))) ,
        #                            feed_dict={model.x:test_in, model.y:test_target})
        accuracy += model.sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.output_layer,1), tf.argmax(model.y, 1)), tf.float32)),
                                feed_dict={model.x:test_in, model.y:test_target, model.keep_probs:keep_probs_values})
    # print'accuracy: {}'.format(accuracy/test_batch_num)
    return accuracy/test_batch_num


seismic_rgb = cm.get_cmap(plt.get_cmap('seismic'))(np.linspace(0.0, 1.0, 100))[:, :3]
print(seismic_rgb.shape)

seismic_gray = np.mean(seismic_rgb,axis=1)
seismic_gray = np.stack([seismic_gray, seismic_gray, seismic_gray], axis=1)
print(seismic_gray.shape)

seismic_gray = colors.ListedColormap(seismic_gray, name='seismic_gray')




# Load
#train_history = pd.read_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-train_history.csv"), index_col=0)
#test_history = pd.read_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-test_history.csv"), index_col=0)
#cluster_history = pd.read_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-cluster_history.csv"), index_col=0)
#param_history = pd.read_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-param_history.csv"), index_col=0)
#
#model.load(os.path.join(data_save_path, model_name, '400.ckpt'))


profile=False
first=True
test_count = 0
Q_conv_count=0

for epoch in range(num_epochs):
    start_time = time.time()
    loss = 0
    err = 0
    Q = 0
    for batch in iterate_minibatches(inputs=train_data, targets=train_label, batchsize=batch_size):
        train_in, train_target = batch
        #train_in = train_in[:,np.newaxis,:,np.newaxis]
        tmp_sum, loss_, err_, Q_ = model.train(train_in, train_target, profile)
        if profile:
            fetched_timeline = timeline.Timeline(model.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('grcnn-timeline_01_step_0.json', 'w') as f:
                f.write(chrome_trace)
        if first:
            model.writer.add_summary(tmp_sum, epoch)
        profile=False
        loss +=loss_
        err += err_
        Q += Q_
    model.writer.add_summary(tmp_sum, epoch)#model.global_step.eval())
    train_loss.append(loss/train_batch_num)
    train_err.append(err/train_batch_num)
    train_Q.append(Q/train_batch_num)
    train_history.loc[epoch] = [epoch+1, train_loss[-1], train_err[-1], train_Q[-1], 
                                time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    cluster_history.loc[epoch] = [epoch]+ model.get_cluster(train_in).tolist() + [time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    if (epoch+1)% save_freq == 0:
        model.save(os.path.join(data_save_path, model_name, str(epoch+1)+'.ckpt'))
        train_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-train_history.csv"))
        #valid_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-valid_history.csv"))
        cluster_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-cluster_history.csv"))
        w_mask_pass = []
        for e in range(len(em_layers)):
            w_mask = model.w_mask[e].eval(feed_dict={model.x:train_in})
            if len(w_mask.shape)==4:
                w_mask = w_mask[0,0,:,:]
            w_mask_pass.append(np.sum(w_mask==1))
        param_history.loc[epoch/save_freq] = [epoch+1] + w_mask_pass +[time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        param_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-param_history.csv"))
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:    {:.6f}".format(train_loss[-1]))
    print("  training err:     {:.6f}".format(train_err[-1]))
    print("  training Q:       {:.6f}".format(train_Q[-1]))
    if epoch>30:
        if (train_Q[-1] - train_Q[-2])/train_Q[-2] <0 :
            Q_conv_count += 1
        else:
            Q_conv_count = 0
        if Q_conv_count>=3:
            if not model.frozen:
                model.freeze()
                frozen_epoch = epoch+1
                test_epoch[0] = frozen_epoch
                print('##### Model frozen at epoch '+str(epoch+1)+'#####')
    if (epoch+1) in test_epoch:
        test_accuracy.append(test(test_data, test_label, batch_size, model, test_batch_num))
        train_accuracy.append(test(train_data, train_label, batch_size, model, train_batch_num))
        test_history.loc[test_count] = [epoch+1, train_accuracy[-1], test_accuracy[-1], time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        test_count +=1
        test_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-test_history.csv"))


print(os.path.join(save_path,model_name))
print('Frozen at '+str(frozen_epoch))




"""
Draw Figures
"""

plt.figure(figsize=(8,6))
ax = plt.subplot(111)
ax.plot(train_history['loss'].tolist(), label='train loss')
#ax.plot( range(valid_freq, len(train_history)+valid_freq, valid_freq), valid_history['loss'].tolist(), label='valid loss', color='Red')
         #marker='o', markersize=5, 
#plt.axis([0, len(train_history), 0, 1.5])
ax.set_xlim([0, len(train_history)])
ax.set_ylim([-0.05,0.2])
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.title('Loss graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('loss', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)
plt.figtext(0.7, 0.7, 'accuracy: '+str(test_accuracy[-1]))
plt.figtext(0.7, 0.6, 'frozen at '+str(frozen_epoch))

plt.savefig(os.path.join(save_path,  model_name, str(len(train_history))+'epochs_tvloss.png'))
print("Figure saved: "+os.path.join(save_path,  model_name, str(len(train_history))+'epochs_tvloss.png'))
#plt.close()



plt.figure(figsize=(8,6))
ax = plt.subplot(111)
ax.plot(train_history['err'].tolist(), label='train err')
#ax.plot( range(valid_freq, len(train_history)+valid_freq, valid_freq), valid_history['err'].tolist(), label='valid err', color='Red')
         #marker='o', markersize=5
#plt.axis([0, len(train_history), 0, 1.5])
ax.set_xlim([0, len(train_history)])
ax.set_ylim([0,0.2])
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.title('Error graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('err', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)
#plt.legend(['train loss'])#,'test loss'])#,'accuracy'])
plt.figtext(0.7, 0.7, 'accuracy: '+str(test_accuracy[-1]))
plt.figtext(0.7, 0.6, 'frozen at '+str(frozen_epoch))

plt.savefig(os.path.join(save_path,  model_name, str(len(train_history))+'epochs_tverr.png'))
print("Figure saved: "+os.path.join(save_path,  model_name, str(len(train_history))+'epochs_tverr.png'))
#plt.close()



plt.figure(figsize=(8,6))
ax = plt.subplot(111)
ax.plot(train_history['Q'].tolist(), label='train Q')
#ax.plot( range(valid_freq, len(train_history)+valid_freq, valid_freq), valid_history['Q'].tolist(), label='valid Q', color='Red')
         #marker='o', markersize=5, 
#plt.axis([0, len(train_history), 0, 1.5])
ax.set_xlim([0, len(train_history)])
#plt.ylim([0,1])
ax.yaxis.set_minor_locator(ticker.MultipleLocator(200))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

plt.title('Q graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('Q', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc=2)
#plt.legend(['train Q'])#,'test Q'])#,'accuracy'])
plt.figtext(0.7, 0.7, 'test accuracy: '+str(test_accuracy[-1]))
plt.figtext(0.7, 0.6, 'train accuracy: '+str(train_accuracy[-1]))
plt.figtext(0.7, 0.5, 'frozen at '+str(frozen_epoch))

plt.savefig(os.path.join(save_path,  model_name, str(len(train_history))+'epochs_tvQ.png'))
print("Figure saved: "+os.path.join(save_path,  model_name, str(len(train_history))+'epochs_tvQ.png'))
#plt.close()




"""
Draw filters
"""

# model.load(os.path.join(data_save_path, model_name, '400.ckpt'))


for em_layer_idx in range(len(em_layers)):
    w_mask=[]
    w_masked=[]
    model_cluster=[]
    count=0
    rand_idx = np.random.randint(50000-batch_size-10)

    for batch in iterate_minibatches(inputs=train_data[rand_idx:,:], targets=train_label[rand_idx:,:], batchsize=batch_size):
        train_in, train_target = batch   
        model_cluster.append(model.cluster[em_layer_idx].eval(feed_dict={model.x:train_in}))
        w_mask.append(model.w_mask[em_layer_idx].eval(feed_dict={model.x:train_in}))
        w_masked.append(model.w_masked[em_layer_idx].eval(feed_dict={model.x:train_in}))
        count += 1
        if count == 10:
            break
    fig = plt.figure(figsize=(17,6))
    for i in range(10):
        ax = fig.add_subplot(2,5,i+1)
        ax.scatter(range(len(model_cluster[0])), model_cluster[i])
    #plt.scatter(range(len(model_cluster[0])), model_cluster[1], c='blue')
    plt.savefig(os.path.join(save_path,model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-cluster_ex.png'))

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_mask[i][0,0,:,:], cmap='gray')
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_mask_ex2.pdf'))
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_mask[i][0,0,:,:], cmap='gray')
    #plt.xlabel('weight mask', fontsize=15)
    plt.colorbar(fraction=0.045)
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_mask_ex4.pdf'))


    cluster_ = model_cluster[i]
    w_mask_ = w_mask[i][0,0,:,:]
    print(cluster_.shape, w_mask_.shape)

    w_mask_reordered_cl=[]
    for i in range(cluster_num):
        w_mask_reordered_cl.append(w_mask_[cluster_==i,:])
    w_mask_reordered = np.concatenate(w_mask_reordered_cl)
    print(w_mask_reordered.shape)

    w_mask_reordered_cl=[]
    for i in range(cluster_num):
        w_mask_reordered_cl.append(w_mask_reordered[:,w_mask_[cluster_==i,:][0]!=0])
    w_mask_reordered = np.concatenate(w_mask_reordered_cl, axis=1)
    print(w_mask_reordered.shape)

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_mask_reordered, cmap='gray')
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_mask_reordered_ex2.pdf'))
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_mask_reordered, cmap='gray')
    #plt.xlabel('masked weight', fontsize=15)
    plt.colorbar(fraction=0.045)
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_mask_reordered_ex4.pdf'))
    #print(os.path.join(save_path, model_name, 'w_masked_ex2.pdf'))
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_masked[i][0,0,:,:], cmap=seismic_gray, clim=(-0.3,0.3))#cmap='seismic', clim=(-0.4, 0.4))#cmap='Greys', clim=(-0.301414,0.528816))
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_masked_ex2.pdf'))
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_masked[i][0,0,:,:], cmap=seismic_gray, clim=(-0.3,0.3))#cmap='seismic', clim=(-0.4, 0.4))#cmap='Greys', clim=(-0.301414,0.528816))
    #plt.xlabel('masked weight', fontsize=15)
    plt.colorbar(fraction=0.045)
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_masked_ex4.pdf'))
    #print(os.path.join(save_path, model_name, 'w_masked_ex2.pdf'))
    plt.close()


    w_masked_reordered_cl=[]
    for i in range(cluster_num):
        w_masked_reordered_cl.append(w_mask_[cluster_==i,:])
    w_masked_reordered = np.concatenate(w_masked_reordered_cl)
    print(w_masked_reordered.shape)

    w_masked_reordered_cl=[]
    for i in range(cluster_num):
        w_masked_reordered_cl.append(w_masked_reordered[:,w_mask_[cluster_==i,:][0]!=0])
    w_masked_reordered = np.concatenate(w_masked_reordered_cl, axis=1)
    print(w_masked_reordered.shape)

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_masked_reordered, cmap=seismic_gray, clim=(-0.3,0.3))
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_masked_reordered_ex2.pdf'))
    plt.close()

    fig = plt.figure(figsize=(6,6))
    plt.imshow(w_masked_reordered, cmap=seismic_gray, clim=(-0.3,0.3))
    #plt.xlabel('masked weight', fontsize=15)
    plt.colorbar(fraction=0.045)
    plt.savefig(os.path.join(save_path, model_name, str(epoch+1)+'-'+str(em_layers[em_layer_idx])+'layer-w_masked_reordered_ex4.pdf'))
    #print(os.path.join(save_path, model_name, 'w_masked_ex2.pdf'))
    plt.close()




