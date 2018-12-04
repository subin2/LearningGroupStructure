#!/usr/bin/env python
# coding: utf-8


import os
import csv
import time
import random
import cPickle
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
import models


def gmm_pdf_log(x, mu=[], sigma=[], sess=None):
    #check shape
    if type(mu) is list:
        multi=True
        if len(mu) != len(sigma):
            raise ValueError('mu and sigma number not matched.')
    if multi:
        x = [tf.reshape(t, [-1]) for t in tf.split(axis=0,
                                                   num_or_size_splits=x.get_shape()[0].value,
                                                   value=x)]
        dim=mu[-1].get_shape()[-1].value #100
    else:
        if len(x.get_shape()) != len(mu.get_shape()) or x.get_shape()[0].value!=mu.get_shape()[0].value:
            raise ValueError('x shape error')
        dim=mu.get_shape()[-1].value

    # return pdf. Note that it is not log(pdf).
    if multi:
        output=[]
        for i in range(len(mu)): #3
            flag=False
            tmp_sigma = tf.abs(sigma[i])
            tmp_sigma = tf.matrix_diag(tf.matrix_diag_part(tmp_sigma))
            tmp_mu = mu[i]
            #
            log_det = tf.reduce_sum(tf.log(tf.matrix_diag_part(tmp_sigma) + 1e-40))
            inv = tf.clip_by_value(tf.matrix_diag(tf.div(tf.cast(1.0, dtype=tf.float64),
                                                                         tf.matrix_diag_part(tmp_sigma))),
                                           clip_value_min=-1e30, clip_value_max=1e30)
            tmp1 = -(dim*0.5*tf.log(tf.cast(2*np.pi, dtype=tf.float64))) - (0.5*log_det)
            tmp2_1 = tf.matmul(tf.matmul(tf.reshape(tf.cast(x[i], dtype=tf.float64) - tmp_mu,[1,-1]),
                                         inv),#tf.matrix_inverse(sigma)),
                               tf.reshape(tf.cast(x[i], dtype=tf.float64)-tmp_mu,[-1,1]))
            tmp2 = -0.5*tmp2_1
            pdf = tf.exp(tmp1+tmp2) #remove log
            output.append(pdf)
        output = tf.reshape(tf.concat(values=output, axis=0),[-1])
        return output
    else:
        #det = tf.matrix_determinant(self.sigma[i])
        det = tf.cond(tf.equal(tf.matrix_determinant(self.sigma),0), 
                    lambda:  tf.constant(1e-30),
                    lambda: tf.matrix_determinant(self.sigma))
        tmp1_1 = tf.matmul(tf.pow((2*np.pi),self.dim), det)
        tmp1 = tf.pow(tmp1_1, -1.0/2)
        tmp2 = tf.exp(-tf.matmul(tf.matmul(tf.reshape(x[i]-self.mu,[1,-1]),
                                            tf.matrix_inverse(self.sigma)),
                                tf.reshape(x[i]-self.mu,[-1,1])
                                )/2)
        output = tf.matmul(tmp1,tmp2)
        return output



class mlp_em(object):
    def __init__(self, inputs, feed_forwards, em_layers=[1], cluster_num=4,init_cluster=None, reg=False,
                 nonlinearity=None, err_func=None, keep_probs=None, use_dropout=False, use_batchnorm=False, 
                 l_rate=0.001, l_step=1e15, l_decay=1.0, q_param=1e-3,
                 std=0.05, offset=1e-10, scale=1, epsilon=1e-10, summary_path='./'):
        print('Start constructing...')
        self.init = True
        self.frozen=False
        self.reg = reg
        self.inputs = inputs
        self.feed_forwards = feed_forwards
        self.weight_size = [inputs[-1]] + feed_forwards
        self.em_layers=em_layers
        self.cluster_num = cluster_num
        self.init_cluster = init_cluster
        self.nonlinearity = nonlinearity
        self.err_func = err_func
        self.keep_probs_values = keep_probs
        self.use_batchnorm = use_batchnorm
        self.use_dropout=use_dropout
        if keep_probs == None:
            self.keep_probs_values = [1.0 for i in range(len(feed_forwards)-1)]
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
        self.cost = self.err - self.q_param*tf.cast(self.Q, tf.float32)
        #self.optimizer = tf.train.AdamOptimizer(self.l_rate).minimize(self.cost, global_step=global_step)
        self.optimizer = tf.train.GradientDescentOptimizer(self.l_rate)
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
        network = self.x
        self.layers = []
        layer_count=0
        ### convs before em
        for i in range(layer_count, self.em_layers[0]-1):
            layer = models.feedforward(input = network,
                                       weight_size=[self.weight_size[i], self.weight_size[i+1]],
                                         nonlinearity=self.nonlinearity,
                                         use_dropout = self.use_dropout, 
                                         keep_prob=self.keep_probs[i], 
                                         use_batchnorm = self.use_batchnorm,
                                         std=self.std,
                                         offset=self.offset,
                                         scale=self.scale,
                                         epsilon=self.epsilon, 
                                         name='layer'+str(i+1))
            self.layers.append(layer)
            network = layer.get_layer()
            print('    {:{length}} : {}'.format('layer'+str(i+1), network, length=12))
            layer_count += 1
        
        ### em
        length = self.weight_size[self.em_layers[0]]
        mu_init_value = np.zeros([self.cluster_num, length])
        sigma_init_value = np.zeros([self.cluster_num, length, length])
        pi_init_value = np.ones([cluster_num]) / self.cluster_num
        
        self.mu = [tf.Variable(tf.random_normal([length], dtype=tf.float64), name='mu'+str(t)) for t in range(self.cluster_num)]
        self.sigma = [tf.Variable(tf.random_normal([length,length], dtype=tf.float64), name='sigma'+str(t)) for t in range(self.cluster_num)]
        self.pi = tf.Variable(tf.multiply(tf.ones([1, self.cluster_num], tf.float64), pi_init_value),
                            trainable=True,name='pi')
        # force the sum of elements of pi vector to be 1.
        self.pi_normed = tf.div(tf.maximum(self.pi, 0.0), tf.reduce_sum(tf.maximum(self.pi, 0.0))) 
        self.mu_summary = tf.summary.image("mu", tf.reshape(tf.cast(tf.stack(self.mu),tf.float32), [1,self.cluster_num, length, 1]))
        self.sigma1_summary = tf.summary.image("sigma1", tf.reshape(tf.cast(self.sigma[0], tf.float32), [1,length,length,1]))
        self.pi_summary = tf.summary.image("pi", tf.reshape(tf.cast(self.pi_normed, tf.float32),[1,1,-1,1]))
        #
        self.em_w=[]
        self.w_mask=[]
        self.w_masked=[]
        self.cluster = []
        self.max_idx = []
        for em in range(len(self.em_layers)):
            with tf.name_scope('layer'+str(layer_count+1)+'em'):
                self.em_w.append(tf.Variable( tf.random_normal( [self.weight_size[layer_count],self.weight_size[layer_count+1]], stddev=self.std, dtype=tf.float32), name='w' ))                
                if em == 0:
                    gamma_elem = []
                    Q_elem = []
                    self.x_batch = tf.stop_gradient(self.em_w[-1])
                    for w in range(self.weight_size[layer_count]):
                        x_pdf = gmm_pdf_log(mu=self.mu, 
                                                        sigma=self.sigma, 
                                                        x=tf.reshape(tf.tile(self.x_batch[w,:],[self.cluster_num]),[self.cluster_num,-1]), #[3, 100]
                                                       sess=self.sess)                
                        pi_pdf = tf.multiply(self.pi_normed, x_pdf)
                        gamma_tmp = tf.reshape(tf.div(pi_pdf,
                                                      tf.maximum(tf.reduce_sum(pi_pdf),1e-30)),
                                               [-1])
                        gamma_tmp = tf.stop_gradient(gamma_tmp) # fix the value. do not calculate the gradient of this term.
                        gamma_elem.append(gamma_tmp)
                        tmp = tf.reduce_sum(tf.multiply(gamma_tmp, 
                                                                    tf.log(pi_pdf+1e-30)))
                        Q_elem.append(tmp)
                    self.Q = tf.reduce_sum(Q_elem)
                    self.Q_summary = tf.summary.scalar("Q", self.Q)
                    self.gamma = tf.stack(gamma_elem)
                    self.cluster.append(tf.cast(tf.argmax(self.gamma, axis=1), dtype=tf.int32))
                    print('      {:{length}} : {}'.format('cluster', self.cluster[-1], length=12))

                    i = tf.constant(0)
                    w_mean = tf.TensorArray(dtype=tf.float32, size=self.cluster_num)#tf.constant(0.0, shape=tf.TensorShape([]))
                    cond = lambda i,w_mean : i<self.cluster_num
                    x_batch = tf.stop_gradient(self.em_w[-1])
                    def func(i,w_mean):
                        mean = tf.reduce_mean(tf.boolean_mask(x_batch, tf.equal(self.cluster[-1],i)), axis=[0])
                        w_mean = w_mean.write(i, mean)
                        return i+1, w_mean
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
                    x_batch = tf.stop_gradient(self.em_w[-1])
                    def func(i,w_mean):
                        mean = tf.reduce_mean(tf.boolean_mask(x_batch, tf.equal(self.cluster[-1],i)), axis=[0])
                        w_mean = w_mean.write(i, mean)
                        return i+1, w_mean
                    i, w_mean_ = tf.while_loop(cond, func, [i,w_mean])
                    self.max_idx.append(tf.cast(tf.argmax(w_mean_.stack(), axis=0), tf.int32))
                    print('      {:{length}} : {}'.format('max_idx', self.max_idx[-1], length=12))
                #
                i = tf.constant(0)
                w_mask_array = tf.TensorArray(dtype=tf.float32, size=self.weight_size[layer_count+1])
                cond2 = lambda i,w_mask_array : i<self.weight_size[layer_count+1]# self.weight_size[1][-1]
                def func2(i, w_mask_array):
                    w_mask_array_column = tf.cast(tf.equal(self.cluster[-1], self.max_idx[-1][i]), dtype=tf.float32)
                    w_mask_array = w_mask_array.write(i, w_mask_array_column)
                    return i+1, w_mask_array
                i, w_mask_array = tf.while_loop(cond2, func2, [i, w_mask_array])
                w_mask_pack = tf.transpose(w_mask_array.stack())#.pack())
                #self.w_mask = tf.expand_dims(tf.stack( [w_mask_pack for i in range(self.weight_size[self.em_layer])] ), 0)
                self.w_mask.append(w_mask_pack)
                #print('    {:{length}} : {}'.format('w_mask', self.w_mask, length=12))
                self.w_masked.append(tf.multiply(self.em_w[-1], self.w_mask[-1]))                
            # end if-else
            layer = models.feedforward(input = network,
                                       weight_size=[self.weight_size[layer_count], self.weight_size[layer_count+1]],
                                       nonlinearity=self.nonlinearity,
                                       use_dropout = self.use_dropout, 
                                       keep_prob=self.keep_probs[layer_count], 
                                       use_batchnorm = self.use_batchnorm,
                                       std=self.std,
                                       offset=self.offset,
                                       scale=self.scale,
                                       epsilon=self.epsilon, 
                                       name='layer'+str(layer_count+1))
            self.layers.append(layer)
            network = layer.get_layer()
            print('    {:{length}} : {}'.format('layer'+str(layer_count+1), network, length=12))
            layer_count += 1
            if layer_count>=len(self.feed_forwards):
                break
        # end for
        #
        for i in range(len(self.feed_forwards)-layer_count-1):
            network  = models.feedforward(input = network, 
                                          weight_size=[self.weight_size[layer_count], self.weight_size[layer_count+1]],
                                          nonlinearity=self.nonlinearity, 
                                          use_dropout = self.use_dropout, 
                                          keep_prob = self.keep_probs[layer_count], 
                                          use_batchnorm = self.use_batchnorm,
                                          std=self.std,
                                          offset=self.offset,
                                          scale=self.scale,
                                          epsilon=self.epsilon, 
                                          name='layer'+str(layer_count+1))
            self.layers.append(network)
            network = network.get_layer()
            print('    {:{length}} : {}'.format('layer'+str(layer_count+1), network, length=12))
            layer_count += 1
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
        self.layers.append(network)#.get_layer()
        network = network.get_layer()
        self.output_layer = network
        print('    {:{length}} : {}'.format('layer'+str(len(self.feed_forwards)), network, length=12))
    
    def freeze(self):
        if not self.frozen:
            self.optimizer = tf.train.GradientDescentOptimizer(self.l_rate)
            self.optimize = self.optimizer.minimize(self.err, global_step=self.global_step)
            self.frozen=True
            print('Frozen.')            
        
    def train(self, data, target, profile=False):
        train_feed_dict = {self.x:data}
        train_feed_dict.update({self.y:target})
        train_feed_dict.update({self.keep_probs:self.keep_probs_values})
        if self.init:
            x_batch = self.x_batch.eval(feed_dict=train_feed_dict)
            cluster = np.random.randint(0, self.cluster_num, x_batch.shape[0]) if self.init_cluster is None else self.init_cluster
            length = x_batch.shape[1]
            mu_init = []
            sigma_init = []
            for i in range(model.cluster_num):
                X = x_batch[(cluster==i), :]
                mu_init = np.mean(X, axis=0)
                sigma_init = np.cov(X.T)
                self.sess.run(tf.assign(self.mu[i], 
                                                tf.multiply(tf.ones([length], tf.float64), mu_init)))
                self.sess.run(tf.assign(self.sigma[i], 
                                                tf.multiply(tf.ones([length, length], tf.float64), sigma_init)))
            self.init = False
        if profile:
            opt, cost, err, Q = self.sess.run((self.optimize, self.cost, self.err, self.Q), 
                                      feed_dict=train_feed_dict,
                                      options=self.options,
                                      run_metadata=self.run_metadata
                                     )
            return cost, err, Q
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




# Main


"""
 Load dataset
"""

data_path = './MNIST_data'
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print 'train: {}'.format(mnist.train.num_examples)
print 'validation: {}'.format(mnist.validation.num_examples)
print 'test: {}'.format(mnist.test.num_examples)

train_data = mnist.train.images#[:,np.newaxis,:] # Returns np.array
train_data_resized = tf.reshape(tf.image.resize_images(train_data.reshape([train_data.shape[0], 28,28,1]), size=[20, 20]), [train_data.shape[0], -1])
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
print 'train: {}, {}'.format(train_data.shape, train_data_resized.shape, train_labels.shape)

valid_data = mnist.validation.images#[:,np.newaxis,:] # Returns np.array
valid_data_resized = tf.reshape(tf.image.resize_images(valid_data.reshape([valid_data.shape[0], 28,28,1]), size=[20, 20]), [valid_data.shape[0], -1])
valid_labels = np.asarray(mnist.validation.labels, dtype=np.int32)
print 'validation: {}, {}'.format(valid_data.shape, valid_data_resized.shape, valid_labels.shape)

test_data = mnist.test.images#[:,np.newaxis,:] # Returns np.array
test_data_resized = tf.reshape(tf.image.resize_images(test_data.reshape([test_data.shape[0], 28,28,1]), size=[20, 20]), [test_data.shape[0], -1])
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
print 'test: {}, {}'.format(test_data.shape, test_data_resized.shape, test_labels.shape)

sess = tf.Session()
with sess.as_default():
    train_data_resized = train_data_resized.eval()
    valid_data_resized = valid_data_resized.eval()
    test_data_resized = test_data_resized.eval()
sess.close()



"""
Set parameters
"""

inputs=[None, 400] #resize 28*28 -> 20*20 -> 400
batch_size=64
feed_forwards = [200, 200, 10]
nonlinearity = tf.nn.relu #tf.nn.elu
err_func = tf.nn.softmax_cross_entropy_with_logits
keep_probs=[1, 1]
use_dropout=True if (np.array(keep_probs)!=1.0).any() and len(keep_probs)>0 else False
use_batchnorm=False

l_rate = 0.01
l_step = 1e30
l_decay=0.9
std=0.05
offset=1e-10
scale=1
epsilon=1e-10

num_epochs=500
train_batch_num = train_data.shape[0] / batch_size
valid_batch_num = valid_data.shape[0] / batch_size
test_batch_num = test_data.shape[0] / batch_size

## em parameters
em_layers = [1,2]
cluster_num = 3
q_param = 1e-5




"""
Set Path
"""

save_path = os.path.join(data_path, 'MLP_REG_freeze')
if not os.path.exists(save_path):
    print 'creating difectory {}'.format(save_path)
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'use_batch_norm') if use_batchnorm else os.path.join(save_path, 'no_batch_norm')
if not os.path.exists(save_path):
    print 'creating difectory {}'.format(save_path)
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'no_dropout') if (keep_probs == None or keep_probs == [1.0 for i in range(len(keep_probs))]) else os.path.join(save_path, 'use_dropout')
if not os.path.exists(save_path):
    print 'creating difectory {}'.format(save_path)
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, str(nonlinearity).split(' ')[1]) if nonlinearity is not None else os.path.join(save_path, 'None')
if not os.path.exists(save_path):
    print 'creating difectory {}'.format(save_path)
    os.mkdir(os.path.join(save_path))

model_name =str(inputs[1])+'-'+str(feed_forwards).replace(', ','-')[1:-1]+'-em_layer'+str(em_layers)+'_'+str(cluster_num)+'cl'+'-qparam'+str(q_param)+                        '-batch'+str(batch_size)+'-lrate'+str(l_rate)+'-ldecay'+str(l_decay)+'-std'+str(std)+'-'+str(num_epochs)+'epoch'
                    #+'-lstep'+str(l_step)
if use_dropout:
    model_name = model_name+'-keep'+str(keep_probs)[1:-1].replace(', ','_')
print model_name

print(os.path.exists(os.path.join(save_path,model_name)))


if not os.path.exists(os.path.join(save_path, model_name)):
    os.mkdir( os.path.join(save_path, model_name) )
    print os.path.join(save_path, model_name)




"""
Train
"""

start_time = time.time()
if 'model' in globals():
    model.terminate()
model = mlp_em(inputs=inputs,
               feed_forwards=feed_forwards, 
               em_layers=em_layers,
               cluster_num=cluster_num,
               nonlinearity=nonlinearity, 
               err_func=err_func, 
               keep_probs=keep_probs, 
               use_dropout=use_dropout, 
               use_batchnorm=use_batchnorm, 
               l_rate=l_rate, 
               l_step=l_step, 
               l_decay=l_decay, 
               q_param=q_param,
               reg=True,
               std=std, offset=offset, scale=scale, epsilon=epsilon,
               summary_path=os.path.join(save_path,model_name))
print('{:.3f}s taken.'.format(time.time()-start_time))


valid_freq = 10
save_freq = 50
frozen_epoch = 0#350 
test_epoch = [frozen_epoch, num_epochs]

train_history = pd.DataFrame(index=np.arange(0, num_epochs), 
                             columns=['epoch', 'loss', 'err', 'Q', 'timestamp'])
valid_history = pd.DataFrame(index=np.arange(0, num_epochs/valid_freq),
                             columns=['epoch', 'loss', 'err', 'Q', 'timestamp'])
test_history = pd.DataFrame(index=np.arange(0, len(test_epoch)),
                             columns=['epoch', 'accuracy', 'timestamp'])
cluster_history = pd.DataFrame(index=np.arange(0, num_epochs),
                             columns=['epoch']+ range(0,model.cluster[0].get_shape()[0].value)+['timestamp'])
col = ['epoch'] + ['layer'+str(i)+'-w_mask_pass' for i in em_layers] +['timestamp']
param_history = pd.DataFrame(index=np.arange(0, num_epochs/save_freq), columns=col)

train_loss=[]
train_err=[]
train_Q=[]
valid_loss=[]
valid_err=[]
valid_Q=[]
test_accuracy=[]



# Load
#train_history = pd.read_csv(os.path.join(save_path,model_name,str(num_epochs)+"epochs-train_history.csv"), index_col=0)
#valid_history = pd.read_csv(os.path.join(save_path,model_name,str(num_epochs)+"epochs-valid_history.csv"), index_col=0)
##test_history = pd.read_csv(os.path.join(save_path,model_name,str(num_epochs)+"epochs-test_history.csv"), index_col=0)
#cluster_history = pd.read_csv(os.path.join(save_path,model_name,str(num_epochs)+"epochs-cluster_history.csv"), index_col=0)
#param_history = pd.read_csv(os.path.join(save_path,model_name,str(num_epochs)+"epochs-param_history.csv"), index_col=0)
#
#model.load(os.path.join(save_path, model_name, '50.ckpt'))



def test(test_data, test_labels, batch_size, model, test_batch_num):
    accuracy=0.0
    keep_probs_values = [1.0 for i in range(len(model.keep_probs_values))]
    for batch in utils.iterate_minibatches(inputs=test_data, targets=test_labels, batchsize=batch_size):
        test_in, test_target = batch
        #test_in = test_in[:,np.newaxis,:,np.newaxis]
        #print model.sess.run(tf.reduce_sum(tf.equal(tf.argmax(model.output_layer,1), tf.argmax(model.y, 1))) ,
        #                            feed_dict={model.x:test_in, model.y:test_target})
        accuracy += model.sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.output_layer,1), tf.argmax(model.y, 1)), tf.float32)),
                                feed_dict={model.x:test_in, model.y:test_target, model.keep_probs:keep_probs_values})
    print'accuracy: {}'.format(accuracy/test_batch_num)
    return accuracy/test_batch_num

seismic_rgb = cm.get_cmap(plt.get_cmap('seismic'))(np.linspace(0.0, 1.0, 100))[:, :3]
print(seismic_rgb.shape)

seismic_gray = np.mean(seismic_rgb,axis=1)
seismic_gray = np.stack([seismic_gray, seismic_gray, seismic_gray], axis=1)
print(seismic_gray.shape)

seismic_gray = colors.ListedColormap(seismic_gray, name='seismic_gray')




profile=False
test_count = 0
first = True
Q_conv_count=0
frozen_epoch=0

for epoch in range(num_epochs):
    start_time = time.time()
    loss = 0
    err = 0
    Q = 0
    for batch in utils.iterate_minibatches(inputs=train_data_resized, targets=train_labels, batchsize=batch_size):
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
        #print loss
    train_loss.append(loss/train_batch_num)
    train_err.append(err/train_batch_num)
    train_Q.append(Q/train_batch_num)
    model.writer.add_summary(tmp_sum, epoch)#model.global_step.eval())
    train_history.loc[epoch] = [epoch+1, train_loss[-1], train_err[-1], train_Q[-1], 
                                time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    cluster_history.loc[epoch] = [epoch]+ model.get_cluster(train_in).tolist() + [time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    if (epoch+1)%valid_freq == 0:
        loss = 0
        err = 0
        Q = 0
        for batch in utils.iterate_minibatches(inputs=valid_data_resized, targets=valid_labels, batchsize=batch_size):
            valid_in, valid_target = batch
            #valid_in = valid_in[:,np.newaxis,:,np.newaxis]
            loss_, err_, Q_ = model.test(valid_in, valid_target)            
            loss +=loss_
            err += err_
            Q += Q_
        valid_loss.append(loss/valid_batch_num)
        valid_err.append(err/valid_batch_num)
        valid_Q.append(Q/valid_batch_num)
        valid_history.loc[epoch/valid_freq] = [epoch+1, loss/valid_batch_num, err/valid_batch_num, Q/valid_batch_num, 
                                               time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    #print("  training loss:    {:.6f}".format(train_loss[-1]))
    print("  training err:     {:.6f}".format(train_err[-1]))
    print("  training Q:       {:.6f}".format(train_Q[-1]))
    if (epoch+1)%valid_freq==0:
        #print("  validation loss: {:.6f}".format(loss/valid_batch_num))        
        print("  validation err:  {:.6f}".format(valid_err[-1]))
        print("  validation Q:    {:.6f}".format(valid_Q[-1]))
    if (epoch+1)% save_freq == 0:
        model.save(os.path.join(save_path, model_name, str(epoch+1)+'.ckpt'))
        train_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-train_history.csv"))
        valid_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-valid_history.csv"))
        cluster_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-cluster_history.csv"))
        param_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-param_history.csv"))
        w_mask_pass = []
        for e in range(len(em_layers)):
            w_mask = model.w_mask[e].eval(feed_dict={model.x:train_in})
            if len(w_mask.shape)==4:
                w_mask = w_mask[0,0,:,:]
            w_mask_pass.append(np.sum(w_mask==1))
        param_history.loc[epoch/save_freq] = [epoch+1] + w_mask_pass +[time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        param_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-param_history.csv"))
    if epoch>60:
        if (train_Q[-1] - train_Q[-2])<0 :
            #if (train_Q[-1] - train_Q[-2])/train_Q[-2] <0 :
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
        test_accuracy.append(test(test_data_resized, test_labels, batch_size, model, test_batch_num))
        test_history.loc[test_count] = [epoch+1, test_accuracy[-1], time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        test_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-test_history.csv"))
        test_count +=1
model.save(os.path.join(save_path, model_name, str(epoch+1)+'.ckpt'))
train_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-train_history.csv"))
valid_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-valid_history.csv"))
test_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-test_history.csv"))
cluster_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-cluster_history.csv"))
param_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-param_history.csv"))

print("model frozen at: "+str(frozen_epoch))




"""
Draw Figures
"""

plt.figure(figsize=(8,5))
ax = plt.subplot(111)

ax.plot(train_history['loss'].tolist(), label='train loss')
ax.plot(valid_history['epoch'], valid_history['loss'], label='valid loss', color='Red')#, marker='o'
ax.set_xlim(0, len(train_history))
#ax.set_ylim(-1, 1)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_tick_params(which='major', right = 'on')
ax.yaxis.set_tick_params(which='minor', right = 'on')

plt.title('Loss graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('loss', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)

plt.savefig(os.path.join(save_path, model_name, str(len(train_history))+'epochs_tvloss.png'))
print("Figure saved: "+os.path.join(save_path, model_name, str(len(train_history))+'epochs_tvloss.png'))
#plt.savefig(os.path.join(save_path, path, model_name, str(len(train_history))+'epochs_tvloss.png'))
#print os.path.join(save_path, path, model_name, str(len(train_history))+'epochs_tvloss.png')
#plt.close()


plt.figure(figsize=(8,5))
ax = plt.subplot(111)

ax.plot(train_history['err'].tolist(), label='train err')
ax.plot(valid_history['epoch'], valid_history['err'], label='valid err', color='Red')#, marker='o'
ax.set_xlim(0, len(train_history))
ax.set_ylim(0, 1)

ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_tick_params(which='major', right = 'on')
ax.yaxis.set_tick_params(which='minor', right = 'on')
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)
#plt.legend(['train loss'])#,'test loss'])#,'accuracy'])
plt.title('Err graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('Err', fontsize=13)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)

plt.savefig(os.path.join(save_path, model_name, str(len(train_history))+'epochs_tverr.png'))
print("Figure saved: "+ os.path.join(save_path, model_name, str(len(train_history))+'epochs_tverr.png'))
#plt.savefig(os.path.join(save_path, path, model_name, str(len(train_history))+'epochs_tvloss.png'))
#print os.path.join(save_path, path, model_name, str(len(train_history))+'epochs_tvloss.png')
#plt.close()


plt.figure(figsize=(8,5))
ax = plt.subplot(111)

ax.plot(train_history['Q'].tolist(), label='train Q')
ax.plot(valid_history['epoch'], valid_history['Q'], label='valid Q', color='Red')#, marker='o'
#plt.axis([0, len(train_history), 0, 2])
#ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#ax.yaxis.set_tick_params(which='major', right = 'on')
#ax.yaxis.set_tick_params(which='minor', right = 'on')
plt.title('Q graph', fontsize=15)
plt.xlabel('epoch', fontsize=13)
plt.ylabel('Q', fontsize=13)

plt.figtext(0.7, 0.7, 'accuracy: '+str(test_accuracy[-1]))
plt.figtext(0.7, 0.6, 'frozen at '+str(frozen_epoch))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.7, box.height])
plt.legend(fontsize=12, bbox_to_anchor=(1.05,1),loc=2)

plt.savefig(os.path.join(save_path, model_name, str(len(train_history))+'epochs_tvQ.png'))
print("Figure saved: "+os.path.join(save_path, model_name, str(len(train_history))+'epochs_tvQ.png'))
#plt.close()


