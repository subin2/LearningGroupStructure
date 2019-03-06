#!/usr/bin/env python
# encoding: utf-8
'''
@author: Pan
@software: PyCharm
@file: densenet_encoder_Pan.py
@time: 2019/3/4 15:08
@desc:
'''
from encoder_decoder_model import cnn_base_model
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class DenseDecoder(cnn_base_model.CNNBaseModel):
    def __init__(self,input_tensor,phase, block_num, growth_rate,class_num,epsilon=1e-4,dropout_rate=0.2):
        self._phase = True if phase == 'train' else False
        self._block_num = block_num
        self._growth_rate = growth_rate
        self._class_num = class_num
        self._epsilon = epsilon
        self._dropout_rate = dropout_rate
        self.model = self.Dense_net(input_tensor)

    def bottleneck_layer(self, x, scope):
        print(x)
        with tf.name_scope(scope):

            x = self.layer_bn(x,is_training=self._phase,name=scope+'_batch1')
            x = self.relu(x,scope+'_relu1')

            x = self.conv2d(input_data=x, out_channel=4*self._growth_rate,
                                kernel_size=1, use_bias=False, name=scope+'conv1')
            x = self.dropout(x,keep_prob=self._dropout_rate)
            x = self.layer_bn(x, is_training=self._phase,name=scope+'_batch2')
            x = self.relu(x,scope+'_relu2')
            x = self.conv2d(input_data=x, out_channel=self._growth_rate,
                            kernel_size=3, use_bias=False, name=scope + 'conv2')
            x = self.dropout(x, keep_prob=self._dropout_rate)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = self.layer_bn(x, is_training=self._phase, name=scope + '_batch1')
            x = self.relu(x, scope + '_relu1')
            x = self.conv2d(input_data=x, out_channel=self._growth_rate,
                            kernel_size=3, use_bias=False, name=scope + 'conv1')
            x = self.dropout(x, keep_prob=self._dropout_rate)
            x = self.avg_pooling(x,kernel_size=2,stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = self.concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

                x = self.concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = self.conv2d(input_data=input_x, out_channel=2*self._growth_rate,
                        kernel_size=7, use_bias=False, stride=2,name='conv1')

        for i in range(self._block_num) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))


        # 100 Layer
        x = self.layer_bn(x, is_training=self._phase, name='linear_batch1')
        x = self.relu(x, 'linear_relu1')
        x = self.global_avg_pooling(x)
        x = flatten(x)
        x = self.linear(x,class_num=self._class_num)


        # x = tf.reshape(x, [-1, 10])
        return x