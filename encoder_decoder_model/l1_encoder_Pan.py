# encoding: utf-8
'''
@author: Pan
@software: PyCharm
@file: l1_encoder_Pan.py
@time: 2018/12/18 10:39
@desc:
    构建CNN-l1模型
'''
import tensorflow as tf
from collections import OrderedDict
from encoder_decoder_model import cnn_base_model
import numpy as np
import os


class L1Encoder(cnn_base_model.CNNBaseModel):
    def __init__(self,phase,input_tensor,feed_forwards,optimizer,l_rate,l_step,l_decay,use_bn,keep_probs,std=0.01,regular_scale=0.0,use_bias = True):
        super(L1Encoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()
        self._use_bn = use_bn
        self._use_bias = use_bias
        self.feed_forwards = feed_forwards
        self.keep_probs_values = keep_probs
        if keep_probs == None:
            self.keep_probs_values = [1.0 for i in range(4+len(feed_forwards)-1)]
        self.keep_probs = tf.placeholder(tf.float32, [len(self.keep_probs_values)], name='keep_probs')

        self.global_step = tf.Variable(0, trainable=False)
        self.l_rate = tf.train.exponential_decay(l_rate, self.global_step, l_step, l_decay, staircase=True)
        self.std = std

        self.regularizer = tf.contrib.layers.l1_regularizer(regular_scale, scope=None)

        self.data = tf.placeholder(tf.float32, shape=input_tensor, name='data')
        self.target = tf.placeholder(tf.float32, shape=[input_tensor[0],feed_forwards[-1]], name='target')
        self.layers = []

        self.session_conf = tf.ConfigProto()
        self.sess = tf.InteractiveSession(config=self.session_conf)
        self.saver = tf.train.Saver(max_to_keep=1)


        self._build_model(input_tensor=self.data)


        # 优化器
        if optimizer is None:
            self.optimizer = tf.train.GradientDescentOptimizer(self.l_rate)
        else:
            self.optimizer = optimizer(self.l_rate)
        self.optimize = self.optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())


    def _init_phase(self):
        return tf.equal(self._phase, self._train_phase)

    def _conv_stage(self, input_tensor, k_size, out_dims, name, regularizer=None,stride=1,padding='SAME'):
        """
        packing convolution function and activation function

        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param padding:
        :return:
        """
        with tf.variable_scope(name):

            # conv = self.conv2d(input_data=input_tensor, out_channel=out_dims,
            #                    kernel_size=k_size, stride=stride,regularizer=regularizer,
            #                    use_bias=False, padding=padding, name='conv')
            b_init = tf.truncated_normal_initializer(stddev=self.std) if(self._use_bias) else None
            conv = tf.contrib.layers.conv2d(inputs=input_tensor,
                                            num_outputs = out_dims,
                                            kernel_size = [k_size,k_size],
                                            weights_initializer=tf.truncated_normal_initializer(stddev=self.std),
                                            biases_initializer=b_init,
                                            stride=stride,
                                            padding = padding,
                                            activation_fn=tf.nn.relu,
                                            weights_regularizer=regularizer)

            if(self._use_bn):
                bn = self.layer_bn(input_data=conv, is_training=self._is_training, name='bn')
                return bn

            return conv

    def _full_connect_stage(self,input_tensor, out_dims, name,regularizer=None,use_activation = False, use_bias=True):
        with tf.variable_scope(name):
            # w_init = tf.random_normal([self.feed_forwards[0], self.feed_forwards[1]], stddev=self.std, dtype=tf.float32)
            # b_init = tf.random_normal([self.feed_forwards[1]], stddev=self.std, dtype=tf.float32)
            w_init = tf.truncated_normal_initializer(stddev=self.std)
            b_init = tf.truncated_normal_initializer(stddev=self.std)
            fc = self.fully_connect(input_data=input_tensor, out_dim=out_dims, w_init=w_init,b_init=b_init,name=name,regularizer=regularizer,
                                    use_bias=use_bias)
            if(self._use_bn):
                fc = self.layer_bn(input_data=fc, is_training=self._is_training, name='bn')
                fc = self.relu(input_data=fc, name='relu')
            if(use_activation):
                fc = self.relu(input_data=fc, name='relu')

        return fc

    def _build_model(self,input_tensor):
        print("layer name: {:s} shape: {}".format('input', input_tensor))


        with tf.variable_scope('convs'):
            #pool_input = self.max_pooling(input_data=input_tensor, kernel_size=2, stride=2, name='pool_input')
            conv_1 = self._conv_stage(input_tensor=input_tensor, k_size=3,out_dims=128, regularizer=self.regularizer,name='conv_1')
            pool_1 = self.max_pooling(input_data=conv_1, kernel_size=2, stride=2, name='pool1')
            self.layers.append(conv_1)
            self.layers.append(pool_1)
            # print("layer name: {:s} shape: {}".format('conv_1', conv_1))
            print("layer name: {:s} shape: {}".format('conv_1', pool_1))

            conv_2 = self._conv_stage(input_tensor=pool_1, k_size=3, out_dims=128,regularizer=self.regularizer,name='conv_2')
            pool_2 = self.max_pooling(input_data=conv_2, kernel_size=2, stride=2, name='pool2')
            self.layers.append(conv_2)
            self.layers.append(pool_2)
            print("layer name: {:s} shape: {}".format('conv_2', pool_2))

            conv_3 = self._conv_stage(input_tensor=pool_2, k_size=3, out_dims=128,regularizer=self.regularizer,name='conv_3')
            pool_3 = self.max_pooling(input_data=conv_3, kernel_size=2, stride=2, name='pool3')
            self.layers.append(conv_3)
            self.layers.append(pool_3)
            print("layer name: {:s} shape: {}".format('conv_3', pool_3))

            conv_4 = self._conv_stage(input_tensor=pool_3, k_size=3, out_dims=128,regularizer=self.regularizer,name='conv_4')
            pool_4 = self.max_pooling(input_data=conv_4, kernel_size=2, stride=2, name='pool3')

            self.layers.append(pool_4)
            print("layer name: {:s} shape: {}".format('conv_4', pool_4))


        with tf.variable_scope('full_conn'):
            # 全连接层
            flatten = tf.reshape(self.layers[-1], shape=[-1, self.feed_forwards[0]])
            print("layer name: {:s} shape: {}".format('flatten', flatten))
            fc1 = self._full_connect_stage(input_tensor=flatten,use_activation=True,out_dims=self.feed_forwards[1],use_bias=self._use_bias,regularizer=self.regularizer,name='fc1')
            fc2 = self._full_connect_stage(input_tensor=fc1,out_dims=self.feed_forwards[-1],use_bias=self._use_bias,regularizer=self.regularizer,name='fc2')
            self.layers.append(fc1)
            self.layers.append(fc2)
            print("layer name: {:s} shape: {}".format('fc_1', fc1))
            print("layer name: {:s} shape: {}".format('fc_2', fc2))
        print("Build model has done. ")

        # softmax
        self.prediction = tf.nn.softmax(logits=self.layers[-1])

        # 损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.layers[-1], labels=self.target,
                                                                name='cross')

        # 添加正则

        #tf.contrib.layers.apply_regularization(regularizer)

        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        print("tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES): {}".format(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
        self.loss = tf.reduce_mean(cross_entropy, name='loss')+reg_loss

        tf.summary.scalar('loss', self.loss)


    def train(self,data,target):
        train_feed_dict = {self.data:data}
        train_feed_dict.update({self.target:target})

        self.sess.run(self.optimize,feed_dict=train_feed_dict)
        loss,train_preds = self.sess.run([self.loss,self.prediction],train_feed_dict)
        accuracy = self.get_accuracy(train_preds,target)
        return loss,accuracy

    def encode(self, input_tensor, name):
        """
        initialize CNN-l1 network structure

        :param input_tensor:
        :param name:
        :return:
        """
        ret = OrderedDict()


        return ret

    def get_accuracy(self,logits, targets):
        batch_predictions = np.argmax(logits, axis=1)
        target = np.argmax(targets,axis=1)
        num_correct = np.sum(np.equal(batch_predictions, target))
        return 100* num_correct / batch_predictions.shape[0]

    def get_num_params(self):
        num_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        return num_params

    def save(self, save_path='./model/model.ckpt'):
        if(~os.path.exists('./model')):
            os.mkdir('./model')
        saved_path = self.saver.save(self.sess, save_path)
        print("Model saved in file: %s"%saved_path)




if __name__ == '__main__':
    a = tf.placeholder(dtype=tf.float32, shape=[1, 2048, 2048, 3], name='input')
    encoder = L1Encoder(phase=tf.constant('train', dtype=tf.string),use_bn=True)
    ret = encoder.encode(a, name='encode')
    for layer_name, layer_info in ret.items():
        print("layer name: {:s} shape: {}".format(layer_name, layer_info['shape']))