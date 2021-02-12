#!/usr/bin/env python
# encoding: utf-8
'''
@author: Pan
@software: PyCharm
@file: train_cnn_dense.py
@time: 2019/3/4 15:41
@desc:
'''

import tensorflow as tf
import time
from data_provider import data_provider
from encoder_decoder_model.densenet_encoder_Pan import DenseDecoder
from utils import iterate_minibatches as iterate_minibatches
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"




'''
Load Dataset
'''


start_time = time.time()
train_x, train_y = data_provider.get_train_data()
test_x,test_y = data_provider.get_test_data()
# input_shape = [batch_size, train_x.shape[1], train_x.shape[2], train_x.shape[3]]
input_shape = [None, train_x.shape[1], train_x.shape[2], train_x.shape[3]]


# image_size = 32, img_channels = 3, class_num = 10 in cifar10
x = tf.placeholder(tf.float32, shape=input_shape, name='input_data')
label = tf.placeholder(tf.float32, shape=[input_shape[0], train_y.shape[1]], name='input_label')
print("input_data:batch, in_height, in_width, in_channels : {:s}".format(str(input_shape)))
print("finish unzipping training data. {:.3f}s taken.".format(time.time()-start_time))

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

'''
set parameters
'''

# Hyperparameter
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-4 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
batch_size = 16

train_batch_num = train_x.shape[0] / batch_size
test_batch_num = test_x.shape[0] / batch_size

test_iteration = 10

total_epochs = 100

class_num = 10

'''
Set training model
'''

logits = DenseDecoder(input_tensor=x,phase='train',block_num=nb_block, growth_rate=growth_k,class_num=class_num).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

'''
Define the test method
'''
def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0

    for test_batch_x,test_batch_y in iterate_minibatches(inputs=test_x, targets=test_y, batchsize=batch_size):
        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_ / test_batch_num
        test_acc += acc_ / test_batch_num

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary
'''
Train
'''
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess :
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate
    for epoch in range(1, total_epochs + 1):
        print("Start epoch:%d/%d" % (epoch, total_epochs))
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        pre_index = 0
        train_acc = 0.0
        train_loss = 0.0

        for train_input, train_target in iterate_minibatches(inputs=train_x, targets=train_y,
                                                             batchsize=batch_size):
            train_feed_dict = {
                x: train_input,
                label: train_target,
                learning_rate: epoch_learning_rate,
                training_flag : True
            }

            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)

            train_loss += batch_loss
            train_acc += batch_acc
            pre_index += batch_size

        train_loss /= train_batch_num  # average loss
        train_acc /= train_batch_num  # average accuracy

        train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                          tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

        test_acc, test_loss, test_summary = Evaluate(sess)

        summary_writer.add_summary(summary=train_summary, global_step=epoch)
        summary_writer.add_summary(summary=test_summary, global_step=epoch)
        summary_writer.flush()

        line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
            epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
        print(line)

        with open('logs.txt', 'a') as f:
            f.write(line)



        saver.save(sess=sess, save_path='./model/dense.ckpt')