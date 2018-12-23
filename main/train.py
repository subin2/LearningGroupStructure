import os
import csv
import time
import random
# import cPickle
from tqdm import tqdm
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib import colors
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from utils import iterate_minibatches as iterate_minibatches
from tensorflow.python.client import timeline

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        # dict = pickle.load(fo)
        dict = pickle.load(fo, encoding='latin1')
    return dict



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
    print(tmp)
    train_data.append(tmp["data"])
    train_label.append(tmp["labels"])
train_data = np.concatenate(train_data).reshape([-1, 32, 32, 3], order='F') # num, width=32, height=32, channel = 3
train_label = np.concatenate(train_label)
train_label = one_hot_enc.fit_transform(train_label.reshape([-1, 1])) # [data.num, class = 10]
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

num_epochs = 100
train_batch_num = train_data.shape[0] / batch_size
# valid_batch_num = valid_data.shape[0] / batch_size
test_batch_num = test_data.shape[0] / batch_size

l_step = 300 * train_batch_num
l_decay = 0.1

regular_scale=0.001

"""
Set Path
"""

data_path = './CIFAR-10'
data_save_path = os.path.join('/data2/subin/regularize', data_path[2:])

if not os.path.exists(data_path):
    print('creating difectory {}'.format(data_path))
    os.mkdir(os.path.join(data_path))

save_path = os.path.join(data_path, 'l1cnn-clvar-freeze')
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'gradientdescentopt') if optimizer is None else os.path.join(save_path,
                                                                                                 str(optimizer).split(
                                                                                                     '.')[-1][:-3])
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'use_batch_norm') if use_batchnorm else os.path.join(save_path, 'no_batch_norm')
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, 'no_dropout') if not use_dropout else os.path.join(save_path, 'use_dropout')
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))

save_path = os.path.join(save_path, str(nonlinearity).split(' ')[1]) if nonlinearity is not None else os.path.join(
    save_path, 'None')
if not os.path.exists(save_path):
    print('creating difectory {}'.format(save_path))
    os.mkdir(os.path.join(save_path))


model_name = "model"
if use_dropout:
    model_name = model_name + '-keep' + str(keep_probs).replace(', ', '_')[1:-1]
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

from encoder_decoder_model.l1_encoder_Pan import L1Encoder

start_time = time.time()
if 'model' in globals():
    model = globals()["model"]
    model.terminate()

# 在此更改模型
model = L1Encoder(input_tensor=inputs,
                  phase=tf.constant('train', dtype=tf.string),
                  feed_forwards=feed_forwards,
                  l_decay=l_decay,
                  l_rate=l_rate,
                  l_step=l_step,
                  optimizer=optimizer,
                  keep_probs=keep_probs,
                  regular_scale=regular_scale,
                  std=std,
                  use_bn=use_batchnorm)

print('Model build Done. {:.3f}s taken.'.format(time.time()-start_time))

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



train_loss=[]
train_err=[]
valid_loss=[]
valid_err=[]
train_accuracy=[]
test_accuracy=[]

seismic_rgb = cm.get_cmap(plt.get_cmap('seismic'))(np.linspace(0.0, 1.0, 100))[:, :3]
print(seismic_rgb.shape)

seismic_gray = np.mean(seismic_rgb,axis=1)
seismic_gray = np.stack([seismic_gray, seismic_gray, seismic_gray], axis=1)
print(seismic_gray.shape)

seismic_gray = colors.ListedColormap(seismic_gray, name='seismic_gray')

profile=False
first=True
test_count = 0

def test(test_data, test_labels, batch_size, model, test_batch_num):
    accuracy=0.0
    keep_probs_values = [1.0 for i in range(len(model.keep_probs_values))]
    for batch in iterate_minibatches(inputs=test_data, targets=test_labels, batchsize=batch_size):
        test_in, test_target = batch
        #test_in = test_in[:,np.newaxis,:,np.newaxis]
        #print model.sess.run(tf.reduce_sum(tf.equal(tf.argmax(model.output_layer,1), tf.argmax(model.y, 1))) ,
        #                            feed_dict={model.x:test_in, model.y:test_target})

        accuracy += model.sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.layers[-1],1), tf.argmax(model.target, 1)), tf.float32)),
                                feed_dict={model.data:test_in, model.target:test_target, model.keep_probs:keep_probs_values})
    # print'accuracy: {}'.format(accuracy/test_batch_num)
    return accuracy/test_batch_num


for epoch in tqdm(range(num_epochs)):
    start_time = time.time()
    loss = 0
    err = 0
    accuracy = 0
    for batch in iterate_minibatches(inputs=train_data, targets=train_label, batchsize=batch_size):
        train_in, train_target = batch
        #train_in = train_in[:,np.newaxis,:,np.newaxis]
        loss_,accuracy_ = model.train(data=train_in,target=train_target)
        if profile:
            fetched_timeline = timeline.Timeline(model.run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('l1cnn-timeline_01_step_0.json', 'w') as f:
                f.write(chrome_trace)
        #if first:
            #model.writer.add_summary(tmp_sum, epoch)
        profile=False
        loss +=loss_
        accuracy += accuracy_
        #err += err_
    #model.writer.add_summary(tmp_sum, epoch)#model.global_step.eval())
    train_loss.append(loss/train_batch_num)
    train_accuracy.append(accuracy/train_batch_num)
    train_err.append(err/train_batch_num)
    #train_Q.append(Q/train_batch_num)
    # train_history.loc[epoch] = [epoch+1, train_loss[-1], train_err[-1],
    #                             time.strftime("%Y-%m-%d-%H:%M", time.localtime())]

    # if (epoch+1)% save_freq == 0:
    #     model.save(os.path.join(data_save_path, model_name, str(epoch+1)+'.ckpt'))
    #     train_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-train_history.csv"))
    #     #valid_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-valid_history.csv"))
    #
    #     w_mask_pass = []
    #
    #     param_history.loc[epoch/save_freq] = [epoch+1] + w_mask_pass +[time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
    #     param_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-param_history.csv"))
    print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:    {:.6f}".format(train_loss[-1]))
    print("  training accuracy:    {:.2f}%".format(train_accuracy[-1]))
    #print("  training err:     {:.6f}".format(train_err[-1]))

    # if epoch>30:
    #     if (train_Q[-1] - train_Q[-2])/train_Q[-2] <0 :
    #         Q_conv_count += 1
    #     else:
    #         Q_conv_count = 0
    #     if Q_conv_count>=3:
    #         if not model.frozen:
    #             model.freeze()
    #             frozen_epoch = epoch+1
    #             test_epoch[0] = frozen_epoch
    #             print('##### Model frozen at epoch '+str(epoch+1)+'#####')
    if (epoch+1) in test_epoch:
        test_accuracy.append(test(test_data, test_label, batch_size, model, test_batch_num))
        train_accuracy.append(test(train_data, train_label, batch_size, model, train_batch_num))
        test_history.loc[test_count] = [epoch+1, train_accuracy[-1], test_accuracy[-1], time.strftime("%Y-%m-%d-%H:%M", time.localtime())]
        test_count +=1
        test_history.to_csv(os.path.join(save_path, model_name, str(num_epochs)+"epochs-test_history.csv"))
        print("test accuracy:   {:.2f}%".format(test_accuracy[-1] * 100))




print(os.path.join(save_path,model_name))
print('Frozen at '+str(frozen_epoch))
params_num = model.get_num_params()
print("params_num:  {}".format(params_num))

#model.save()

# b = tf.get_default_graph().get_tensor_by_name("convs/conv_1/Conv/biases:0")
# w = tf.get_default_graph().get_tensor_by_name("convs/conv_1/Conv/weights:0")
# #print("weight:{}\n bias:{}".format(w,b))
# model.sess.run(tf.print(w))
# for tv in tf.trainable_variables():
#     print (tv.name)