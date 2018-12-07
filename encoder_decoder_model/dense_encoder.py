# -*- coding: utf-8 -*-
# @Time    : 18-12-07
# @Author  : Yang Jiao
# @Site    : http://github.com/mrjiao2018
# @File    : dense_encoder.py
# @IDE     : PyCharm Community Edition
"""
packing Dense neural network model
"""
import tensorflow as tf
from collections import OrderedDict
from encoder_decoder_model import cnn_base_model


class DenseEncoder(cnn_base_model.CNNBaseModel):
    """
    packing Dense neural network model
    """
    def __init__(self, phase, L, N, growth_rate, with_bc=False, bc_theta=0.5):
        """

        :param phase:           is training or testing
        :param L:               L refers to the depth of the network according to DenseNet paper
        :param N:               N refers to block numbers of the network according to DenseNet paper
        :param growth_rate:     growth_rate refers to the output dimensions of the dense block
        :param with_bc:         whether to use DenseNet-BC in the model
        :param bc_theta:        transition theta threshold
        """
        super(DenseEncoder, self).__init__()
        self._phase = phase
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._is_training = self._init_phase()
        self._L = L
        self._N = N
        self._block_depth = int((L - N - 1) / N)
        self._growth_rate = growth_rate
        self._with_bc = with_bc
        self._bc_theta = bc_theta


    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def _composite_conv(self, input_data, out_channel, name):
        """
        Implement the composite function mentioned in DenseNet paper

        :param input_data:
        :param out_channel:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            bn_1 = self.layer_bn(input_data=input_data, is_training=self._is_training, name='bn_1')

            relu_1 = self.relu(input_data=bn_1, name='relu_1')

            if self._with_bc:
                conv_1 = self.conv2d(input_data=relu_1, out_channel=out_channel, kernel_size=1,
                                     padding='SAME', stride=1, use_bias=False, name='conv_1')

                bn_2 = self.layer_bn(input_data=conv_1, is_training=self._is_training, name='bn_2')
                relu_2 = self.relu(input_data=bn_2, name='relu_2')
                conv_2 = self.conv2d(input_data=relu_2, out_channel=out_channel, kernel_size=3,
                                     padding='SAME', stride=1, use_bias=False, name='conv_2')

            else:
                conv_2 = self.conv2d(input_data=relu_1, out_channel=out_channel, kernel_size=3,
                                     padding='SAME', stride=1, use_bias=False, name='conv_2')

            return conv_2

    def _dense_connect_layer(self, input_data, name):
        """
        implement the equation (2) in the DenseNet paper to concatenate the dense block feature maps

        :param input_data:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv_out = self._composite_conv(input_data=input_data, out_channel=self._growth_rate,
                                            name='composite_conv')
            concate_cout = tf.concat(values=[conv_out, input_data], axis=3, name='concatenate')

        return concate_cout

    def _transition_layer(self, input_data, name):
        """
        consist of 1 x 1 conv & average pool

        :param input_data:
        :param name:
        :return:
        """
        input_channels = input_data.get_shape().as_list()[3]

        with tf.variable_scope(name):
            # First batch norm
            bn = self.layer_bn(input_data=input_data, is_training=self._is_training, name='bn')

            # Second 1 X 1 conv
            if self._with_bc:
                out_channels = int(input_channels * self._bc_theta)
                conv = self.conv2d(input_data=bn, out_channel=out_channels, kernel_size=1,
                                   stride=1, use_bias=False, name='conv')
                # Third average pooling
                avgpool_out = self.avg_pooling(input_data=conv, kernel_size=2, stride=2, name='avgpool')
            else:
                conv = self.conv2d(input_data=bn, out_channel=input_channels, kernel_size=1,
                                   stride=1, use_bias=False, name='conv')
                # Third average pooling
                avgpool_out = self.avg_pooling(input_data=conv, kernel_size=2, stride=2, name='avgpool')

            return avgpool_out

    def _dense_block(self, input_data, name):
        """
        implement dense block mentioned in DenseNet Figure 1

        :param input_data:
        :param name:
        :return:
        """
        block_input = input_data
        with tf.variable_scope(name):
            for i in range(self._block_depth):
                block_layer_name = '{:s}_layer_{:d}'.format(name, i+1)
                block_input = self._dense_connect_layer(input_data=block_input, name=block_layer_name)

        return block_input

    def encode(self, input_tensor, name):
        """
        implement DenseNet structure

        :param input_tensor:
        :param name:
        :return:
        """
        encode_ret = OrderedDict()

        # First apply a 3*3*16 out channels conv layer
        # mentioned in DenseNet paper Implementation Details part
        with tf.variable_scope(name):
            conv1 = self.conv2d(input_data=input_tensor, out_channel=16,
                                kernel_size=3, use_bias=False, name='conv1')
            dense_block_input = conv1

            for dense_block_index in range(self._N):
                dense_block_name = 'Dense_Block_{:d}'.format(dense_block_index + 1)
                dense_block_name = 'Dense_Block_{:d}'.format(dense_block_index + 1)

                # dense connectivity
                dense_block_out = self._dense_block(input_data=dense_block_input, name=dense_block_name)

                # apply the transition part
                dense_block_out = self._transition_layer(input_data=dense_block_out, name=dense_block_name)

                dense_block_input = dense_block_out

                encode_ret[dense_block_name] = dict()
                encode_ret[dense_block_name]['data'] = dense_block_out
                encode_ret[dense_block_name]['shape'] = dense_block_out.get_shape().as_list()

        return encode_ret


if __name__ == '__main__':
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 384, 1248, 3], name='input_tensor')
    encoder = DenseEncoder(L=100, N=5, growth_rate=16, with_bc=True, phase=tf.constant('train'))
    ret = encoder.encode(input_tensor=input_tensor, name='Dense_encode')
    for layer_name, layer_info in ret.items():
        print('layer_name: {:s} shape: {}'.format(layer_name, layer_info['shape']))