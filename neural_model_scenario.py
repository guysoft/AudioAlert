import numpy as np
import tensorflow as tf


class AudioNet_1D:
    def __init__(self, lenght, class_num=2, sess=tf.Session()):
        # Initialize an saver for store model checkpoints
        self.sess = sess
        self.in_sound = tf.placeholder(tf.float32, [None, lenght], name="pure_in_sound")
        # self.in_sound_target = tf.placeholder(tf.float32, [None, lenght])
        self.class_type = tf.placeholder(tf.float32, [None, 2])
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
        self.keep_prob = tf.placeholder(tf.float32)

        # Start building CNN layers
        # First general layers
        stride = 4
        with tf.variable_scope("first"):
            net = AudioNet_1D.conv(tf.expand_dims(self.in_sound, 2), 4, 16, stride, "conv1_0",
                                   regularizer=self.regularizer, padding="valid")
            net = AudioNet_1D.conv(net, 7, 16, stride, "conv1_1",
                                   regularizer=self.regularizer, padding="valid")
        # Resnet Blocks
        with tf.variable_scope("BlockA"):
            for n in range(5):
                net = AudioNet_1D.resblock(net, 9, 16, 1, "BlockA_" + str(n), regularizer=self.regularizer,
                                           padding="SAME")
        with tf.variable_scope("second"):
            net = AudioNet_1D.conv(net, 9, 32, stride, "conv2", regularizer=self.regularizer, padding="valid")
        with tf.variable_scope("BlockB"):
            for n in range(5):
                net = AudioNet_1D.resblock(net, 9, 32, 1, "BlockB_" + str(n), regularizer=self.regularizer,
                                           padding="SAME")
        with tf.variable_scope("thrid"):
            net = AudioNet_1D.conv(net, 9, 64, stride, "conv3", regularizer=self.regularizer, padding="valid")
        with tf.variable_scope("BlockC"):
            for n in range(20):
                net = AudioNet_1D.resblock(net, 9, 64, 1, "BlockC_" + str(n), regularizer=self.regularizer,
                                           padding="SAME")
        # Fully Connected logic classifier head
        with tf.variable_scope("cls_block"):
            net_cls = AudioNet_1D.conv(net, 9, 64, 1, "cls_layer1", regularizer=self.regularizer, padding="valid")
            net_cls = AudioNet_1D.conv(net_cls, 9, 32, 1, "cls_layer2", regularizer=self.regularizer, padding="valid")
            net_cls = AudioNet_1D.conv(net_cls, 50, class_num, 1, "cls_layer3", regularizer=self.regularizer,
                                       padding="valid")
            net_cls = tf.reduce_mean(net_cls, 1)

        self.net_cls_logit = net_cls
        self.net_cls = tf.nn.softmax(net_cls, name="prob")
        self.saver = tf.train.Saver()

    def eval_tensor(self, tensor, dict):
        res = self.sess.run(tensor, feed_dict=dict)

    def eval_track(self, x):
        return self.sess.run(self.net_cls, feed_dict={self.in_sound: np.expand_dims(x, axis=0),
                                                      self.keep_prob: 1.0})

    def load_model(self, path):
        self.saver.restore(self.sess, path)

    @staticmethod
    def resblock(x, filter_width, num_filters, stride, name,
                 padding='SAME', groups=1, regularizer=None):
        tmp = AudioNet_1D.conv(x, filter_width, num_filters, stride, name + 'a', padding, groups, True, regularizer)
        tmp = AudioNet_1D.conv(tmp, filter_width, num_filters, stride, name + 'b', padding, groups, False,
                               regularizer) + x
        # tmp =tf.add( AudioNet_1D.conv(tmp, filter_width, num_filters, stride, name + 'b', padding, groups, False,
        #                        regularizer) , x,name=name+'_merge')
        tmp = tf.nn.relu(tmp, name=name + 'relu')
        return tmp

    # @staticmethod
    # def conv(x, filter_width, num_filters, stride, name,
    #          padding='SAME', groups=1, is_relu=True, regularizer=None):
    #
    #     # Get number of input channels
    #     input_channels = int(x.get_shape()[-1])
    #
    #     # Create lambda function for the convolution
    #     convolve = lambda i, k: tf.nn.conv1d(i, k,
    #                                          stride=stride,
    #                                          padding=padding)
    #
    #     with tf.variable_scope(name) as scope:
    #         # Create tf variables for the weights and biases of the conv layer
    #         weights = tf.get_variable('weights', shape=[filter_width, input_channels / groups, num_filters],
    #                                   regularizer=regularizer)
    #         biases = tf.get_variable('biases', shape=[num_filters], regularizer=regularizer)
    #
    #         if groups == 1:
    #             conv = convolve(x, weights)
    #
    #         # In the cases of multiple groups, split inputs & weights and
    #         else:
    #             # Split input and weights and convolve them separately
    #             input_groups = tf.split(x, groups, 3)
    #             weight_groups = tf.split(weights, groups, 3)
    #             output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
    #
    #             # Concat the convolved output together again
    #             conv = tf.concat(3, output_groups)
    #
    #         # Add biases
    #         bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    #
    #         # Apply relu function
    #         if is_relu:
    #             act = tf.nn.relu(bias, name=scope.name)
    #         else:
    #             act = bias
    #
    #         return act
    @staticmethod
    def conv(x, filter_width, num_filters, stride, name,
             padding='SAME', groups=1, is_relu=True, regularizer=None):
        if is_relu:
            act = tf.layers.conv1d(x, num_filters, filter_width, stride, padding, activation=tf.nn.relu,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer, name=name)
        else:
            act = tf.layers.conv1d(x, num_filters, filter_width, stride, padding, activation=None,
                                   kernel_regularizer=regularizer, bias_regularizer=regularizer, name=name)
        return act

    @staticmethod
    def conv_tran(x, filter_width, num_filters, stride, name,
                  padding='SAME', groups=1, is_relu=True, regularizer=None):
        if is_relu:
            act = tf.layers.conv2d_transpose(x, num_filters, filter_width, stride, padding, activation=tf.nn.relu,
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer, name=name)
        else:
            act = tf.layers.conv2d_transpose(x, num_filters, filter_width, stride, padding, activation=None,
                                             kernel_regularizer=regularizer, bias_regularizer=regularizer, name=name)
        return act

    @staticmethod
    def fc(x, num_in, num_out, name, relu=True, regularizer=None):
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True, regularizer=regularizer)
            biases = tf.get_variable('biases', [num_out], trainable=True)

            # Matrix multiply weights and inputs and add bias
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            if relu == True:
                # Apply ReLu non linearity
                relu = tf.nn.relu(act)
                return relu
            else:
                return act

    @staticmethod
    def dropout(x, keep_prob):
        return tf.nn.dropout(x, keep_prob)
