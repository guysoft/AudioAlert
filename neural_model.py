import tensorflow as tf
import numpy as np
import math
class AudioNet_1D:
    def __init__(self, lenght, class_num=2, sess=None, blocks_depth=None, filter_width=11):
        # Initialize an saver for store model checkpoints
        # if sess is None:
        #     sess = tf.Session()
        self.sess = sess
        self.in_sound = tf.placeholder(tf.float32, [None, lenght], name="pure_in_sound")
        # self.in_sound_target = tf.placeholder(tf.float32, [None, lenght])
        self.class_type = tf.placeholder(tf.float32, [None, class_num])
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
        self.keep_prob = tf.placeholder(tf.float32)

        # Start building CNN layers
        # First general layers
        stride = 2
        # stride = 4
        with tf.variable_scope("first"):
            net = AudioNet_1D.conv(tf.expand_dims(self.in_sound, 2), 31, 16, stride, "conv1_0",
                                   regularizer=self.regularizer,padding="SAME")
        # Resnet Blocks

        if blocks_depth is None:
            blocks_depth = [3, 3, 5, 5, 5]

        with tf.variable_scope("BlockA"):
            for n in range(blocks_depth[0]):
                net = AudioNet_1D.resblock(net, filter_width, 16, 1, "BlockA_" + str(n), regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("second"):
            net = AudioNet_1D.conv(net, filter_width, 32, stride, "conv2", regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("BlockB"):
            for n in range(blocks_depth[1]):
                net = AudioNet_1D.resblock(net, filter_width, 32, 1, "BlockB_" + str(n), regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("thrid"):
            net = AudioNet_1D.conv(net, filter_width, 64, stride, "conv3", regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("BlockC"):
            for n in range(blocks_depth[2]):
                net = AudioNet_1D.resblock(net, filter_width, 64, 1, "BlockC_" + str(n), regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("forth"):
            net = AudioNet_1D.conv(net, filter_width, 128, stride, "conv3", regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("BlockD"):
            for n in range(blocks_depth[3]):
                net = AudioNet_1D.resblock(net, filter_width, 128, 1, "BlockC_" + str(n), regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("forth"):
            net = AudioNet_1D.conv(net, filter_width, 256, stride, "conv4", regularizer=self.regularizer, padding="SAME")
        with tf.variable_scope("BlockD"):
            for n in range(blocks_depth[4]):
                net = AudioNet_1D.resblock(net, filter_width, 256, 1, "BlockD_" + str(n), regularizer=self.regularizer, padding="SAME")
        # Fully Connected logic classifier head
        with tf.variable_scope("cls_block"):
            net_cls = AudioNet_1D.conv(net, filter_width, 64, 1, "cls_layer1", regularizer=self.regularizer, padding="SAME")
            net_cls = AudioNet_1D.conv(net_cls, filter_width, 32, 1, "cls_layer2", regularizer=self.regularizer, padding="SAME")
            net_cls = self.dropout(net_cls,self.keep_prob   )
            # net_cls = AudioNet_1D.conv(net_cls, 32, class_num, 1, "cls_layer3", regularizer=self.regularizer,padding="SAME")
            net_cls = AudioNet_1D.conv(net_cls, 9, 2 * class_num, 1, "cls_layer3", regularizer=self.regularizer,
                                       padding="SAME")
            net_cls = tf.layers.flatten(net_cls)
            net_cls = AudioNet_1D.fc(net_cls,math.ceil(lenght/2**5) *2 * class_num, class_num, relu=False, name="fc_classification")

        self.net_cls_logit = net_cls
        self.net_cls = tf.nn.softmax(net_cls, name="prob", axis=1)

        self.saver = tf.train.Saver(tf.trainable_variables())

    def eval_tensor(self, tensor, dict):
        res = self.sess.run(tensor, feed_dict=dict)

    def eval_track(self, x):
        return self.sess.run(self.net_cls, feed_dict={self.in_sound: np.expand_dims(x, axis=0),
                                                      self.keep_prob: 1.0})

    def load_model(self, path):
        self.saver.restore(self.sess, path)

    @staticmethod
    def resblock(x, filter_width, num_filters, stride, name,
                 padding='SAME', groups=1, regularizer=None,concate = 1):
        tmp = AudioNet_1D.conv(x, filter_width, num_filters, stride, name + 'a', padding, groups, True, regularizer)
        tmp = AudioNet_1D.conv(tmp, filter_width, num_filters, stride, name + 'b', padding, groups, False,
                               regularizer)
        tmp = tf.concat([tmp for _ in range(concate)],axis=2) + x
        tmp = tf.nn.relu(tmp, name=name + 'relu')
        return tmp


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
