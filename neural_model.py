import tensorflow as tf


class AudioNet_1D:
    def __init__(self, lenght):
        self.in_sound = tf.placeholder(tf.float32, [None, lenght, 1])
        self.in_sound_target = tf.placeholder(tf.float32, [None, lenght, 1])
        self.class_type = tf.placeholder(tf.float32, [None, 2])
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1.0)
        self.keep_prob = tf.placeholder(tf.float32)

        # Start building CNN layers
        # First general layers
        net = AudioNet_1D.conv(self.in_sound, 49, 16, 2, "conv1", regularizer=self.regularizer)
        # Resnet Blocks
        with tf.name_scope("BlockA"):
            for n in range(5):
                net = AudioNet_1D.resblock(net, 9, 16, 1, "BlockA_" + str(n), regularizer=self.regularizer)
        net = AudioNet_1D.conv(net, 9, 32, 2, "conv2", regularizer=self.regularizer)
        with tf.name_scope("BlockB"):
            for n in range(5):
                net = AudioNet_1D.resblock(net, 9, 32, 1, "BlockB_" + str(n), regularizer=self.regularizer)
        net = AudioNet_1D.conv(net, 9, 64, 2, "conv3", regularizer=self.regularizer)
        with tf.name_scope("BlockC"):
            for n in range(20):
                net = AudioNet_1D.resblock(net, 9, 64, 1, "BlockC_" + str(n), regularizer=self.regularizer)
        # Fully Connected logic classifier head
        with tf.name_scope("cls_block"):
            net_cls = AudioNet_1D.conv(net, 9, 64, 1, "cls_layer1", regularizer=self.regularizer)
            net_cls = AudioNet_1D.conv(net_cls, 9, 32, 1, "cls_layer2", regularizer=self.regularizer)
            net_cls = AudioNet_1D.conv(net_cls, 50, 2, 1, "cls_layer3", regularizer=self.regularizer)
            net_cls = tf.reduce_max(net_cls, 1)
        # Extraction head
        with tf.name_scope("splt_block"):
            net = AudioNet_1D.conv(net, 9, 32, 2, "conv3", regularizer=self.regularizer)
            net = tf.reshape(tf.image.resize_nearest_neighbor(tf.expand_dims(net, 3), (lenght, 32)), (-1, lenght, 32))
            net_splt_1 = AudioNet_1D.conv(net, 9, 32, 1, "cls_layer1", regularizer=self.regularizer)
            net_splt_2 = AudioNet_1D.conv(net_splt_1, 9, 16, 1, "cls_layer2", regularizer=self.regularizer)
            net_splt_3 = AudioNet_1D.conv(net_splt_2, 9, 8, 1, "cls_layer3", regularizer=self.regularizer)
            # net_splt =  tf.concat(values=[net_splt_1,net_splt_2,net_splt_3, net], axis=1)
            net_splt = AudioNet_1D.conv(net_splt_3, 9, 1, 1, "clean_out", regularizer=self.regularizer, is_relu=False)

        self.net_cls_logit = net_cls
        self.net_cls = tf.nn.softmax(net_cls, name="prob")
        self.net_splt = net_splt

    @staticmethod
    def resblock(x, filter_width, num_filters, stride, name,
                 padding='SAME', groups=1, regularizer=None):
        net = AudioNet_1D.conv(x, filter_width, num_filters, stride, name + 'a', padding, groups, True, regularizer)
        net = AudioNet_1D.conv(net, filter_width, num_filters, stride, name + 'b', padding, groups, False,
                               regularizer) + x
        net = tf.nn.relu(net)
        return net

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
                                   kernel_regularizer=regularizer)
        else:
            act = tf.layers.conv1d(x, num_filters, filter_width, stride, padding, activation=None,
                                   kernel_regularizer=regularizer)
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
