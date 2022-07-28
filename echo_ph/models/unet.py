import tensorflow as tf


# Convolution Layer
def conv(x, filter_size, num_filters, stride, name, padding='SAME', groups=1, trainable=True, relu=True):
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda x, W: tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)

    with tf.compat.v1.variable_scope(name):
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.compat.v1.get_variable('W',
                                            shape=[filter_size, filter_size, input_channels // groups, num_filters],
                                            initializer=tf.initializers.GlorotUniform(),  # same as xaviar
                                            trainable=trainable,
                                            collections=['variables'])
        biases = tf.compat.v1.get_variable('b', shape=[num_filters], trainable=trainable,
                                           initializer=tf.zeros_initializer())

        if groups == 1:
            conv = convolve(x, weights)

        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, axis=3)
            weight_groups = tf.split(weights, groups, axis=3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(output_groups, axis=3)
        if relu:
            return tf.nn.relu(conv + biases)
        else:
            return conv + biases


def deconv(x, filter_size, num_filters, stride, name, padding='SAME', relu=True):
    activation = None
    if relu:
        activation = tf.nn.relu
    return tf.compat.v1.layers.conv2d_transpose(x, num_filters, filter_size, stride, padding=padding,
                                                kernel_initializer=tf.initializers.GlorotUniform(),
                                                activation=activation, name=name)


def max_pool(x, filter_size, stride, name=None, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1], strides=[1, stride, stride, 1], padding=padding,
                          name=name)


class Unet(object):
    def __init__(self, mean, weight_decay, learning_rate, label_dim, maxout=False):
        self.x_train = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_train = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.x_test = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_test = tf.compat.v1.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.unet(self.x_train, mean)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_train))
        self.opt = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.unet(self.x_test, mean, keep_prob=1.0, reuse=True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    #  self.train_summary = tf.summary.scalar('training_accuracy', self.train_accuracy)

    # Gradient Descent on mini-batch
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary),
                                         feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary

    def predict(self, sess, x):
        prediction = sess.run(self.pred, feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob=0.5, reuse=None):
        width = 1
        label_dim = self.label_dim
        with tf.compat.v1.variable_scope('vgg', reuse=reuse):
            input = input - mean
            pool_ = lambda x: max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding='SAME', relu=True, filter_size=3: conv(x, filter_size,
                                                                                                 output_depth, 1,
                                                                                                 name=name,
                                                                                                 padding=padding,
                                                                                                 relu=relu)
            deconv_ = lambda x, output_depth, name: deconv(x, 2, output_depth, 2, name=name)
            conv_1_1 = conv_(input, int(64 * width), 'conv1_1')
            conv_1_2 = conv_(conv_1_1, int(64 * width), 'conv1_2')

            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, int(128 * width), 'conv2_1')
            conv_2_2 = conv_(conv_2_1, int(128 * width), 'conv2_2')

            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, int(256 * width), 'conv3_1')
            conv_3_2 = conv_(conv_3_1, int(256 * width), 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, int(512 * width), 'conv4_1')
            conv_4_2 = conv_(conv_4_1, int(512 * width), 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = conv_(pool_4, int(1024 * width), 'conv5_1')
            conv_5_2 = conv_(conv_5_1, int(1024 * width), 'conv5_2')

            pool_5 = pool_(conv_5_2)

            conv_6_1 = tf.compat.v1.nn.dropout(conv_(pool_5, int(2048 * width), 'conv6_1'), keep_prob)
            conv_6_2 = tf.compat.v1.nn.dropout(conv_(conv_6_1, int(2048 * width), 'conv6_2'), keep_prob)

            up_7 = tf.concat([deconv_(conv_6_2, int(1024 * width), 'up7'), conv_5_2], 3)

            conv_7_1 = conv_(up_7, int(1024 * width), 'conv7_1')
            conv_7_2 = conv_(conv_7_1, int(1024 * width), 'conv7_2')

            up_8 = tf.concat([deconv_(conv_7_2, int(512 * width), 'up8'), conv_4_2], 3)

            conv_8_1 = conv_(up_8, int(512 * width), 'conv8_1')
            conv_8_2 = conv_(conv_8_1, int(512 * width), 'conv8_2')

            up_9 = tf.concat([deconv_(conv_8_2, int(256 * width), 'up9'), conv_3_2], 3)

            conv_9_1 = conv_(up_9, int(256 * width), 'conv9_1')
            conv_9_2 = conv_(conv_9_1, int(256 * width), 'conv9_2')

            up_10 = tf.concat([deconv_(conv_9_2, int(128 * width), 'up10'), conv_2_2], 3)

            conv_10_1 = conv_(up_10, int(128 * width), 'conv10_1')
            conv_10_2 = conv_(conv_10_1, int(128 * width), 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, int(64 * width), 'up11'), conv_1_2], 3)

            conv_11_1 = conv_(up_11, int(64 * width), 'conv11_1')
            conv_11_2 = conv_(conv_11_1, int(64 * width), 'conv11_2')

            conv_12 = conv_(conv_11_2, label_dim, 'conv12_2', filter_size=1, relu=False)
            return conv_12
