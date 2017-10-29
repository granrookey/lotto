import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def preprocess(input_data, label_data):
    inputs = []
    labels =[]
    for input in input_data:
        # print(input)
        input_embedding = np.zeros((len(input_data[0]), 10))
        for i, num in enumerate(input):
            input_embedding[i][num] = 1

        inputs.append(input_embedding)

    inputs = tf.stack(inputs)
    print(inputs)

    for label in label_data:
        label_embedding = np.zeros(46)
        for num in label:
            label_embedding[num] = 1

        labels.append(label_embedding)

    labels = tf.stack(labels)
    print(labels)

    return inputs, labels

def lenet(inputs, filter_size):

    pooled_outputs = []
    for i, filter in enumerate(filter_size):
        with tf.name_scope("conv-maxpool-%s" % filter):
            net = slim.conv2d(inputs, 92, [filter, filter], scope='conv1')
            net = slim.max_pool2d(net, [2,2], scope='pool1')
            net = slim.conv2d(net, 184, [filter, filter], scope='conv2')
            net = slim.max_pool2d(net, [2,2], scope='pool2')
            pooled_outputs.append(net)

    net = slim.flatten(net)
    with tf.name_scope("dropout"):
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
    return net


def load_batch(dataset, batch_size=32, is_training=False):
    input, label = preprocess(dataset['input'], dataset['label'])

    images, labels = tf.train.batch(
        [input, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels