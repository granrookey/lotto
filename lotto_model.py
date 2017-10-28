import tensorflow as tf

slim = tf.contrib.slim

def preprocess(inputs, labels):
    dataset['input']
    dataset['label']

def lenet(images):
    net = slim.conv2d(images, 20, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 50, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten3')
    net = slim.fully_connected(net, 500, scope='fc4')
    net = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
    return net


def load_batch(dataset, batch_size=32, is_training=False):
    inputs, labels = preprocess(dataset['input'], dataset['label'])

    images, labels = tf.train.  batch(
        [image, label],
        batch_size=batch_size,
        allow_smaller_final_batch=True)

    return images, labels