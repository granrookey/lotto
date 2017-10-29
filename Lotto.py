import numpy as np
import datetime
import csv
import tensorflow as tf

from random import shuffle
from lotto_model import lenet, load_batch, preprocess

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_string('log_dir', './log/train', 'Directory with the log data.')
flags.DEFINE_float('test_size', .1, 'test_set_size')
FLAGS = flags.FLAGS

def generate_dataset(subset='train'):
    train = {'input': [], 'label':[]}
    test = {'input': [], 'label':[]}
    dataset = {'train': train, 'test': test}

    with open('Lotto_csv.csv', 'r', encoding='utf-8') as f:
        lucky_csv = csv.reader(f)
        next(lucky_csv)

        lucky_data = list(lucky_csv)
        shuffle(lucky_data)
        print(len(lucky_data))
        test_data = int(len(lucky_data) * .1)
        train_data = len(lucky_data) - test_data

        print ("Train data set is {}".format(train_data))
        print ("Test data set is {}".format(test_data))

        inputs = []
        labels = []
        for idx, lucky in enumerate(lucky_data):
            lucky_input = []
            lucky = lucky[1:]
            lucky_date = datetime.date(int(lucky[0]), int(lucky[1]), int(lucky[2]))
            lucky_input.append(lucky_date.strftime("%Y%m%d"))
            random_lucky = list(np.random.randint(1, high=45, size=7))
            random_lucky.sort()
            for i in random_lucky:
                str_num = '{:02d}'.format(i)
                lucky_input.append(str_num)

            separate_input = [input for item in lucky_input for input in item[:]]
            separate_input = list(map(int, separate_input))
            inputs.append(separate_input)

            lucky_label = [label for label in lucky[3:]]
            labels.append(list(map(int, lucky_label)))

        train['input'] = inputs[:train_data]
        train['label'] = labels[:train_data]
        test['input'] = inputs[train_data :]
        test['label'] = labels[train_data :]

        print(train['input'][0])
        print(train['label'][0])
        print(test['input'][0])
        print(test['label'][0])

    return dataset[subset]

def main(args):
    # load the dataset
    dataset = generate_dataset('train')

    # print (dataset)

    # for input in dataset['input']:
    #     print (input)
    #
    # for label in dataset['label']:
    #     print(label)
    # load batch of dataset
    inputs, labels = load_batch(
        dataset,
        FLAGS.batch_size,
        is_training=True)

    print(inputs)
    print(labels)

    # run the image through the model
    predictions = lenet(inputs, FLAGS.filter_sizes)

    # get the cross-entropy loss
    one_hot_labels = slim.one_hot_encoding(
        labels,
        dataset.num_classes)
    slim.losses.softmax_cross_entropy(
        predictions,
        one_hot_labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', total_loss)
    #
    # # use RMSProp to optimize
    # optimizer = tf.train.RMSPropOptimizer(0.001, 0.9)
    #
    # # create train op
    # train_op = slim.learning.create_train_op(
    #     total_loss,
    #     optimizer,
    #     summarize_gradients=True)
    #
    # # run training
    # slim.learning.train(
    #     train_op,
    #     FLAGS.log_dir,
    #     save_summaries_secs=20)


if __name__ == '__main__':
    tf.app.run()
