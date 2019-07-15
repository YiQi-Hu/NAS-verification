# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import tensorflow.python.platform
from tensorflow.python.platform import gfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from cifar10_input import distorted_inputs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.


class NetworkUnit:
    def __init__(self, graph_part=[[]], cell_list=[]):
        self.graph_part = graph_part
        self.cell_list = cell_list
        return


def _makeconv(inputs, hplist, node):
    """Generates a convolutional layer according to information in hplist

    Args:
    inputs: inputing data.
    hplist: hyperparameters for building this layer
    node: number of this cell
    Returns:
    tensor.
    """
    # print('Evaluater:right now we are making conv layer, its node is %d, and the inputs is'%node,inputs,'and the node before it is ',cellist[node-1])
    with tf.variable_scope('conv' + str(node)) as scope:
        inputdim = inputs.shape[3]
        kernel = tf.get_variable('weights', shape=[hplist[2], hplist[2], inputdim, hplist[1]],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', hplist[1], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if hplist[3] == 'relu':
            conv1 = tf.nn.relu(bias, name=scope.name)
        elif hplist[3] == 'tenh' or hplist[3] == 'tanh':
            conv1 = tf.tanh(bias, name=scope.name)
        elif hplist[3] == 'sigmoid':
            conv1 = tf.sigmoid(bias, name=scope.name)
        elif hplist[3] == 'identity':
            conv1 = tf.identity(bias, name=scope.name)
        elif hplist[3] == 'leakyrelu':
            conv1 = tf.nn.leaky_relu(bias, name=scope.name)
    return conv1


def _makepool(inputs, hplist):
    """Generates a pooling layer according to information in hplist

    Args:
        inputs: inputing data.
        hplist: hyperparameters for building this layer
    Returns:
        tensor.
    """
    if hplist[1] == 'avg':
        return tf.nn.avg_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                              strides=[1, hplist[2], hplist[2], 1], padding='SAME')
    elif hplist[1] == 'max':
        return tf.nn.max_pool(inputs, ksize=[1, hplist[2], hplist[2], 1],
                              strides=[1, hplist[2], hplist[2], 1], padding='SAME')
    elif hplist[1] == 'global':
        return tf.reduce_mean(inputs, [1, 2], keep_dims=True)


def batch_norm(input):
    # return input
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def _makedense(inputs, hplist):
    """Generates dense layers according to information in hplist

    Args:
               inputs: inputing data.
               hplist: hyperparameters for building layers
               node: number of this cell
    Returns:
               tensor.
    """
    i = 0
    print(inputs.shape)
    inputs = tf.reshape(inputs, [-1, 2 * 2 * 512])

    for neural_num in hplist[1]:
        with tf.variable_scope('dense' + str(i)) as scope:
            weights = tf.get_variable('weights', shape=[inputs.shape[-1], neural_num],
                                      initializer=tf.contrib.keras.initializers.he_normal())
            weight = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
            # tf.add_to_collection('losses', weight)
            biases = tf.get_variable('biases', [neural_num], initializer=tf.constant_initializer(0.0))
            if hplist[2] == 'relu':
                local3 = tf.nn.relu(tf.matmul(inputs, weights) + biases, name=scope.name)
            elif hplist[2] == 'tanh':
                local3 = tf.tanh(tf.matmul(inputs, weights) + biases, name=scope.name)
            elif hplist[2] == 'sigmoid':
                local3 = tf.sigmoid(tf.matmul(inputs, weights) + biases, name=scope.name)
            elif hplist[2] == 'identity':
                local3 = tf.identity(tf.matmul(inputs, weights) + biases, name=scope.name)
        inputs = local3
        i += 1
    return inputs


def inference(images, graph_part, cellist):  # ,regularizer):
    '''Method for recovering the network model provided by graph_part and cellist.
    Args:
      images: Images returned from Dataset() or inputs().
      graph_part: The topology structure of th network given by adjacency table
      cellist:

    Returns:
      Logits.'''
    # print('Evaluater:starting to reconstruct the network')
    nodelen = len(graph_part)
    inputs = [0 for i in range(nodelen)]  # input list for every cell in network
    inputs[0] = images
    getinput = [False for i in range(nodelen)]  # bool list for whether this cell has already got input or not
    getinput[0] = True
    # bool list for whether this cell has already been in the queue or not
    inqueue = [False for i in range(nodelen)]
    inqueue[0] = True
    q = []
    q.append(0)

    # starting to build network through width-first searching
    while len(q) > 0:
        # making layers according to information provided by cellist
        node = q.pop(0)
        # print('Evaluater:right now we are processing node %d'%node,', ',cellist[node])
        if cellist[node][0] == 'conv':
            layer = _makeconv(inputs[node], cellist[node], node)
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        elif cellist[node][0] == 'pooling':
            layer = _makepool(inputs[node], cellist[node])
        elif cellist[node][0] == 'dense':
            layer = _makedense(inputs[node], cellist[node])
        else:
            print('WRONG!!!!! Notice that you got a layer type we cant process!', cellist[node][0])
            layer = []

        # update inputs information of the cells below this cell
        for j in graph_part[node]:
            if getinput[j]:  # if this cell already got inputs from other cells precedes it
                # padding
                a = int(layer.shape[1])
                b = int(inputs[j].shape[1])
                pad = abs(a - b)
                if layer.shape[1] > inputs[j].shape[1]:
                    inputs[j] = tf.pad(inputs[j], [[0, 0], [0, pad], [0, pad], [0, 0]])
                if layer.shape[1] < inputs[j].shape[1]:
                    layer = tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])
                inputs[j] = tf.concat([inputs[j], layer], 3)
            else:
                inputs[j] = layer
                getinput[j] = True
            if not inqueue[j]:
                q.append(j)
                inqueue[j] = True

    # softmax
    layer = tf.reshape(layer, [FLAGS.batch_size, -1])
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[layer.shape[-1], NUM_CLASSES],
                                  initializer=tf.contrib.keras.initializers.he_normal())
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        # softmax_linear = tf.nn.softmax(tf.matmul(layer, weights)+ biases, name=scope.name)
        softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
        # tf.add_to_collection('losses', regularizer(weights))
    return softmax_linear


def lossop(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Reshape the labels into a dense Tensor of
    # shape [batch_size, NUM_CLASSES].
    sparse_labels = tf.reshape(labels, [FLAGS.batch_size, 1])
    indices = tf.reshape(tf.range(FLAGS.batch_size), [FLAGS.batch_size, 1])
    concated = tf.concat([indices, sparse_labels], 1)
    dense_labels = tf.sparse_to_dense(concated,
                                      [FLAGS.batch_size, NUM_CLASSES],
                                      1.0, 0.0)

    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=dense_labels,
                                                            name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(graph_part, cell_list):
    # data_dir='C:\\Users\\Jalynn\\PycharmProjects\\CompetitionNAS\\nas\\cifar-10-batches-bin'
    data_dir = os.path.join(os.getcwd(), 'cifar-10-batches-bin')
    # graph_part = [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6], [7]]
    # cell_list = [('conv', 32, 1, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 64, 1, 'relu'),
    #                            ('conv', 128, 3, 'relu'), ('conv', 64, 1, 'relu'), ('conv', 256, 3, 'relu'),
    #                            ('pooling', 'global', 7), ('conv', 192, 1, 'relu'), ('conv', 128, 3, 'relu'),
    #                            ('conv', 128, 1, 'relu'), ('conv', 32, 3, 'relu'), ('conv', 256, 3, 'relu'),
    #                            ('conv', 256, 3, 'leakyrelu'), ('conv', 48, 5, 'relu'), ('conv', 32, 3, 'relu')]
    # # graph_part = [[1], [2], [3], [4], []]
    # # cell_list = [('conv', 64, 5, 'relu'), ('pooling', 'max', 2), ('conv', 64, 5, 'relu'), ('pooling', 'max', 2),
    # #                     ('dense', [120, 84], 'relu')]
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = distorted_inputs(data_dir=data_dir,
                                          batch_size=FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = inference(images, graph_part, cell_list)

        # Calculate loss.
        loss = lossop(logits, labels)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        # Generate moving averages of all losses
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [loss])
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.GradientDescentOptimizer(lr)
            grads = opt.compute_gradients(loss)
        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size)
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        return precision


def main(argv=None):  # pylint: disable=unused-argument
    lenet = NetworkUnit()
    lenet.graph_part = [[1], [2], [3], [4], []]
    lenet.cell_list = [[('conv', 64, 5, 'relu'), ('pooling', 'max', 2), ('conv', 64, 5, 'relu'), ('pooling', 'max', 2),
                        ('dense', [120, 84], 'relu')]]
    best_network = NetworkUnit()
    best_network.graph_part = [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [], [11], [12], [13], [6], [7]]
    best_network.cell_list = [[('conv', 32, 1, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 64, 1, 'relu'),
                               ('conv', 128, 3, 'relu'), ('conv', 64, 1, 'relu'), ('conv', 256, 3, 'relu'),
                               ('pooling', 'global', 7), ('conv', 192, 1, 'relu'), ('conv', 128, 3, 'relu'),
                               ('conv', 128, 1, 'relu'), ('conv', 32, 3, 'relu'), ('conv', 256, 3, 'relu'),
                               ('conv', 256, 3, 'leakyrelu'), ('conv', 48, 5, 'relu'), ('conv', 32, 3, 'relu')]]
    vgg16 = NetworkUnit()
    vgg16.graph_part = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17],
                        [18], []]
    vgg16.cell_list = [
        [('conv', 64, 3, 'relu'), ('conv', 64, 3, 'relu'), ('pooling', 'max', 2), ('conv', 128, 3, 'relu'),
         ('conv', 128, 3, 'relu'), ('pooling', 'max', 2), ('conv', 256, 3, 'relu'),
         ('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('pooling', 'max', 2),
         ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
         ('pooling', 'max', 2), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
         ('conv', 512, 3, 'relu'), ('pooling', 'max', 2), ('dense', [4096, 4096, 1000], 'relu')]]

    network_list = []
    network_list.append(lenet)
    network_list.append(best_network)
    network_list.append(vgg16)
    a = []

    for network in network_list:
        a.append(train(network.graph_part, network.cell_list))

    print(a)


if __name__ == '__main__':
    tf.app.run()
