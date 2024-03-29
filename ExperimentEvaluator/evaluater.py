from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from base import NetworkUnit
from base import Dataset
from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf
import random
import pickle

# from .cifar10_input import inputs

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = '/data/data'  # + '/../'

# FLAGS = tf.app.flags.FLAGS

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = 32
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# Constants describing the training process.
INITIAL_LEARNING_RATE = 0.001  # Initial learning rate.
MOVING_AVERAGE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
batch_size = 250
epoch = 500
weight_decay = 0.0003
momentum_rate = 0.9
log_save_path = './logs'
model_save_path = './model/'


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


def load_data(files, data_dir, label_count):
    data, labels = load_data_one(os.path.join(data_dir, files[0]))
    for f in files[1:]:
        data_n, labels_n = load_data_one(os.path.join(data_dir, f))
        data = np.append(data, data_n, axis=0)
        labels = np.append(labels, labels_n, axis=0)
    labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
    data = data.reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE])
    data = data.transpose([0, 2, 3, 1])
    return data, labels


def prepare_data():
    print("======Loading data======")
    # download_data()
    data_dir = os.path.join(path, 'cifar-10-batches-py')
    # image_dim = IMAGE_SIZE * image_size * img_channels
    meta = unpickle(data_dir + '/batches.meta')

    print(meta)
    label_names = meta[b'label_names']
    label_count = 10
    train_files = ['data_batch_%d' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    return train_data, train_labels, test_data, test_labels


def cut_out(x_train):
    for i in range(len(x_train)):
        cut_size = random.randint(0, int(IMAGE_SIZE/2))
        x = random.randint(0, IMAGE_SIZE - cut_size)
        x_train[i,x:x+cut_size,x:x+cut_size,:]=0
    return x_train

def batch_norm(input):
    # return input
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        updates_collections=None)


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def data_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:, :, :, 0] = (x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])
    x_train=cut_out(x_train)

    x_test[:, :, :, 0] = (x_test[:, :, :, 0] - np.mean(x_test[:, :, :, 0])) / np.std(x_test[:, :, :, 0])
    x_test[:, :, :, 1] = (x_test[:, :, :, 1] - np.mean(x_test[:, :, :, 1])) / np.std(x_test[:, :, :, 1])
    x_test[:, :, :, 2] = (x_test[:, :, :, 2] - np.mean(x_test[:, :, :, 2])) / np.std(x_test[:, :, :, 2])

    return x_train, x_test


def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32, 32], 4)
    return batch


def learning_rate_schedule(epoch_num):
    if epoch_num < 81:
        return 0.1
    elif epoch_num < 121:
        return 0.01
    elif epoch_num<151:
        return 0.001
    else:
        return  0.001*pow(0.98,int((epoch_num-151)/5))


def makeconv(inputs, hplist, node):
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
        bias = batch_norm(tf.nn.bias_add(conv, biases))
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


def makepool(inputs, hplist):
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


def makedense(inputs, hplist):
    """Generates dense layers according to information in hplist

    Args:
               inputs: inputing data.
               hplist: hyperparameters for building layers
               node: number of this cell
    Returns:
               tensor.
    """
    i = 0
    inputs = tf.reshape(inputs, [-1, 2 * 2 * 512])

    for neural_num in hplist[1]:
        with tf.variable_scope('dense' + str(i)) as scope:
            weights = tf.get_variable('weights', shape=[inputs.shape[-1], neural_num],
                                      initializer=tf.contrib.keras.initializers.he_normal())
            # weight = tf.multiply(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
            # tf.add_to_collection('losses', weight)
            biases = tf.get_variable('biases', [neural_num], initializer=tf.constant_initializer(0.0))
            if hplist[2] == 'relu':
                local3 = tf.nn.relu(batch_norm(tf.matmul(inputs, weights) + biases), name=scope.name)
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
            layer = makeconv(inputs[node], cellist[node], node)
            layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        elif cellist[node][0] == 'pooling':
            layer = makepool(inputs[node], cellist[node])
        elif cellist[node][0] == 'dense':
            layer = makedense(inputs[node], cellist[node])
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
    # layer=tf.reduce_mean(layer,[1,2],name="gpool",keep_dims=True)
    layer = tf.reshape(layer, [batch_size, -1])
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[layer.shape[-1], NUM_CLASSES],
                                  initializer=tf.contrib.keras.initializers.he_normal())  # 1 / float(dim)))
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        # softmax_linear = tf.nn.softmax(tf.matmul(layer, weights)+ biases, name=scope.name)
        softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
        # tf.add_to_collection('losses', regularizer(weights))
    return softmax_linear


class Evaluator:
    def __init__(self):
        self.dtrain = Dataset()
        self.dvalid = Dataset()
        self.dataset = Dataset()
        self.dtrain.feature = []
        self.dtrain.label = []
        self.trainindex = []
        self.dataset.feature, self.dataset.label, self.dvalid.feature, self.dvalid.label = prepare_data()
        self.dataset.feature, self.dvalid.feature = data_preprocessing(self.dataset.feature, self.dvalid.feature)
        self.leftindex = range(self.dataset.label.shape[0])
        self.train_num = 0
        self.network_num = 0
        self.max_steps = 0

    def evaluate(self, network=None):
        # define placeholder x, y_ , keep_prob, learning_rate
        x = tf.placeholder(tf.float32, [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3])
        y_ = tf.placeholder(tf.int64, [batch_size, NUM_CLASSES])
        learning_rate = tf.placeholder(tf.float32)
        train_flag = tf.placeholder(tf.bool)

        output = inference(x, network.graph_part, network.cell_list[-1])

        # output  = tf.reshape(output,[-1,10])

        # loss function: cross_entropy
        # train_step: training operation
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True). \
            minimize(cross_entropy + l2 * weight_decay)

        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # initial an saver to save model
        saver = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)

            # epoch = 164
            # make sure [bath_size * iteration = data_set_number]

            for ep in range(1, epoch + 1):
                lr = learning_rate_schedule(ep)
                pre_index = 0
                train_acc = 0.0
                train_loss = 0.0
                start_time = time.time()

                print("\n epoch %d/%d:" % (ep, epoch))

                for it in range(1, self.max_steps + 1):
                    batch_x = self.dtrain.feature[pre_index:pre_index + batch_size]
                    batch_y = self.dtrain.label[pre_index:pre_index + batch_size]

                    batch_new = data_augmentation(batch_x)
                    _, batch_loss = sess.run([train_step, cross_entropy],
                                             feed_dict={x: batch_new, y_: batch_y,
                                                        learning_rate: lr, train_flag: True})
                    train_loss += batch_loss

                    _, batch_loss = sess.run([train_step, cross_entropy],
                                             feed_dict={x: batch_x, y_: batch_y,
                                                        learning_rate: lr, train_flag: True})
                    batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, train_flag: True})
                    batch_acc_new=accuracy.eval(feed_dict={x: batch_new, y_: batch_y, train_flag: True})

                    train_loss += batch_loss
                    train_acc += batch_acc
                    train_acc += batch_acc_new
                    pre_index += batch_size

                    if it == self.max_steps:
                        train_loss /= self.max_steps
                        train_acc /= self.max_steps

                        loss_, acc_ = sess.run([cross_entropy, accuracy],
                                               feed_dict={x: batch_x, y_: batch_y, train_flag: True})
                        train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                          tf.Summary.Value(tag="train_accuracy",
                                                                           simple_value=train_acc)])

                        val_acc = 0.0
                        val_loss = 0.0
                        pre_index = 0
                        add = 250
                        for ite in range(40):
                            batch_x = self.dvalid.feature[pre_index:pre_index + add]
                            batch_y = self.dvalid.label[pre_index:pre_index + add]
                            pre_index = pre_index + add
                            loss_, acc_ = sess.run([cross_entropy, accuracy],
                                                   feed_dict={x: batch_x, y_: batch_y, train_flag: False})
                            val_loss += loss_ / 40.0
                            val_acc += acc_ / 40.0
                        test_summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=val_loss),
                                                         tf.Summary.Value(tag="test_accuracy", simple_value=val_acc)])

                        summary_writer.add_summary(train_summary, ep)
                        summary_writer.add_summary(test_summary, ep)
                        summary_writer.flush()

                        print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                              "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                              % (
                                  it, self.max_steps, int(time.time() - start_time), train_loss, train_acc, val_loss,
                                  val_acc))
                    # else:
                    #     print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                    #           % (it, self.max_steps, train_loss / it, train_acc / it))

            save_path = saver.save(sess, model_save_path)
            print("Model saved in file: %s" % save_path)

    def add_data(self, add_num=0):

        if self.train_num + add_num > 50000:
            add_num = 50000 - self.train_num
            self.train_num = 50000
            print('Warning! ')
        else:
            self.train_num += add_num
        # print('************A NEW ROUND************')
        self.network_num = 0
        self.max_steps = int(self.train_num / batch_size)

        # print('Evaluater: Adding data')
        if add_num:
            catag = 10
            for cat in range(catag):
                # num_train_samples = self.dataset.label.shape[0]
                cata_index = [i for i in self.leftindex if np.argmax(self.dataset.label[i]) == cat]
                if len(cata_index) < int(add_num / catag):
                    selected = cata_index
                else:
                    selected = random.sample(cata_index, int(add_num / catag))
                self.trainindex += selected
                self.leftindex = [i for i in self.leftindex if not (i in selected)]
                random.shuffle(self.trainindex)
                self.dtrain.feature = self.dataset.feature[self.trainindex]
                self.dtrain.label = self.dataset.label[self.trainindex]
        return 0

    def get_train_size(self):
        return self.train_num

    def exp(self, network, epoch):
        cellist = network.cell_list[-1]
        graph_part = network.graph_part
        precision_list = []
        tf.reset_default_graph()
        with tf.Graph().as_default():
            # images = tf.placeholder(tf.float32, [128, 32, 32, 3])
            # labels = tf.placeholder(tf.int32, [128, ])
            #
            # y = self._inference(images, graph_part, cellist)
            #
            # global_steps = tf.Variable(0, trainable=False)
            # variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_steps)
            # variable_average_op = variable_average.apply(tf.trainable_variables())
            # labels = tf.cast(labels, tf.int64)
            # top_k_op = tf.nn.in_top_k(y, labels, 1)  # Calculate predictions
            # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels)
            # cross_entropy_mean = tf.reduce_mean(cross_entropy)
            # tf.add_to_collection('losses', cross_entropy_mean)
            #
            # loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
            # train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_steps)
            #
            # with tf.control_dependencies([train_step, variable_average_op]):
            #     train_op = tf.no_op(name='train')
            train_op, loss, top_k_op, images, labels = self.train(graph_part, cellist)

            sess = tf.InteractiveSession()

            tf.global_variables_initializer().run()

            one_epoch = int(self.train_num / batch_size)
            max_steps = epoch * one_epoch
            num_iter = int(math.ceil(num_examples / batch_size)) - 1
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * batch_size
            for i in range(max_steps):
                st = (i * batch_size) % self.train_num
                end = ((i + 1) * batch_size) % self.train_num
                if end < st:
                    end = st + 128
                    if end > self.train_num:
                        st = 0
                        end = 128

                batch_index = self.trainindex[st:end]
                _, loss_value = sess.run([train_op, loss]
                                         , feed_dict={images: self.dataset.feature[batch_index],
                                                      labels: self.dataset.label[batch_index]})
                if (i + 1) % one_epoch == 0:
                    print("After %d training steps, loss on training batch is %f" % ((i + 1) / one_epoch, loss_value))
                    step = 0
                    true_count = 0
                    while step < num_iter:
                        predictions = sess.run([top_k_op], feed_dict={
                            images: self.dvalid.feature[step * batch_size:(step + 1) * batch_size],
                            labels: self.dvalid.label[step * batch_size:(step + 1) * batch_size]})
                        true_count += np.sum(predictions)
                        step += 1
                    precision = true_count / total_sample_count  # Compute precision.
                    precision_list.append(precision)
                    print('%s: precision = %.3f' % (datetime.now(), precision))

            sess.close()
            return precision_list


if __name__ == '__main__':
    subsample_ratio_range = np.arange(0.05, 1, 0.05)
    epoch_range = range(0, 50)
    # for the experiment of data ratio and epoch
    lenet = NetworkUnit()
    lenet.graph_part = [[1], [2], [3], [4], []]
    lenet.cell_list = [[('conv', 64, 5, 'relu'), ('pooling', 'max', 2), ('conv', 64, 5, 'relu'), ('pooling', 'max', 2),
                        ('dense', [120, 84], 'relu')]]
    best_network = NetworkUnit()
    best_network.graph_part = [[1, 10], [2, 14], [3], [4], [5], [6], [7], [8], [9], [15], [11], [12], [13], [6], [7],
                               []]
    best_network.cell_list = [[('conv', 32, 1, 'relu'), ('conv', 48, 3, 'relu'), ('conv', 64, 1, 'relu'),
                               ('conv', 128, 3, 'relu'), ('conv', 64, 1, 'relu'), ('conv', 256, 3, 'relu'),
                               ('pooling', 'global', 7), ('conv', 192, 1, 'relu'), ('conv', 128, 3, 'relu'),
                               ('conv', 128, 1, 'relu'), ('conv', 32, 3, 'relu'), ('conv', 256, 3, 'relu'),
                               ('conv', 256, 3, 'leakyrelu'), ('conv', 48, 5, 'relu'), ('conv', 32, 3, 'relu'),
                               ('dense', [4096, 4096, 1000], 'relu')]]
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
    # network_list.append(lenet)
    # network_list.append(best_network)
    network_list.append(vgg16)
    network=NetworkUnit()
    network.cell_list=[[('conv', 48, 3, 'relu'), ('conv', 128, 5, 'relu'), ('pooling', 'avg', 3), ('conv', 512, 5, 'relu'),
	('pooling', 'avg', 7),('conv', 192, 5, 'relu'), ('pooling', 'avg', 7), ('pooling', 'avg', 8) , ('pooling', 'max', 6),
	('pooling', 'max', 8),('conv', 48, 3, 'leakyrelu'),('pooling', 'avg', 6), ('pooling', 'max', 8), ('pooling', 'avg', 3),('pooling', 'avg', 8),]]
    network.graph_part=[[1,5,7,9,12,14],[2,5,10,12,14],[3, 4,5,6,7,8,9,11], [4, 6, 7,12,13], [5,10,12,13],
	[6,7,8,9,10,11,12,13], [7,10,11,12,14],[8,9,13,14],[9,11,13,14], [10, 11, 12, 13, 14],
	[11, 14],[12,13,14], [13], [14],[]]
    eval = Evaluator()
    eval.add_data(50000)
    # eval = eval.evaluate(best_network)
    eval.evaluate(network)

    # for network in network_list:
    #     eval = Evaluator()
    #     g = []
    #     for r in subsample_ratio_range:
    #         eval.add_data(int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * r))
    #         a = eval.exp(network, 50)
    #         g.append(a)
    #     print(np.shape(g))
    #     g = np.array(g)
    #     np.savetxt("2res.txt", g, "%.3f")
