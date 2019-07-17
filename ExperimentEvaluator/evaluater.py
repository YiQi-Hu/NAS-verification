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
import matplotlib as mpl
import matplotlib.pyplot as plt

# from .cifar10_input import inputs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

path = 'C:\\Users\\Jalynn\\Desktop'  # os.getcwd()  # + '/../'

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
num_examples = 10000
batch_size = 128
log_device_placement = False
epoch = 1
# If a model is trained with multiple GPU's prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'
num_gpu = 2


class Evaluator:
    def __init__(self):

        self.log = "****************"
        self.dtrain = Dataset()
        self.dvalid = Dataset()
        self.dataset = Dataset()
        self.dtrain.feature = self.dtrain.label = []
        self.trainindex = []
        data_dir = os.path.join(path, 'cifar-10-batches-py')
        self.dataset.feature, self.dataset.label, self.dvalid.feature, self.dvalid.label = self.prepare_data()
        # self.dvalid.feature, self.dvalid.label = self._load_batch(os.path.join(data_dir, 'test_batch.bin'))
        # self.dataset.feature, self.dataset.label = self._load_data(data_dir)
        self.leftindex = range(self.dataset.label.shape[0])
        # ind = list(range(10000))
        # random.shuffle(ind)
        # self.dvalid.feature = self.dvalid.feature[ind]
        # self.dvalid.label = self.dvalid.label[ind]
        self.train_num = 0
        self.network_num = 0
        self.max_steps = 60000

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_data_one(self, file):
        batch = self.unpickle(file)
        data = batch[b'data']
        labels = batch[b'labels']
        print("Loading %s : %d." % (file, len(data)))
        return data, labels

    def load_data(self, files, data_dir, label_count):
        global image_size, img_channels
        data, labels = self.load_data_one(os.path.join(data_dir, files[0]))
        for f in files[1:]:
            data_n, labels_n = self.load_data_one(os.path.join(data_dir, f))
            data = np.append(data, data_n, axis=0)
            labels = np.append(labels, labels_n, axis=0)
        labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
        data = data.reshape([-1, 3, IMAGE_SIZE, IMAGE_SIZE])
        data = data.transpose([0, 2, 3, 1])
        return data, labels

    def prepare_data(self):
        print("======Loading data======")
        # download_data()
        data_dir = os.path.join(path, 'cifar-10-batches-py')
        # image_dim = image_size * image_size * img_channels
        meta = self.unpickle(data_dir + '/batches.meta')

        print(meta)
        label_names = meta[b'label_names']
        label_count = 10
        train_files = ['data_batch_%d' % d for d in range(1, 6)]
        train_data, train_labels = self.load_data(train_files, data_dir, label_count)
        test_data, test_labels = self.load_data(['test_batch'], data_dir, label_count)

        print("Train data:", np.shape(train_data), np.shape(train_labels))
        print("Test data :", np.shape(test_data), np.shape(test_labels))
        print("======Load finished======")

        # print("======Shuffling data======")
        # indices = np.random.permutation(len(train_data))
        # train_data = train_data[indices]
        # train_labels = train_labels[indices]
        print("======Prepare Finished======")

        return train_data, train_labels, test_data, test_labels


    def _variable_on_cpu(self, name, shape, initializer):
        """Helper to create a Variable stored on CPU memory.

        Args:
          name: name of the variable
          shape: list of ints
          initializer: initializer for Variable

        Returns:
          Variable Tensor
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def batch_norm(self,input):
        # return input
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                             updates_collections=None)

    def _makeconv(self, inputs, hplist, node):
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
            bias = self.batch_norm(tf.nn.bias_add(conv, biases))
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

    def _makepool(self, inputs, hplist):
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

    def _makedense(self, inputs, hplist):
        """Generates dense layers according to information in hplist

        Args:
                   inputs: inputing data.
                   hplist: hyperparameters for building layers
                   node: number of this cell
        Returns:
                   tensor.
        """
        i = 0
        inputs = tf.reshape(inputs, [batch_size, -1])
        for neural_num in hplist[1]:
            with tf.variable_scope('dense' + str(i)) as scope:
                weights = tf.get_variable('weights', shape=[inputs.shape[-1], neural_num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                biases = tf.get_variable('biases', [neural_num], initializer=tf.constant_initializer(0.0))
                if hplist[2] == 'relu':
                    local3 = tf.nn.relu(self.batch_norm(tf.matmul(inputs, weights) + biases), name=scope.name)
                elif hplist[2] == 'tanh':
                    local3 = tf.tanh(self.batch_norm(tf.matmul(inputs, weights) + biases), name=scope.name)
                elif hplist[2] == 'sigmoid':
                    local3 = tf.sigmoid(self.batch_norm(tf.matmul(inputs, weights) + biases), name=scope.name)
                elif hplist[2] == 'identity':
                    local3 = tf.identity(self.batch_norm(tf.matmul(inputs, weights) + biases), name=scope.name)
            inputs = local3
            i += 1
        return inputs

    def _inference(self, images, graph_part, cellist):  # ,regularizer):
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
                layer = self._makeconv(inputs[node], cellist[node], node)
                # layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
            elif cellist[node][0] == 'pooling':
                layer = self._makepool(inputs[node], cellist[node])
            elif cellist[node][0] == 'dense':
                layer = self._makedense(inputs[node], cellist[node])
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
        layer = tf.reshape(layer, [batch_size, -1])
        with tf.variable_scope('softmax_linear') as scope:
            weights = tf.get_variable('weights', shape=[layer.shape[-1], NUM_CLASSES],
                                      initializer=tf.truncated_normal_initializer(stddev=0.04))  # 1 / float(dim)))
            biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
            # softmax_linear = tf.nn.softmax(tf.matmul(layer, weights)+ biases, name=scope.name)
            softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
            # tf.add_to_collection('losses', regularizer(weights))
        return softmax_linear

    def __average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def train_multi_gpu(self, graph_part, cellist):
        with tf.device("/cpu:0"):
            tower_grads = []
            opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE)
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(num_gpu):
                    with tf.device("/gpu:%d" % (i + 2)):
                        with tf.name_scope("tower_%d" % i):
                            images = tf.placeholder(tf.float32, [128, 32, 32, 3])
                            labels = tf.placeholder(tf.int32, [128, ])

                            y = self._inference(images, graph_part, cellist)
                            top_k_op = tf.nn.in_top_k(y, labels, 1)
                            tf.get_variable_scope().reuse_variables()
                            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=labels)
                            cross_entropy_mean = tf.reduce_mean(cross_entropy)
                            tf.add_to_collection('losses', cross_entropy_mean)

                            loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
                            grads = opt.compute_gradients(loss)
                            tower_grads.append(grads)
            grads = self.__average_gradients(tower_grads)
            train_op = opt.apply_gradients(grads)

            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     for i in range(self.max_steps):
            #         st = (i * batch_size) % self.train_num
            #         end = ((i + 1) * batch_size) % self.train_num
            #         if end < st:
            #             end = st + 128
            #             if end > self.train_num:
            #                 st = 0
            #                 end = 128
            #         batch_index = self.trainindex[st:end]
            #         _, loss_value = sess.run([train_op, loss]
            #                                  , feed_dict={images: self.dataset.feature[batch_index],
            #                                               labels: self.dataset.label[batch_index]})
            #         if i % 50 == 0 or i == self.max_steps - 1:
            #             print("After %d training steps, loss on training"
            #                   "batch is %f" % (i, loss_value))
            #     '''print("Done")
            #     print("Testing Accuracy:",
            #           np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size],
            #                                                  Y: mnist.test.labels[i:i + batch_size]}) for i in
            #                    range(0, len(mnist.test.images), batch_size)]))'''
        return train_op, loss, top_k_op,images,labels

    def train(self, graph_part, cellist):
            global_step = tf.Variable(0, trainable=False)
            # Get images and labels.
            # input_queue = tf.train.slice_input_producer(
            #     [self.dataset.feature[self.trainindex], self.dataset.label[self.trainindex]], shuffle=True)
            # input_queue=tf.image.per_image_standardization(input_queue)
            # images, trainlabel = tf.train.batch(input_queue, batch_size=batch_size, num_threads=16, capacity=500,
            #                                     allow_smaller_final_batch=False)
            # images, labels = inputs(eval_data=False, data_dir=os.path.join(path, 'cifar-10-batches-bin'),
            #                                       batch_size=batch_size,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN=self.train_num)
            images = tf.placeholder(tf.float32, [128, 32, 32, 3])
            labels = tf.placeholder(tf.int32, [128, 10])

            y = self._inference(images, graph_part, cellist)

            global_steps = tf.Variable(0, trainable=False)
            variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_steps)
            variable_average_op = variable_average.apply(
                tf.trainable_variables())
            labels = tf.cast(labels, tf.int64)
            top_k_op = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
            # top_k_op = tf.nn.in_top_k(y, labels, 1)  # Calculate predictions
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', cross_entropy_mean)

            loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
            train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_steps)

            with tf.control_dependencies([train_step, variable_average_op]):
                # with tf.control_dependencies([train_step]):
                train_op = tf.no_op(name='train')

            return train_op, loss, top_k_op,images,labels

    def evaluate(self, network):  #, network_index, start_time):
        '''Train'''
        # print('Evaluater:start training')
        cellist = network.cell_list[-1]
        # cellist.append(('dense', [384, 192], 'relu'))
        self.network_num += 1
        tf.reset_default_graph()
        print('*********network %d config***********' % (self.network_num))
        print(cellist)
        print(network.graph_part)
        print('***********************************')

        # try:
        # precision, time_cost = self.train(network.graph_part, cellist, network_index, start_time)  # start training

        with tf.Graph().as_default():
            train_op, loss, top_k_op,images,labels = self.train(network.graph_part, cellist)

            sess = tf.InteractiveSession()

            tf.global_variables_initializer().run()

            for i in range(self.max_steps):
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
                # print(np.argmax(yy,1)[:10])
                # print('zhenshi')
                # print(label[:10])
                #
                #
                if i % 50 == 0 or i == self.max_steps - 1:
                    print("After %d training steps, loss on training"
                          "batch is %f" % (i, loss_value))
            num_iter = int(math.ceil(num_examples / batch_size)) - 1
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * batch_size
            step = 0
            while step < num_iter:
                predictions = sess.run([top_k_op]
                                       , feed_dict={
                        images: self.dvalid.feature[step * batch_size:(step + 1) * batch_size],
                        labels: self.dvalid.label[step * batch_size:(step + 1) * batch_size]})
                true_count += np.sum(predictions)
                step += 1
            precision = true_count / total_sample_count  # Compute precision.
            print('%s: precision network@ %d = %.3f, training time:%s' % (
                datetime.now(), network_index, precision, time.time() - start_time))

        return precision, time.time() - start_time

    def add_data(self, add_num=0):

        if self.train_num + add_num > 50000:
            add_num = 50000 - self.train_num
            self.train_num = 50000
        else:
            self.train_num += add_num
        # print('************A NEW ROUND************')
        self.network_num = 0
        self.max_steps = int(self.train_num / batch_size) * epoch

        # print('Evaluater: Adding data')
        if add_num:
            catag = 10
            for cat in range(catag):
                # num_train_samples = self.dataset.label.shape[0]
                print(type(self.dataset.label))
                print(np.argmax(self.dataset.label[0]))
                print(type(self.leftindex))
                cata_index = [i for i in self.leftindex if np.argmax(self.dataset.label[i]) == cat]
                selected = random.sample(cata_index, int(add_num / catag))
                self.trainindex += selected
                self.leftindex = [i for i in self.leftindex if not (i in selected)]
                random.shuffle(self.trainindex)
        return 0

    def get_train_size(self):
        return self.train_num

    def _load_batch(self, filename, num=10000):
        """ load single batch of cifar """
        bytestream = open(filename, "rb")
        buf = bytestream.read(num * (1 + 32 * 32 * 3))
        bytestream.close()
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num, 1 + 32 * 32 * 3)
        data = np.float32(data)
        labels_images = np.hsplit(data, [1])
        labels = labels_images[0].reshape(num)
        labels = np.int64(labels)
        images = labels_images[1].reshape(num, 32, 32, 3)
        images_mean = np.mean(images, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
        stddev = np.std(images, axis=(1, 2, 3)).reshape((-1, 1, 1, 1))
        images = (images - images_mean) / stddev
        return images, labels

    def _load_data(self, ROOT):
        """ load all of cifar """
        print('Evaluater: loading data')
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d.bin' % (b,))
            X, Y = self._load_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)  # 使变成行向量
        Ytr = np.concatenate(ys)
        del X, Y
        # print('Evaluater:data loaded')
        return Xtr, Ytr

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
            train_op, loss, top_k_op,images,labels = self.train(graph_part,cellist)

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
    # network_list.append(lenet)
    # network_list.append(best_network)
    network_list.append(vgg16)
    eval = Evaluator()
    eval.add_data(50000)
    eval = eval.evaluate(vgg16)

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
