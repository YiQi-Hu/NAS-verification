import os
import pickle
import random
import sys
import time
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from base import Cell, NetworkItem
from info_str import NAS_CONFIG
from utils import NAS_LOG
from densenet import densenet201


class DataSet:

    def __init__(self):
        self.IMAGE_SIZE = 32
        self.NUM_CLASSES = 100
        self.NUM_EXAMPLES_FOR_TRAIN = 40000
        self.NUM_EXAMPLES_FOR_EVAL = 10000
        self.task = "cifar-100"
        self.data_path = '/data/data'
        return

    def inputs(self):
        print("======Loading data======")
        if self.task == 'cifar-10':
            test_files = ['test_batch']
            train_files = ['data_batch_%d' % d for d in range(1, 6)]
        else:
            train_files = ['train']
            test_files = ['test']
        train_data, train_label = self._load(train_files)
        train_data, train_label, valid_data, valid_label = self._split(
            train_data, train_label)
        test_data, test_label = self._load(test_files)
        print("======Data Process Done======")
        return train_data, train_label, valid_data, valid_label, test_data, test_label

    def _load_one(self, file):
        with open(file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        data = batch[b'data']
        label = batch[b'labels'] if self.task == 'cifar-10' else batch[b'fine_labels']
        return data, label

    def _load(self, files):
        file_name = 'cifar-10-batches-py' if self.task == 'cifar-10' else 'cifar-100-python'
        data_dir = os.path.join(self.data_path, file_name)
        data, label = self._load_one(os.path.join(data_dir, files[0]))
        for f in files[1:]:
            batch_data, batch_label = self._load_one(os.path.join(data_dir, f))
            data = np.append(data, batch_data, axis=0)
            label = np.append(label, batch_label, axis=0)
        label = np.array([[float(i == label)
                           for i in range(self.NUM_CLASSES)] for label in label])
        data = data.reshape([-1, 3, self.IMAGE_SIZE, self.IMAGE_SIZE])
        data = data.transpose([0, 2, 3, 1])
        # pre-process
        data = self._normalize(data)

        return data, label

    def _split(self, data, label):
        # shuffle
        index = [i for i in range(len(data))]
        random.shuffle(index)
        data = data[index]
        label = label[index]
        return data[:self.NUM_EXAMPLES_FOR_TRAIN], label[:self.NUM_EXAMPLES_FOR_TRAIN], \
               data[self.NUM_EXAMPLES_FOR_TRAIN:self.NUM_EXAMPLES_FOR_TRAIN + self.NUM_EXAMPLES_FOR_EVAL], \
               label[self.NUM_EXAMPLES_FOR_TRAIN:self.NUM_EXAMPLES_FOR_TRAIN +
                                                 self.NUM_EXAMPLES_FOR_EVAL]

    def _normalize(self, x_train):
        x_train = x_train.astype('float32')

        x_train[:, :, :, 0] = (
                                      x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
        x_train[:, :, :, 1] = (
                                      x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
        x_train[:, :, :, 2] = (
                                      x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

        return x_train

    def process(self, x):
        x = self._random_flip_leftright(x)
        x = self._random_crop(x, [32, 32], 4)
        x = self._cutout(x)
        return x

    def _random_crop(self, batch, crop_shape, padding=None):
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
        return np.array(new_batch)

    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

    def _cutout(self, x):
        for i in range(len(x)):
            cut_size = random.randint(0, self.IMAGE_SIZE // 2)
            s = random.randint(0, self.IMAGE_SIZE - cut_size)
            x[i, s:s + cut_size, s:s + cut_size, :] = 0
        return x


class Evaluator:
    def __init__(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        # Global constants describing the CIFAR-10 data set.
        self.IMAGE_SIZE = 32
        self.NUM_CLASSES = 100
        self.NUM_EXAMPLES_FOR_TRAIN = 40000
        self.NUM_EXAMPLES_FOR_EVAL = 10000
        # Constants describing the training process.
        # Initial learning rate.
        self.INITIAL_LEARNING_RATE = 0.025
        # Epochs after which learning rate decays
        self.NUM_EPOCHS_PER_DECAY = 80.0
        # Learning rate decay factor.
        self.LEARNING_RATE_DECAY_FACTOR = 0.1
        self.MOVING_AVERAGE_DECAY = 0.98
        self.batch_size = 200
        self.weight_decay = 0.0003
        self.momentum_rate = 0.9
        self.model_path = './model'
        self.train_num = 0
        self.block_num = 0
        self.log = ''
        # self.train_data, self.train_label, self.valid_data, self.valid_label, \
        # self.test_data, self.test_label = DataSet().inputs()
        self.epoch = 1

    def set_epoch(self, e):
        self.epoch = e
        return

    def get_epoch(self):
        return self.epoch

    def _inference(self, images, graph_part, cell_list, train_flag):
        '''Method for recovering the network model provided by graph_part and cellist.
        Args:
          images: Images returned from Dataset() or inputs().
          graph_part: The topology structure of th network given by adjacency table
          cellist:
        Returns:
          Logits.'''
        topo_order = self._toposort(graph_part)
        nodelen = len(graph_part)
        # input list for every cell in network
        inputs = [images for _ in range(nodelen)]
        # bool list for whether this cell has already got input or not
        getinput = [False for _ in range(nodelen)]
        getinput[0] = True

        for node in topo_order:
            layer = self._make_layer(inputs[node], cell_list[node], node, train_flag)

            # update inputs information of the cells below this cell
            for j in graph_part[node]:
                if getinput[j]:  # if this cell already got inputs from other cells precedes it
                    inputs[j] = self._pad(inputs[j], layer)
                else:
                    inputs[j] = layer
                    getinput[j] = True

        # give last layer a name
        last_layer = tf.identity(layer, name="last_layer" + str(self.block_num))
        return last_layer

    def _toposort(self, graph):
        node_len = len(graph)
        in_degrees = dict((u, 0) for u in range(node_len))
        for u in range(node_len):
            for v in graph[u]:
                in_degrees[v] += 1
        queue = [u for u in range(node_len) if in_degrees[u] == 0]
        result = []
        while queue:
            u = queue.pop()
            result.append(u)
            for v in graph[u]:
                in_degrees[v] -= 1
                if in_degrees[v] == 0:
                    queue.append(v)
        return result

    def _make_layer(self, inputs, cell, node, train_flag):
        '''Method for constructing and calculating cell in tensorflow
        Args:
                  cell: Class Cell(), hyper parameters for building this layer
        Returns:
                  layer: tensor.'''
        if cell.type == 'conv':
            layer = self._makeconv(
                inputs, cell, node, train_flag)
        elif cell.type == 'pooling':
            layer = self._makepool(inputs, cell)
        elif cell.type == 'sep_conv':
            layer = self._makeconv(
                inputs, cell, node, train_flag)
        else:
            layer = tf.identity(inputs)

        return layer

    def _makeconv(self, inputs, hplist, node, train_flag):
        """Generates a convolutional layer according to information in hplist
        Args:
        inputs: inputing data.
        hplist: hyperparameters for building this layer
        node: number of this cell
        Returns:
        tensor.
        """
        with tf.variable_scope('block' + str(self.block_num) + 'conv' + str(node)) as scope:
            inputdim = inputs.shape[3]
            kernel = self._get_variable(
                'weights', shape=[hplist.kernel_size, hplist.kernel_size, inputdim, hplist.filter_size])
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            bn = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            conv_layer = self._activation_layer(hplist.activation, bn, scope)

        return conv_layer

    def _makesep_conv(self, inputs, hplist, node, train_flag):
        with tf.variable_scope('block' + str(self.block_num) + 'conv' + str(node)) as scope:
            inputdim = inputs.shape[3]
            kernel = self._get_variable(
                'weights', shape=[hplist.kernel_size, hplist.kernel_size, inputdim, 1])
            pfilter = self._get_variable(
                'pointwise_filter', [1, 1, inputdim, hplist.filter_size])
            conv = tf.nn.separable_conv2d(
                inputs, kernel, pfilter, strides=[1, 1, 1, 1], padding='SAME')
            biases = self._get_variable('biases', hplist.filter_size)
            bn = self._batch_norm(tf.nn.bias_add(conv, biases), train_flag)
            conv_layer = self._activation_layer(hplist.activation, bn, scope)

        return conv_layer

    def _batch_norm(self, input, train_flag):
        return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                            updates_collections=None, is_training=train_flag)

    def _get_variable(self, name, shape):
        if name == "weights":
            ini = tf.contrib.keras.initializers.he_normal()
        else:
            ini = tf.constant_initializer(0.0)
        return tf.get_variable(name, shape, initializer=ini)

    def _activation_layer(self, type, inputs, scope):
        if type == 'relu':
            layer = tf.nn.relu(inputs, name=scope.name)
        elif type == 'relu6':
            layer = tf.nn.relu6(inputs, name=scope.name)
        elif type == 'tanh':
            layer = tf.tanh(inputs, name=scope.name)
        elif type == 'sigmoid':
            layer = tf.sigmoid(inputs, name=scope.name)
        elif type == 'leakyrelu':
            layer = tf.nn.leaky_relu(inputs, name=scope.name)
        else:
            layer = tf.identity(inputs, name=scope.name)

        return layer

    def _makepool(self, inputs, hplist):
        """Generates a pooling layer according to information in hplist
        Args:
            inputs: inputing data.
            hplist: hyperparameters for building this layer
        Returns:
            tensor.
        """
        if hplist.ptype == 'avg':
            return tf.nn.avg_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.ptype == 'max':
            return tf.nn.max_pool(inputs, ksize=[1, hplist.kernel_size, hplist.kernel_size, 1],
                                  strides=[1, hplist.kernel_size, hplist.kernel_size, 1], padding='SAME')
        elif hplist.ptype == 'global':
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
        inputs = tf.reshape(inputs, [self.batch_size, -1])

        for i, neural_num in enumerate(hplist[1]):
            with tf.variable_scope('block' + str(self.block_num) + 'dense' + str(i)) as scope:
                weights = self._get_variable('weights', shape=[inputs.shape[-1], neural_num])
                biases = self._get_variable('biases', [neural_num])
                mul = tf.matmul(inputs, weights) + biases
                if neural_num == self.NUM_CLASSES:
                    local3 = self._activation_layer('', mul, scope)
                else:
                    local3 = self._activation_layer(hplist[2], mul, scope)
            inputs = local3
        return inputs

    def _pad(self, inputs, layer):
        # padding
        a = int(layer.shape[1])
        b = int(inputs.shape[1])
        pad = abs(a - b)
        if layer.shape[1] > inputs.shape[1]:
            tmp = tf.pad(inputs, [[0, 0], [0, pad], [0, pad], [0, 0]])
            inputs = tf.concat([tmp, layer], 3)
        elif layer.shape[1] < inputs.shape[1]:
            tmp = tf.pad(layer, [[0, 0], [0, pad], [0, pad], [0, 0]])
            inputs = tf.concat([inputs, tmp], 3)
        else:
            inputs = tf.concat([inputs, layer], 3)

        return inputs

    def evaluate(self, network, pre_block=[], is_bestNN=False, update_pre_weight=False):
        '''Method for evaluate the given network.
        Args:
            network: NetworkItem()
            pre_block: The pre-block structure, every block has two parts: graph_part and cell_list of this block.
            is_bestNN: Symbol for indicating whether the evaluating network is the best network of this round, default False.
            update_pre_weight: Symbol for indicating whether to update previous blocks' weight, default by False.
        Returns:
            Accuracy'''
        assert self.train_num >= self.batch_size
        tf.reset_default_graph()
        self.block_num = len(pre_block) * NAS_CONFIG['nas_main']['repeat_search']

        # print("-" * 20, network.id, "-" * 20)
        # print(network.graph, network.cell_list, Network.pre_block)
        self.log = "-" * 20 + str(network.id) + "-" * 20 + '\n'
        for block in pre_block:
            self.log = self.log + str(block.graph) + str(block.cell_list) + '\n'
        self.log = self.log + str(network.graph) + str(network.cell_list) + '\n'

        with tf.Session() as sess:
            data_x, data_y, block_input, train_flag = self._get_input(sess, pre_block, update_pre_weight)

            for _ in range(NAS_CONFIG['nas_main']['repeat_search'] - 1):
                graph_full = network.graph + [[]]
                cell_list = network.cell_list + [Cell('pooling', 'max', 1)]
                block_input = self._inference(block_input, graph_full, cell_list, train_flag)
                self.block_num += 1
            # a pooling layer for last repeat block
            graph_full = network.graph + [[]]
            cell_list = network.cell_list + [Cell('pooling', 'max', 2)]
            logits = self._inference(block_input, graph_full, cell_list, train_flag)

            logits = tf.nn.dropout(logits, keep_prob=1.0)
            logits = self._makedense(logits, ('', [self.NUM_CLASSES], ''))

            precision, saver, log = self._eval(sess, data_x, data_y, logits, train_flag)
            self.log += log

            if is_bestNN:  # save model
                saver.save(sess, os.path.join(
                    self.model_path, 'model' + str(network.id)))

        NAS_LOG << ('eva', self.log)
        return precision

    def retrain(self, pre_block):
        tf.reset_default_graph()
        assert self.train_num >= self.batch_size
        self.block_num = len(pre_block) * NAS_CONFIG['nas_main']['repeat_search'] + 1

        retrain_log = "-" * 20 + "retrain" + "-" * 20 + '\n'

        with tf.Session() as sess:
            data_x, labels, logits, train_flag = self._get_input(sess, [])

            # for block in pre_block:
            #     graph = block.graph + [[]]
            #     cell_list = []
            #     for cell in block.cell_list:
            #         if cell.type == 'conv':
            #             cell_list.append(
            #                 Cell(cell.type, cell.filter_size * 2, cell.kernel_size, cell.activation))
            #         else:
            #             cell_list.append(cell)
            #     cell_list.append(Cell('pooling', 'max', 1))
            #     # repeat search
            #     for _ in range(NAS_CONFIG['nas_main']['repeat_search'] - 1):
            #         retrain_log = retrain_log + str(graph) + str(cell_list) + '\n'
            #         logits = self._inference(logits, graph, cell_list, train_flag)
            #         self.block_num += 1
            #     # add pooling layer only in last repeat block
            #     cell_list[-1] = Cell('pooling', 'max', 2)
            #     retrain_log = retrain_log + str(graph) + str(cell_list) + '\n'
            logits = densenet201(data_x, reuse=False, is_training=True, kernel_initializer=tf.orthogonal_initializer())

            # logits = tf.nn.dropout(logits, keep_prob=1.0)
            # softmax
            # logits = self._makedense(logits, ('', [256, self.NUM_CLASSES], 'relu'))
            retrain = True

            global_step = tf.Variable(
                0, trainable=False, name='global_step' + str(self.block_num))
            accuracy = self._cal_accuracy(logits, labels)
            loss = self._loss(labels, logits)
            train_op = self._train_op(global_step, loss)

            saver = tf.train.Saver(tf.global_variables())
            sess.run(tf.global_variables_initializer())

            if retrain:
                self.train_data = np.concatenate(
                    (np.array(self.train_data), np.array(self.valid_data)), axis=0).tolist()
                self.train_label = np.concatenate(
                    (np.array(self.train_label), np.array(self.valid_label)), axis=0).tolist()
                max_steps = (self.NUM_EXAMPLES_FOR_TRAIN + self.NUM_EXAMPLES_FOR_EVAL) // self.batch_size
                test_data = copy.deepcopy(self.test_data)
                test_label = copy.deepcopy(self.test_label)
                num_iter = len(test_label) // self.batch_size
            else:
                max_steps = self.train_num // self.batch_size
                test_data = copy.deepcopy(self.valid_data)
                test_label = copy.deepcopy(self.valid_label)
                num_iter = self.NUM_EXAMPLES_FOR_EVAL // self.batch_size

            log = ''
            run_metadata = tf.RunMetadata()
            cost_time = 0
            precision = np.zeros([self.epoch])
            for ep in range(self.epoch):
                # print("epoch", ep, ":")
                # train step
                start_time = time.time()
                for step in range(max_steps):
                    batch_x = self.train_data[step *
                                              self.batch_size:(step + 1) * self.batch_size]
                    batch_y = self.train_label[step *
                                               self.batch_size:(step + 1) * self.batch_size]
                    batch_x = DataSet().process(batch_x)
                    _, loss_value, acc = sess.run([train_op, loss, accuracy],
                                                  feed_dict={data_x: batch_x, labels: batch_y, train_flag: True},
                                                  options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                                  run_metadata=run_metadata)
                    if np.isnan(loss_value):
                        return -1, saver, log
                    sys.stdout.write("\r>> train %d/%d loss %.4f acc %.4f" % (step, max_steps, loss_value, acc))
                sys.stdout.write("\n")
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open('timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format())

                # evaluation step
                for step in range(num_iter):
                    batch_x = test_data[step *
                                        self.batch_size:(step + 1) * self.batch_size]
                    batch_y = test_label[step *
                                         self.batch_size:(step + 1) * self.batch_size]
                    l, acc_ = sess.run([loss, accuracy],
                                       feed_dict={data_x: batch_x, labels: batch_y, train_flag: False})
                    precision[ep] += acc_ / num_iter
                    sys.stdout.write("\r>> valid %d/%d loss %.4f acc %.4f" % (step, num_iter, l, acc_))
                sys.stdout.write("\n")

                # early stop
                if ep > 5 and not retrain:
                    if precision[ep] < 1.2 / self.NUM_CLASSES:
                        precision = [-1]
                        break
                    if 2 * precision[ep] - precision[ep - 5] - precision[ep - 1] < 0.001 / self.NUM_CLASSES:
                        precision = precision[:ep]
                        log += 'early stop at %d epoch\n' % ep
                        break

                cost_time += (float(time.time() - start_time)) / self.epoch
                log += 'epoch %d: precision = %.3f, cost time %.3f\n' % (
                    ep, precision[ep], float(time.time() - start_time))
            retrain_log += log

        # NAS_LOG << ('eva', retrain_log)
        return float(precision)

    def _get_input(self, sess, pre_block, update_pre_weight=False):
        '''Get input for _inference'''
        # if it got previous blocks
        if len(pre_block) > 0:
            new_saver = tf.train.import_meta_graph(
                os.path.join(self.model_path, 'model' + str(pre_block[-1].id) + '.meta'))
            new_saver.restore(sess, os.path.join(
                self.model_path, 'model' + str(pre_block[-1].id)))
            graph = tf.get_default_graph()
            data_x = graph.get_tensor_by_name("input:0")
            data_y = graph.get_tensor_by_name("label:0")
            train_flag = graph.get_tensor_by_name("train_flag:0")
            block_input = graph.get_tensor_by_name(
                "last_layer" + str(self.block_num - 1) + ":0")
            # only when there's not so many network in the pool will we update the previous blocks' weight
            if not update_pre_weight:
                block_input = tf.stop_gradient(block_input, name="stop_gradient")
        # if it's the first block
        else:
            data_x = tf.placeholder(
                tf.float32, [self.batch_size, self.IMAGE_SIZE, self.IMAGE_SIZE, 3], name='input')
            data_y = tf.placeholder(
                tf.int32, [self.batch_size, self.NUM_CLASSES], name="label")
            train_flag = tf.placeholder(tf.bool, name='train_flag')
            block_input = tf.identity(data_x)
        return data_x, data_y, block_input, train_flag

    def _eval(self, sess, data_x, labels, logits, train_flag, retrain=False):
        global_step = tf.Variable(
            0, trainable=False, name='global_step' + str(self.block_num))
        accuracy = self._cal_accuracy(logits, labels)
        loss = self._loss(labels, logits)
        train_op = self._train_op(global_step, loss)

        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        if retrain:
            self.train_data = np.concatenate(
                (np.array(self.train_data), np.array(self.valid_data)), axis=0).tolist()
            self.train_label = np.concatenate(
                (np.array(self.train_label), np.array(self.valid_label)), axis=0).tolist()
            max_steps = (self.NUM_EXAMPLES_FOR_TRAIN + self.NUM_EXAMPLES_FOR_EVAL) // self.batch_size
            test_data = copy.deepcopy(self.test_data)
            test_label = copy.deepcopy(self.test_label)
            num_iter = len(test_label) // self.batch_size
        else:
            max_steps = self.train_num // self.batch_size
            test_data = copy.deepcopy(self.valid_data)
            test_label = copy.deepcopy(self.valid_label)
            num_iter = self.NUM_EXAMPLES_FOR_EVAL // self.batch_size

        log = ''
        run_metadata = tf.RunMetadata()
        cost_time = 0
        precision = np.zeros([self.epoch])
        for ep in range(self.epoch):
            # print("epoch", ep, ":")
            # train step
            start_time = time.time()
            for step in range(max_steps):
                batch_x = self.train_data[step *
                                          self.batch_size:(step + 1) * self.batch_size]
                batch_y = self.train_label[step *
                                           self.batch_size:(step + 1) * self.batch_size]
                batch_x = DataSet().process(batch_x)
                _, loss_value, acc = sess.run([train_op, loss, accuracy],
                                              feed_dict={data_x: batch_x, labels: batch_y, train_flag: True},
                                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                              run_metadata=run_metadata)
                if np.isnan(loss_value):
                    return -1, saver, log
                sys.stdout.write("\r>> train %d/%d loss %.4f acc %.4f" % (step, max_steps, loss_value, acc))
            sys.stdout.write("\n")
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())

            # evaluation step
            for step in range(num_iter):
                batch_x = test_data[step *
                                    self.batch_size:(step + 1) * self.batch_size]
                batch_y = test_label[step *
                                     self.batch_size:(step + 1) * self.batch_size]
                l, acc_ = sess.run([loss, accuracy],
                                   feed_dict={data_x: batch_x, labels: batch_y, train_flag: False})
                precision[ep] += acc_ / num_iter
                sys.stdout.write("\r>> valid %d/%d loss %.4f acc %.4f" % (step, num_iter, l, acc_))
            sys.stdout.write("\n")

            # early stop
            if ep > 5 and not retrain:
                if precision[ep] < 1.2 / self.NUM_CLASSES:
                    precision = [-1]
                    break
                if 2 * precision[ep] - precision[ep - 5] - precision[ep - 1] < 0.001 / self.NUM_CLASSES:
                    precision = precision[:ep]
                    log += 'early stop at %d epoch\n' % ep
                    break

            cost_time += (float(time.time() - start_time)) / self.epoch
            log += 'epoch %d: precision = %.3f, cost time %.3f\n' % (
                ep, precision[ep], float(time.time() - start_time))
            # print('precision = %.3f, cost time %.3f' %
            #       (precision[ep], float(time.time() - start_time)))

        target = self._cal_multi_target(precision[-1], cost_time)
        return target, saver, log

    def _cal_accuracy(self, logits, labels):
        """
        calculate the target of this task
            Args:
                logits: Logits from softmax.
                labels: Labels from distorted_inputs or inputs(). 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
            Returns:
                Target tensor of type float.
        """
        correct_prediction = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    def _loss(self, labels, logits):
        """
          Args:
            logits: Logits from softmax.
            labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [self.batch_size]
          Returns:
            Loss tensor of type float.
          """
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        loss = cross_entropy + l2 * self.weight_decay
        return loss

    def _train_op(self, global_step, loss):
        num_batches_per_epoch = self.train_num / self.batch_size
        decay_steps = int(num_batches_per_epoch * self.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.cosine_decay(self.INITIAL_LEARNING_RATE, global_step, decay_steps)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        opt = tf.train.MomentumOptimizer(lr, self.momentum_rate, name='Momentum' + str(self.block_num),
                                         use_nesterov=True)
        train_op = opt.minimize(loss, global_step=global_step)
        return train_op

    def _stats_graph(self):
        graph = tf.get_default_graph()
        flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
        params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
        return flops.total_float_ops, params.total_parameters

    def _cal_multi_target(self, precision, time):
        flops, model_size = self._stats_graph()
        return precision + 1 / time + 1 / flops + 1 / model_size

    def set_data_size(self, num):
        if num > self.NUM_EXAMPLES_FOR_TRAIN or num < 0:
            num = self.NUM_EXAMPLES_FOR_TRAIN
            self.train_num = self.NUM_EXAMPLES_FOR_TRAIN
            print('Warning! Data size has been changed to',
                  num, ', all data is loaded.')
        else:
            self.train_num = num
        # print('************A NEW ROUND************')
        self.max_steps = self.train_num // self.batch_size
        return


def inputs():
    print("======Loading data======")

    train_files = ['train']
    test_files = ['test']
    train_data, train_label = _load(train_files)
    test_data, test_label = _load(test_files)
    print("======Data Process Done======")
    return train_data, train_label, test_data, test_label


def _load_one(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    data = batch[b'data']
    label = batch[b'fine_labels']
    return data, label


def _load(files):
    file_name = 'cifar-100-python'
    data_dir = os.path.join('/data/data', file_name)
    data, label = _load_one(os.path.join(data_dir, files[0]))
    for f in files[1:]:
        batch_data, batch_label = _load_one(os.path.join(data_dir, f))
        data = np.append(data, batch_data, axis=0)
        label = np.append(label, batch_label, axis=0)
    label = np.array([[float(i == label)
                       for i in range(100)] for label in label])
    data = data.reshape([-1, 3, 32, 32])
    data = data.transpose([0, 2, 3, 1])
    # pre-process
    data = _normalize(data)

    return data, label


def _normalize(x_train):
    x_train = x_train.astype('float32')

    x_train[:, :, :, 0] = (
                                  x_train[:, :, :, 0] - np.mean(x_train[:, :, :, 0])) / np.std(x_train[:, :, :, 0])
    x_train[:, :, :, 1] = (
                                  x_train[:, :, :, 1] - np.mean(x_train[:, :, :, 1])) / np.std(x_train[:, :, :, 1])
    x_train[:, :, :, 2] = (
                                  x_train[:, :, :, 2] - np.mean(x_train[:, :, :, 2])) / np.std(x_train[:, :, :, 2])

    return x_train


def process(x):
    x = _random_flip_leftright(x)
    x = _random_crop(x, [32, 32], 4)
    x = _cutout(x)
    return x


def _random_crop(self, batch, crop_shape, padding=None):
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
    return np.array(new_batch)


def _random_flip_leftright(self, batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def _cutout(self, x):
    for i in range(len(x)):
        cut_size = random.randint(0, self.IMAGE_SIZE // 2)
        s = random.randint(0, self.IMAGE_SIZE - cut_size)
        x[i, s:s + cut_size, s:s + cut_size, :] = 0
    return x


def te():
    retrain_log = "-" * 20 + "retrain" + "-" * 20 + '\n'
    batch_size = 64
    epoch = 200

    with tf.Session() as sess:
        data_x = tf.placeholder(
            tf.float32, [64, 32, 32, 3], name='input')
        labels = tf.placeholder(
            tf.int32, [64, 100], name="label")
        train_flag = tf.placeholder(tf.bool, name='train_flag')

        # for block in pre_block:
        #     graph = block.graph + [[]]
        #     cell_list = []
        #     for cell in block.cell_list:
        #         if cell.type == 'conv':
        #             cell_list.append(
        #                 Cell(cell.type, cell.filter_size * 2, cell.kernel_size, cell.activation))
        #         else:
        #             cell_list.append(cell)
        #     cell_list.append(Cell('pooling', 'max', 1))
        #     # repeat search
        #     for _ in range(NAS_CONFIG['nas_main']['repeat_search'] - 1):
        #         retrain_log = retrain_log + str(graph) + str(cell_list) + '\n'
        #         logits = self._inference(logits, graph, cell_list, train_flag)
        #         self.block_num += 1
        #     # add pooling layer only in last repeat block
        #     cell_list[-1] = Cell('pooling', 'max', 2)
        #     retrain_log = retrain_log + str(graph) + str(cell_list) + '\n'
        logits = densenet201(data_x, reuse=False, is_training=True, kernel_initializer=tf.orthogonal_initializer())

        # logits = tf.nn.dropout(logits, keep_prob=1.0)
        # softmax
        # logits = self._makedense(logits, ('', [256, self.NUM_CLASSES], 'relu'))
        retrain = True

        global_step = tf.Variable(
            0, trainable=False, name='global_step')
        accuracy = cal_accuracy(logits, labels)
        loss = loss_(labels, logits)
        train_op = train_op_(global_step, loss)

        saver = tf.train.Saver(tf.global_variables())
        sess.run(tf.global_variables_initializer())

        train_data, train_label, test_data, test_label = inputs()
        max_steps = (50000) // batch_size
        num_iter = len(test_label) // batch_size

        log = ''
        run_metadata = tf.RunMetadata()
        cost_time = 0
        precision = np.zeros([epoch])
        for ep in range(epoch):
            # print("epoch", ep, ":")
            # train step
            start_time = time.time()
            for step in range(max_steps):
                batch_x = train_data[step *
                                     batch_size:(step + 1) * batch_size]
                batch_y = train_label[step *
                                      batch_size:(step + 1) * batch_size]
                batch_x = DataSet().process(batch_x)
                _, loss_value, acc = sess.run([train_op, loss, accuracy],
                                              feed_dict={data_x: batch_x, labels: batch_y, train_flag: True},
                                              options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                                              run_metadata=run_metadata)
                if np.isnan(loss_value):
                    return -1, saver, log
                sys.stdout.write("\r>> train %d/%d loss %.4f acc %.4f" % (step, max_steps, loss_value, acc))
            sys.stdout.write("\n")
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())

            # evaluation step
            for step in range(num_iter):
                batch_x = test_data[step *
                                    batch_size:(step + 1) * batch_size]
                batch_y = test_label[step *
                                     batch_size:(step + 1) * batch_size]
                l, acc_ = sess.run([loss, accuracy],
                                   feed_dict={data_x: batch_x, labels: batch_y, train_flag: False})
                precision[ep] += acc_ / num_iter
                sys.stdout.write("\r>> valid %d/%d loss %.4f acc %.4f" % (step, num_iter, l, acc_))
            sys.stdout.write("\n")

            cost_time += (float(time.time() - start_time)) / 200
            log += 'epoch %d: precision = %.3f, cost time %.3f\n' % (
                ep, precision[ep], float(time.time() - start_time))
        retrain_log += log

    # NAS_LOG << ('eva', retrain_log)
    return float(precision)


def cal_accuracy(logits, labels):
    """
    calculate the target of this task
        Args:
            logits: Logits from softmax.
            labels: Labels from distorted_inputs or inputs(). 2-D tensor of shape [self.batch_size, self.NUM_CLASS]
        Returns:
            Target tensor of type float.
    """
    correct_prediction = tf.equal(
        tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def loss_(labels, logits):
    """
      Args:
        logits: Logits from softmax.
        labels: Labels from distorted_inputs or inputs(). 1-D tensor of shape [self.batch_size]
      Returns:
        Loss tensor of type float.
      """
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    loss = cross_entropy + l2 * 0.0001
    return loss


def train_op_(global_step, loss):
    num_batches_per_epoch = 50000 / 64
    decay_steps = int(num_batches_per_epoch * 200)
    lr = tf.train.cosine_decay(0.001, global_step, decay_steps)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    opt = tf.train.MomentumOptimizer(lr, 0.9, name='Momentum',
                                     use_nesterov=True)
    train_op = opt.minimize(loss, global_step=global_step)
    return train_op


def parse_function(example_proto):
    features = {'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)

    img = tf.decode_raw(features['image_raw'], tf.float32)
    img = tf.reshape(img, shape=(32, 32, 3))

    img = tf.pad(img, [[4, 4], [4, 4], [0, 0]])
    img = tf.random_crop(img, [32, 32, 3])
    # img = tf.image.random_flip_left_right(img)

    flip = random.getrandbits(1)
    if flip:
        img = img[:, ::-1, :]
    # rot = random.randint(-15, 15)
    # img = tf.contrib.image.rotate(img, rot)
    # img = tf.image.rot90(img, rot)

    label = tf.cast(features['label'], tf.int64)
    return img, label


def compute_mean_var(image):
    # image.shape: [image_num, w, h, c]
    mean = []
    var = []
    for c in range(image.shape[-1]):
        mean.append(np.mean(image[:, :, :, c]))
        var.append(np.std(image[:, :, :, c]))
    return mean, var


def norm_images(image):
    # image.shape: [image_num, w, h, c]
    image = image.astype('float32')
    mean, var = compute_mean_var(image)
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image


def norm_images_using_mean_var(image, mean, var):
    image = image.astype('float32')
    image[:, :, :, 0] = (image[:, :, :, 0] - mean[0]) / var[0]
    image[:, :, :, 1] = (image[:, :, :, 1] - mean[1]) / var[1]
    image[:, :, :, 2] = (image[:, :, :, 2] - mean[2]) / var[2]
    return image


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def generate_tfrecord(train, labels, output_path, output_name):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    writer = tf.python_io.TFRecordWriter(os.path.join(output_path, output_name))
    for ind, (file, label) in enumerate(zip(train, labels)):
        img_raw = file.tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if ind != 0 and ind % 1000 == 0:
            print("%d num imgs processed" % ind)
    writer.close()


def lr_schedule(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    if epoch < 200:
        return 0.0008


def test():
    train = unpickle('/data/data/cifar-100-python/train')
    test = unpickle('/data/data/cifar-100-python/test')
    train_data = train[b'data']
    test_data = test[b'data']

    x_train = train_data.reshape(train_data.shape[0], 3, 32, 32)
    x_train = x_train.transpose(0, 2, 3, 1)
    y_train = train[b'fine_labels']

    x_test = test_data.reshape(test_data.shape[0], 3, 32, 32)
    x_test = x_test.transpose(0, 2, 3, 1)
    y_test = test[b'fine_labels']

    x_train = norm_images(x_train)
    x_test = norm_images(x_test)

    if not os.path.exists('./trans/tran.tfrecords'):
        generate_tfrecord(x_train, y_train, './trans/', 'tran.tfrecords')
        generate_tfrecord(x_test, y_test, './trans/', 'test.tfrecords')

    dataset = tf.data.TFRecordDataset('./trans/tran.tfrecords')
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=50000)
    dataset = dataset.batch(Evaluator().batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    x_input = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y_input = tf.placeholder(tf.int64, [None, ])
    y_input_one_hot = tf.one_hot(y_input, 100)
    lr = tf.placeholder(tf.float32, [])

    prob = densenet201(x_input, reuse=False, is_training=True, kernel_initializer=tf.orthogonal_initializer())

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=y_input_one_hot))

    conv_var = [var for var in tf.trainable_variables() if 'conv' in var.name]
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in conv_var])
    loss = l2_loss * 5e-4 + loss
    opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.minimize(loss)

    logit_softmax = tf.nn.softmax(prob)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit_softmax, 1), y_input), tf.float32))
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    now_lr = 0.001  # Warm Up
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        counter = 0
        max_test_acc = -1
        for i in range(Evaluator().epoch):
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_train, label_train = sess.run(next_element)
                    _, loss_val, acc_val, lr_val = sess.run([train_op, loss, acc, lr],
                                                            feed_dict={x_input: batch_train, y_input: label_train,
                                                                       lr: now_lr})

                    counter += 1

                    if counter % 100 == 0:
                        print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)

                except tf.errors.OutOfRangeError:
                    print('end epoch %d/%d , lr: %f' % (i, Evaluator().epoch, lr_val))
                    now_lr = lr_schedule(i)
                    break


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    eval = Evaluator()
    eval.set_data_size(5000)
    eval.set_epoch(200)
    te()
