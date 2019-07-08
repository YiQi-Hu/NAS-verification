import tensorflow as tf
import numpy as np
import time
import os
import sys
import random
import pickle as pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class_num = 10
image_size = 32
img_channels = 3
iterations = 200
batch_size = 250
total_epoch = 164
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_16_logs'
model_save_path = './model/'
NUM_CLASSES = 10


def download_data():
    dirname = 'cifar10-dataset'
    origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    fname = './CAFIR-10_data/cifar-10-python.tar.gz'
    fpath = './' + dirname

    download = False
    if os.path.exists(fpath) or os.path.isfile(fname):
        download = False
        print("DataSet already exist!")
    else:
        download = True
    if download:
        print('Downloading data from', origin)
        import urllib.request
        import tarfile

        def reporthook(count, block_size, total_size):
            global start_time
            if count == 0:
                start_time = time.time()
                return
            duration = time.time() - start_time
            progress_size = int(count * block_size)
            speed = int(progress_size / (1024 * duration))
            percent = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                             (percent, progress_size / (1024 * 1024), speed, duration))
            sys.stdout.flush()

        urllib.request.urlretrieve(origin, fname, reporthook)
        print('Download finished. Start extract!', origin)
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_data_one(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." % (file, len(data)))
    return data, labels


#
#
# def load_data(files, data_dir, label_count):
#     global image_size, img_channels
#     data, labels = load_data_one(data_dir + '\\' + files[0])
#     for f in files[1:]:
#         data_n, labels_n = load_data_one(data_dir + '\\' + f)
#         data = np.append(data, data_n, axis=0)
#         labels = np.append(labels, labels_n, axis=0)
#     labels = np.array([[float(i == label) for i in range(label_count)] for label in labels])
#     data = data.reshape([-1, img_channels, image_size, image_size])
#     data = data.transpose([0, 2, 3, 1])
#     return data, labels


def prepare_data():
    print("======Loading data======")
    # download_data()
    data_dir = os.path.join(os.getcwd(), 'cifar-10-batches-bin')
    image_dim = image_size * image_size * img_channels
    # meta = unpickle(data_dir + '\\batches.meta.txt')
    #
    #
    # print(meta)
    # label_names = meta[b'label_names']
    label_count = 10
    train_files = ['data_batch_%d.bin' % d for d in range(1, 6)]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data(['test_batch.bin'], data_dir, label_count)

    print("Train data:", np.shape(train_data), np.shape(train_labels))
    print("Test data :", np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")

    print("======Shuffling data======")
    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]
    train_labels = train_labels[indices]
    print("======Prepare Finished======")

    return train_data, train_labels, test_data, test_labels


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input):
    # return input
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


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
    else:
        return 0.001


def run_testing(sess, ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index + add]
        batch_y = test_y[pre_index:pre_index + add]
        pre_index = pre_index + add
        loss_, acc_ = sess.run([cross_entropy, accuracy],
                               feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss),
                                tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss, summary


def load_batch(filename, num=10000):
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


def load_data(ROOT):
    """ load all of cifar """
    print('Evaluater: loading data')
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d.bin' % (b,))
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)  # 使变成行向量
    Ytr = np.concatenate(ys)
    del X, Y
    # print('Evaluater:data loaded')
    return Xtr, Ytr


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
    # layer = tf.reshape(layer, [batch_size, -1])
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[layer.shape[-1], NUM_CLASSES],
                                  initializer=tf.contrib.keras.initializers.he_normal())
        biases = tf.get_variable('biases', shape=[NUM_CLASSES], initializer=tf.constant_initializer(0.0))
        # softmax_linear = tf.nn.softmax(tf.matmul(layer, weights)+ biases, name=scope.name)
        softmax_linear = tf.add(tf.matmul(layer, weights), biases, name=scope.name)
        # tf.add_to_collection('losses', regularizer(weights))
    return softmax_linear


if __name__ == '__main__':
    path = os.getcwd()  # +'/../'
    data_dir = "C:\\Users\\admin\\Documents\\AutoML\\CompetitionNAS\\CompetitionNAS-master\\nas\\cifar-10-batches-bin"

    train_x, train_y = load_data(data_dir)
    test_x, test_y = load_batch(os.path.join(data_dir, 'test_batch.bin'))
    train_x, test_x = data_preprocessing(train_x, test_x)

    # define placeholder x, y_ , keep_prob, learning_rate
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.int64, [None, ])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    graph_part = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17],
                  []]
    cell_list = [('conv', 64, 3, 'relu'), ('conv', 64, 3, 'relu'), ('pooling', 'max', 2), ('conv', 128, 3, 'relu'),
                 ('conv', 128, 3, 'relu'), ('pooling', 'max', 2), ('conv', 256, 3, 'relu'),
                 ('conv', 256, 3, 'relu'), ('conv', 256, 3, 'relu'), ('pooling', 'max', 2),
                 ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
                 ('pooling', 'max', 2), ('conv', 512, 3, 'relu'), ('conv', 512, 3, 'relu'),
                 ('conv', 512, 3, 'relu'), ('dense', [4096, 4096, 1000], 'relu')]

    output = inference(x, graph_part, cell_list)

    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True). \
        minimize(cross_entropy + l2 * weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path, sess.graph)

        # epoch = 164
        # make sure [bath_size * iteration = data_set_number]

        for ep in range(1, total_epoch + 1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\n epoch %d/%d:" % (ep, total_epoch))

            for it in range(1, iterations + 1):
                batch_x = train_x[pre_index:pre_index + batch_size]
                batch_y = train_y[pre_index:pre_index + batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],
                                         feed_dict={x: batch_x, y_: batch_y, keep_prob: dropout_rate,
                                                    learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    loss_, acc_ = sess.run([cross_entropy, accuracy],
                                           feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0, train_flag: True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss),
                                                      tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess, ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, "
                          "train_acc: %.4f, test_loss: %.4f, test_acc: %.4f"
                          % (it, iterations, int(time.time() - start_time), train_loss, train_acc, val_loss, val_acc))
                # else:
                #     print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f"
                #           % (it, iterations, train_loss / it, train_acc / it))

        save_path = saver.save(sess, model_save_path)
        print("Model saved in file: %s" % save_path)
