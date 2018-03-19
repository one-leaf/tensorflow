import gzip
import csv
import numpy as np
import os

import urllib

import functools
import tensorflow as tf
import random


def lazy_property(function):
    attribute = '_lazy_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class AttrDict(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value

# http://ai.stanford.edu/~btaskar/ocr/
# 输出是 [[单词的图片]...] [[单词]...]  => [[101..., 110..., ...], ...]   [[h,e,l,l,o], ...]
# 图片每一个字母是 16*8
class OcrDataset:
    URL = 'http://ai.stanford.edu/~btaskar/ocr/letter.data.gz'

    def __init__(self, cache_dir):
        path = os.path.join(cache_dir, "letter.data.gz")
        urllib.request.urlretrieve(type(self).URL, path)
        lines = self._read(path)
        data, target = self._parse(lines)
        self.data, self.target = self._pad(data, target)

    @staticmethod
    def _read(filepath):
        with gzip.open(filepath, 'rt') as file_:
            reader = csv.reader(file_, delimiter='\t')
            lines = list(reader)
            return lines

    @staticmethod
    def _parse(lines):
        lines = sorted(lines, key=lambda x: int(x[0]))
        data, target = [], []
        next_ = None
        for line in lines:
            if not next_:
                data.append([])
                target.append([])
            else:
                assert next_ == int(line[0])
            next_ = int(line[2]) if int(line[2]) > -1 else None
            pixels = np.array([int(x) for x in line[6:134]])
            pixels = pixels.reshape((16, 8))
            data[-1].append(pixels)
            target[-1].append(line[1])
        return data, target

    @staticmethod
    def _pad(data, target):
        max_length = max(len(x) for x in target)
        print("max_length:",max_length)
        padding = np.zeros((16, 8))
        data = [x + ([padding] * (max_length - len(x))) for x in data]
        target = [x + ([''] * (max_length - len(x))) for x in target]
        return np.array(data), np.array(target)

def batched(data, target, batch_size):
    epoch = 0
    offset = 0
    while True:
        old_offset = offset
        offset = (offset + batch_size) % (target.shape[0] - batch_size)

        if offset < old_offset:
            # New epoch, need to shuffle data
            p = np.random.permutation(len(data))
            data = data[p]
            target = target[p]
            epoch += 1

        batch_data = data[offset:(offset + batch_size), :]
        batch_target = target[offset:(offset + batch_size), :]
        yield batch_data, batch_target, epoch


class SequenceLabellingModel:

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.prediction
        self.cost
        self.error
        self.optimize

    @lazy_property
    def length(self):
        print("self.data.shape:",self.data.shape)
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        print("used.shape:",used.shape)
        length = tf.reduce_sum(used, reduction_indices=1)
        print("length.shape:",length.shape)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        output, _ = tf.nn.dynamic_rnn(
            tf.nn.rnn_cell.GRUCell(self.params.rnn_hidden),
            self.data,
            dtype=tf.float32,
            sequence_length=self.length,
        )
        # Softmax layer.
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight = tf.Variable(tf.truncated_normal(
            [self.params.rnn_hidden, num_classes], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        # Flatten to apply same weights to all time steps.
        output = tf.reshape(output, [-1, self.params.rnn_hidden])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        prediction = tf.reshape(prediction, [-1, max_length, num_classes])
        return prediction

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @lazy_property
    def optimize(self):
        gradient = self.params.optimizer.compute_gradients(self.cost)
        try:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        except AttributeError:
            print('No gradient clipping parameter specified.')
        optimize = self.params.optimizer.apply_gradients(gradient)
        return optimize


class BidirectionalSequenceLabellingModel:
    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.prediction
        self.cost
        self.error
        self.optimize

    @lazy_property
    def length(self):
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    @lazy_property
    def prediction(self):
        output = self._bidirectional_rnn(self.data, self.length)
        num_classes = int(self.target.get_shape()[2])
        prediction = self._shared_softmax(output, num_classes)
        return prediction

    def _bidirectional_rnn(self, data, length):
        length_64 = tf.cast(length, tf.int64)
        forward, _ = tf.nn.dynamic_rnn(
            cell=self.params.rnn_cell(self.params.rnn_hidden),
            inputs=data,
            dtype=tf.float32,
            sequence_length=length,
            scope='rnn-forward')
        backward, _ = tf.nn.dynamic_rnn(
        cell=self.params.rnn_cell(self.params.rnn_hidden),
        inputs=tf.reverse_sequence(data, length_64, seq_dim=1),
        dtype=tf.float32,
        sequence_length=self.length,
        scope='rnn-backward')
        backward = tf.reverse_sequence(backward, length_64, seq_dim=1)
        output = tf.concat([forward, backward],2)
        return output

    def _shared_softmax(self, data, out_size):
        max_length = int(data.get_shape()[1])
        in_size = int(data.get_shape()[2])
        weight = tf.Variable(tf.truncated_normal(
            [in_size, out_size], stddev=0.01))
        bias = tf.Variable(tf.constant(0.1, shape=[out_size]))
        # Flatten to apply same weights to all time steps.
        flat = tf.reshape(data, [-1, in_size])
        output = tf.nn.softmax(tf.matmul(flat, weight) + bias)
        output = tf.reshape(output, [-1, max_length, out_size])
        return output

    @lazy_property
    def cost(self):
        # Compute cross entropy for each frame.
        cross_entropy = self.target * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        cross_entropy *= mask
        # Average over actual sequence lengths.
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, tf.float32)
        mask = tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))
        mistakes *= mask
        # Average over actual sequence lengths.
        mistakes = tf.reduce_sum(mistakes, reduction_indices=1)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @lazy_property
    def optimize(self):
        gradient = self.params.optimizer.compute_gradients(self.cost)
        try:
            limit = self.params.gradient_clipping
            gradient = [
                (tf.clip_by_value(g, -limit, limit), v)
                if g is not None else (None, v)
                for g, v in gradient]
        except AttributeError:
            print('No gradient clipping parameter specified.')
        optimize = self.params.optimizer.apply_gradients(gradient)
        return optimize


params = AttrDict(
    rnn_cell=tf.nn.rnn_cell.GRUCell,
    rnn_hidden=300,
    optimizer=tf.train.RMSPropOptimizer(0.002),
    gradient_clipping=5,
    batch_size=10,
    epochs=5,
    epoch_size=50
)

def get_dataset():
    curr_dir = os.path.dirname(__file__)
    cache_dir = os.path.join(curr_dir,"data")
    if not os.path.exists(cache_dir): os.mkdir(cache_dir)

    dataset = OcrDataset(cache_dir)
    # Flatten images into vectors.
    print("init dataset.data.shape",dataset.data.shape)
    dataset.data = dataset.data.reshape(dataset.data.shape[:2] + (-1,))
    # One-hot encode targets.
    print("init dataset.target.shape",dataset.target.shape)
    target = np.zeros(dataset.target.shape + (26,))
    print("target.shape",target.shape)
    for index, letter in np.ndenumerate(dataset.target):
        if letter:
            target[index][ord(letter) - ord('a')] = 1
    dataset.target = target
    # Shuffle order of examples.
    order = np.random.permutation(len(dataset.data))
    dataset.data = dataset.data[order]
    dataset.target = dataset.target[order]
    return dataset

# Split into training and test data.
dataset = get_dataset()
split = int(0.66 * len(dataset.data))
print("dataset data len:",len(dataset.data),"split:",split)
train_data, test_data = dataset.data[:split], dataset.data[split:]
train_target, test_target = dataset.target[:split], dataset.target[split:]

# Compute graph.
_, length, image_size = train_data.shape
print("train_data shape:",train_data.shape)
num_classes = train_target.shape[2]
print("train_target shape:",train_target.shape)

data = tf.placeholder(tf.float32, [None, length, image_size])
target = tf.placeholder(tf.float32, [None, length, num_classes])
model = SequenceLabellingModel(data, target, params)
# model = BidirectionalSequenceLabellingModel(data, target, params)
batches = batched(train_data, train_target, params.batch_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for index, batch in enumerate(batches):
    batch_data = batch[0]
    batch_target = batch[1]
    epoch = batch[2]
    if epoch >= params.epochs:
        break
    feed = {data: batch_data, target: batch_target}
    error, _ = sess.run([model.error, model.optimize], feed)
    print('{}: {:3.6f}%'.format(index + 1, 100 * error))

test_feed = {data: test_data, target: test_target}
test_error, _ = sess.run([model.error, model.optimize], test_feed)
print('Test error: {:3.6f}%'.format(100 * error))


