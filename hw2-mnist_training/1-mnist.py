from __future__ import division
from __future__ import print_function

import math
import argparse
import copy
import datetime
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics
from tensorflow.examples.tutorials.mnist import input_data
from threading import Thread

class TrainingSetup(object):

    BATCH_SIZES = (10, 50)
    LEARNING_RATES = (0.01, 0.001, 0.0001)

    def __init__(self, optimizer_name, batch_size, learning_rate):
        self._optimizer_name = optimizer_name
        self._batch_size = batch_size
        self._learning_rate = learning_rate

    def setup(self, loss, global_step):
        return None

    @property
    def name(self):
        return "name={0},batch={1},learning-rate={2}".format(self._optimizer_name, self._batch_size, self._learning_rate)

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

class AdamSetup(TrainingSetup):

    LEARNING_RATES = (0.002, 0.001, 0.0005)

    def __init__(self, batch_size, learning_rate, training_data_set_size, epochs):
        super(AdamSetup, self).__init__("Adam", batch_size, learning_rate)

    def setup(self, loss, global_step):
        return tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(loss, global_step=global_step)

class SGDSetup(TrainingSetup):

    def __init__(self, batch_size, learning_rate, training_data_set_size, epochs):
        super(SGDSetup, self).__init__("SGD", batch_size, learning_rate)

    def setup(self, loss, global_step):
        return tf.train.GradientDescentOptimizer(learning_rate=self._learning_rate).minimize(loss, global_step=global_step)

class SGDWithExponentialDecaySetup(TrainingSetup):

    LEARNING_RATES = ((0.01,0.001), (0.01,0.0001), (0.001, 0.0001))

    def __init__(self, batch_size, learning_rate, training_data_set_size, epochs):
        super(SGDWithExponentialDecaySetup, self).__init__("SGDWithExponentialDecay", batch_size, learning_rate)
        self._epochs = epochs
        self._training_data_set_size = training_data_set_size

    def setup(self, loss, global_step):
        decay_rate = math.pow(self._learning_rate[1] / self._learning_rate[0], float(self._batch_size) / (self._training_data_set_size * self._epochs))
        learning_rate = learning_rate = tf.train.exponential_decay(self._learning_rate[0], global_step, 1, decay_rate)
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

class SGDWithMomentumSetup(TrainingSetup):

    def __init__(self, batch_size, learning_rate, training_data_set_size, epochs):
        super(SGDWithMomentumSetup, self).__init__("SGDWithMomentum", batch_size, learning_rate)

    def setup(self, loss, global_step):
        return tf.train.MomentumOptimizer(learning_rate=self._learning_rate, momentum=0.9).minimize(loss, global_step=global_step)

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads=1, logdir=None, expname=None, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        if logdir:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.summary_writer = tf.train.SummaryWriter(("{}/{}-{}" if expname else "{}/{}").format(logdir, timestamp, expname), flush_secs=10)
        else:
            self.summary_writer = None

    def construct(self, hidden_layer_size, training_setup):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            flattened_images = tf_layers.flatten(self.images, scope="preprocessing")
            hidden_layer = tf_layers.fully_connected(flattened_images, num_outputs=hidden_layer_size, activation_fn=tf.nn.relu, scope="hidden_layer")
            output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=self.LABELS, activation_fn=None, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

            self.training = training_setup.setup(loss, self.global_step)

            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", loss),
                                                            tf.scalar_summary("train/accuracy", self.accuracy)])}
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = tf.scalar_summary(dataset+"/accuracy", self.accuracy)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, images, labels, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.images: images, self.labels: labels}}
        targets = [self.training]
        if summaries:
            targets.append(self.summaries["training"])
        if run_metadata:
            args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            args["run_metadata"] = tf.RunMetadata()

        results = self.session.run(targets, **args)
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step - 1)
        if run_metadata:
            self.summary_writer.add_run_metadata(args["run_metadata"], "step{:05}".format(self.training_step - 1))

    def evaluate(self, dataset, images, labels, summaries=False):
        if summaries and not self.summary_writer:
            raise ValueError("Logdir is required for summaries.")

        targets = [self.accuracy]
        if summaries:
            targets.append(self.summaries[dataset])

        results = self.session.run(targets, {self.images: images, self.labels: labels})
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]

class OptimizationThread(Thread):

    def __init__(self, data, training_setup, args):
        super(OptimizationThread, self).__init__()
        self._training_setup = training_setup
        self._args = args
        self._data = data

    def run(self):
        print("Training network {0} started.".format(self._training_setup.name))
        args = self._args
        mnist = self._data 

        # Construct the network
        network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
        network.construct(100, self._training_setup)

        # Train
        for i in range(args.epochs):
            while mnist.train.epochs_completed == i:
                images, labels = mnist.train.next_batch(self._training_setup.batch_size)
                network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

            network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
            network.evaluate("test", mnist.test.images, mnist.test.labels, True)

        # Reprt development set accuracy
        accuracy = network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
        print("Network: {0}, Accuracy: {1}".format(self._training_setup.name, accuracy))

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-mnist", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    for training_setup in (SGDSetup, SGDWithExponentialDecaySetup, SGDWithMomentumSetup, AdamSetup):
        for batch_size in training_setup.BATCH_SIZES:
            threads = []
            for learning_rate in training_setup.LEARNING_RATES:
                training_setup_inst = training_setup(batch_size, learning_rate, mnist.train.num_examples, args.epochs)
                threads.append(OptimizationThread(copy.deepcopy(mnist), training_setup_inst, args))
            for thread in threads:
                thread.start()
                time.sleep(2)
            for thread in threads:
                thread.join()
 
