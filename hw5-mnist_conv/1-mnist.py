from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import argparse
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics
from tensorflow.examples.tutorials.mnist import input_data

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

    def _add_3_double_convolution(self, inputs, channels):
        inner_conv = tf_layers.convolution2d(inputs=inputs, num_outputs=channels, kernel_size=3, stride=1, normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
        outer_conv = tf_layers.convolution2d(inputs=inner_conv, num_outputs=channels, kernel_size=3, stride=1, normalizer_fn=tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
        max_pooling = tf.contrib.layers.max_pool2d(inputs=outer_conv, kernel_size=3, stride=2)
    	return max_pooling

    def construct(self, epochs, train_size, batch_size):
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images")
                self.labels = tf.placeholder(tf.int64, [None], name="labels")

            self.training_flag = tf.placeholder(tf.bool)
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.dropout_keep_probability = tf.placeholder(tf.float32)
            self.learning_rate = tf.train.exponential_decay(0.002, self.global_step, epochs * train_size / batch_size, 0.1)

            # We have 28 x 28 x 1 in self.images
            # Use double 3 x 3 x 7 convolution
            inner_conv = self._add_3_double_convolution(self.images, 9)
            # Usr double 3 x 3 x 13 convolution
            outer_conv = self._add_3_double_convolution(inner_conv, 17)

            # Flatten into fully connected layer
            flattened_conv = tf_layers.flatten(outer_conv)
            flattened_conv_with_dropout = tf.nn.dropout(flattened_conv, self.dropout_keep_probability)
            hidden_layer = flattened_conv_with_dropout

            # Add dropout if necessary
            output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=10, activation_fn=None, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            # Loss function
            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss")

            # Add learning rate.
            self.training = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss, global_step=self.global_step)
            ## self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
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

        args = {"feed_dict": {self.images: images, self.labels: labels, self.training_flag: True, self.dropout_keep_probability: 0.5}}
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

        results = self.session.run(targets, {self.images: images, self.labels: labels, self.training_flag: False, self.dropout_keep_probability: 1})
        if summaries:
            self.summary_writer.add_summary(results[-1], self.training_step)
        return results[0]


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-mnist", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    mnist = input_data.read_data_sets("mnist_data/", reshape=False)

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct(args.epochs, mnist.train.num_examples, args.batch_size)

    # Train
    for i in range(args.epochs):
        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

        network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
        test_acc = network.evaluate("test", mnist.test.images, mnist.test.labels, True)
        if test_acc > 0.994:
            break

    print(network.evaluate("test", mnist.test.images, mnist.test.labels, False))

