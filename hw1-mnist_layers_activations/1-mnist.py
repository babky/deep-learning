from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics

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

    def construct(self, hidden_layer_size, hidden_layer_count, activation):
        self.hidden_layer_count = hidden_layer_count
        self.activation = activation
	suffix = "_{0}_{1}".format(hidden_layer_count, "tanh" if activation == tf.tanh else "relu")
	self._suffix = suffix

        with self.session.graph.as_default():
            with tf.name_scope("inputs" + suffix):
                self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name="images" + suffix)
                self.labels = tf.placeholder(tf.int64, [None], name="labels" + suffix)

            flattened_images = tf_layers.flatten(self.images, scope="preprocessing" + suffix)

            # Hyperparameter optimization - connect the hidden layers
            previous_layer = flattened_images
            for i in range(hidden_layer_count):
                hidden_layer = tf_layers.fully_connected(previous_layer, num_outputs=hidden_layer_size, activation_fn=activation, scope="hidden_layer_{0}".format(i) + suffix)
                previous_layer = hidden_layer

            output_layer = tf_layers.fully_connected(hidden_layer, num_outputs=self.LABELS, activation_fn=None, scope="output_layer" + suffix)
            self.predictions = tf.argmax(output_layer, 1)

            loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.labels, scope="loss" + suffix)
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step" + suffix)
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.labels)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss" + suffix, loss),
                                                            tf.scalar_summary("train/accuracy" + suffix, self.accuracy)])}
            for dataset in ["dev", "test"]:
                self.summaries[dataset] = tf.scalar_summary(dataset+"/accuracy"+suffix, self.accuracy)

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


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-mnist", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    from tensorflow.examples.tutorials.mnist import input_data

    best_network = None
    worst_network = None
    best_accuracy = -1
    worst_accuracy = -1

    for hidden_layers in range(1, 4):
        for activation in (tf.tanh, tf.nn.relu):
            mnist = input_data.read_data_sets("mnist_data/", reshape=False)

            # Construct the network
            network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
            network.construct(100, hidden_layers, activation)

            # Train
            for i in range(args.epochs):
                while mnist.train.epochs_completed == i:
                    images, labels = mnist.train.next_batch(args.batch_size)
                    network.train(images, labels, network.training_step % 100 == 0, network.training_step == 0)

                network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)
                # network.evaluate("test", mnist.test.images, mnist.test.labels, True)

            # Compute the accuracy of the trained network
            accuracy = network.evaluate("dev", mnist.validation.images, mnist.validation.labels, True)

            if best_network is None:
                best_network = network
                best_accuracy = accuracy

                worst_network = network
                worst_accuracy = accuracy
            else:
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_network = network

                if worst_accuracy > accuracy:
                    worst_accuracy = accuracy
                    worst_network = network

    best_result_on_test = best_network.evaluate("test", mnist.test.images, mnist.test.labels, True)
    print("Best result is for {0}/{1} with accuracy {2}.".format(best_network.hidden_layer_count, best_network.activation, best_result_on_test))

    worst_result_on_test = worst_network.evaluate("test", mnist.test.images, mnist.test.labels, True)
    print("Worst result is for {0}/{1} with accuracy {2}.".format(worst_network.hidden_layer_count, worst_network.activation, worst_result_on_test))
