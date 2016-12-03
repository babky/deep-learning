#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

class Network:
    DATA = 144
    TEST = 40
    TRAIN = DATA - TEST
    def __init__(self, rnn_cell, rnn_cell_dim, logdir, expname, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            with tf.name_scope("inputs"):
                self.inputs = tf.placeholder(tf.float32, [1, self.TRAIN], name="input_sequence")

            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

            # First inputs are zeros
            zero_state = rnn_cell.zero_state(1, tf.float32)
            zero_input = tf.zeros((1,1))
            with tf.variable_scope("rnn"):
                cell_output, cell_state = rnn_cell(zero_input, zero_state)
                output = tf_layers.fully_connected(cell_output, 1, activation_fn=None, scope="rnn")

            # Part used for training
            # unpacked_inputs = tf.unpack(self.inputs, axis=1)
            unpacked_predictions = [output]
            for i in range(self.TRAIN - 1):
                # cell_input = tf.reshape(unpacked_inputs[i], (1, 1))
                cell_input = tf.reshape(self.inputs[0, i], (1, 1))
                with tf.variable_scope("rnn", reuse=True):
                    cell = rnn_cell(cell_input, cell_state)
                    cell_output, cell_state = cell
                    output = tf_layers.fully_connected(cell_output, 1, activation_fn=None, reuse=True, scope="rnn")
                unpacked_predictions.append(output)

            # Results
            unpacked_results = []
            for i in range(self.TEST):
                with tf.variable_scope("rnn", reuse=True):
                    cell_output, cell_state = rnn_cell(output, cell_state)
                    output = tf_layers.fully_connected(cell_output, 1, activation_fn=None, reuse=True, scope="rnn")
                unpacked_results.append(output)

            # Training
            self.predictions = tf.concat(1, unpacked_predictions)
            self.results = tf.concat(1, unpacked_predictions + unpacked_results)

            self.mse = tf.contrib.losses.mean_squared_error(self.predictions, self.inputs)
            self.training = tf.train.AdamOptimizer().minimize(self.mse, global_step=self.global_step)

            # Image summaries
            self.image_tag = tf.placeholder(tf.string, [])
            self.image_data = tf.placeholder(tf.uint8, [self.DATA, self.DATA, 3])
            self.summarize_image = tf.image_summary(self.image_tag, tf.expand_dims(self.image_data, 0))

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, train_sequence):
        args = {"feed_dict": {self.inputs: (train_sequence,)}}
        targets = [self.training]
        results = self.session.run(targets, **args)

    def predict(self, train_sequence):
        args = {"feed_dict": {self.inputs: (train_sequence,)}}
        results = self.session.run([self.results], **args)
        return results[0][0]

    def image_summary(self, gold, predictions, epoch):
        min_value = min(np.min(gold), np.min(predictions))
        max_value = max(np.max(gold), np.max(predictions))
        def scale(value):
            return int(self.DATA - self.DATA * (value - min_value) / (max_value - min_value + 1) - 1)

        image_data = np.full([self.DATA, self.DATA, 3], 255, dtype=np.uint8)
        for i in range(self.DATA):
            image_data[scale(gold[i]), i] = [0, 0, 255] if i < self.TRAIN else [0, 255, 0]
            image_data[scale(predictions[i]), i] = [255, 0, 0]

        self.summary_writer.add_summary(
            self.session.run(self.summarize_image, {self.image_tag: "predictions-{:02}".format(epoch), self.image_data: image_data})
        )

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="international-airline-passengers.tsv", type=str, help="Data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    # parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    # parser.add_argument("--rnn_cell_dim", default=10, type=int, help="RNN cell dimension.")
    parser.add_argument("--steps_per_epoch", default=500, type=int, help="Training steps per epoch.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    with open(args.data, "r") as file:
        header = file.readline()
        body = file.readlines()[:Network.DATA]
        assert(len(body) == Network.DATA)
        data_all = np.array([float(line.split("\t")[1]) for line in body], dtype=np.float32)
        data_min, data_max  = np.min(data_all), np.max(data_all)
        data_all = (data_all - data_min) / (data_max - data_min)
        data_train = data_all[:Network.TRAIN]
        data_test = data_all[Network.TRAIN:]

    for rnn_cell in ('LSTM', 'GRU'):
        for rnn_cell_dim in (8, 10, 12):
            # Construct the network
            expname = "sequence-generation-{}{}-epochs{}-per_epoch{}".format(rnn_cell, rnn_cell_dim, args.epochs, args.steps_per_epoch)
            network = Network(rnn_cell=rnn_cell, rnn_cell_dim=rnn_cell_dim, logdir=args.logdir, expname=expname, threads=args.threads)

            # Train
            for epoch in range(args.epochs):
                for step in range(args.steps_per_epoch):
                    network.train(data_train)

                predictions = network.predict(data_train)
                network.image_summary(data_all, predictions, epoch)

