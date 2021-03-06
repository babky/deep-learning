#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime

import numpy
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.metrics as tf_metrics


class Dataset(object):
    def __init__(self, filename, alphabet=None):
        # Load the sentences
        sentences = []
        with open(filename, "r") as file:
            for line in file:
                sentences.append(line.rstrip("\r\n"))

        # Compute sentence lengths
        self._sentence_lens = np.zeros([len(sentences)], np.int32)
        for i in range(len(sentences)):
            self._sentence_lens[i] = len(sentences[i])
        max_sentence_len = np.max(self._sentence_lens)

        # Create alphabet_map
        alphabet_map = {'<pad>': 0, '<unk>': 1}
        if alphabet is not None:
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index

        # Remap input characters using the alphabet_map
        self._sentences = np.zeros([len(sentences), max_sentence_len], np.int32)
        self._labels = np.zeros([len(sentences), max_sentence_len], np.int32)
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                char = sentences[i][j].lower()
                if char not in alphabet_map:
                    if alphabet is None:
                        alphabet_map[char] = len(alphabet_map)
                    else:
                        char = '<unk>'
                self._sentences[i, j] = alphabet_map[char]
                self._labels[i, j] = 0 if sentences[i][j].lower() == sentences[i][j] else 1

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._sentences))

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def sentences(self):
        return self._sentences

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]
        batch_len = np.max(self._sentence_lens[batch_perm])
        return self._sentences[batch_perm, 0:batch_len], self._sentence_lens[batch_perm], self._labels[batch_perm,
                                                                                          0:batch_len]

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentences))
            return True
        return False


class Network(object):
    def __init__(self, embedding, alphabet_size, rnn_cell, rnn_cell_dim, logdir, expname, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.train.SummaryWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentences = tf.placeholder(tf.int32, [None, None])
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.labels = tf.placeholder(tf.int64, [None, None])

            # Embedding
            if embedding == "one-hot":
                represented_sentences = tf.one_hot(self.sentences, alphabet_size)
            elif isinstance(embedding, tuple) and embedding[0] == "lookup":
                embedding_size = embedding[1]
                representation = tf.Variable(tf.random_uniform([alphabet_size, embedding_size], -1.0, 1.0))
                represented_sentences = tf.nn.embedding_lookup(representation, self.sentences)
            else:
                raise ValueError("Unknown embedding {}".format(embedding))

            # RNN
            rnn_outputs, output_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, represented_sentences,
                                                                         self.sentence_lens, dtype=tf.float32)
            combined_outputs = tf.concat(2, rnn_outputs)

            # Linear activation
            outputs = tf_layers.fully_connected(combined_outputs, 2, activation_fn=None)

            # Predictions
            softmax = tf.nn.softmax(outputs)
            self.predictions = tf.argmax(softmax, 2)

            # Mask so that a valid loss is computed
            mask = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)
            boolean_mask = tf.cast(mask, tf.bool)
            labels_masked = tf.boolean_mask(self.labels, boolean_mask)

            # Accuracy
            predictions_masked = tf.boolean_mask(self.predictions, boolean_mask)
            self.accuracy = tf_metrics.accuracy(predictions_masked, labels_masked)

            # Training & loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, self.labels)
            loss_masked = tf.boolean_mask(loss, boolean_mask)
            self.training = tf.train.AdamOptimizer().minimize(loss_masked, global_step=self.global_step)

            # Summaries
            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.scalar_summary(self.dataset_name + "/accuracy", self.accuracy)
            self.summary_writer.add_graph(self.session.graph)

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentences, sentence_lens, labels):
        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentences: sentences, self.sentence_lens: sentence_lens,
                                       self.labels: labels, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentences, sentence_lens, labels, dataset_name):
        summary, accuracy = self.session.run([self.summary, self.accuracy],
                                             {self.sentences: sentences, self.sentence_lens: sentence_lens,
                                              self.labels: labels, self.dataset_name: dataset_name})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def compute(self, sentences, sentence_lens):
        predictions = self.session.run(self.predictions, {self.sentences: sentences, self.sentence_lens: sentence_lens})
        return predictions


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size.")
    parser.add_argument("--data_train", default="en-ud-train.txt", type=str, help="Training data file.")
    parser.add_argument("--data_dev", default="en-ud-dev.txt", type=str, help="Development data file.")
    parser.add_argument("--data_test", default="en-ud-test.txt", type=str, help="Testing data file.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    data_train = Dataset(args.data_train)
    data_dev = Dataset(args.data_dev, data_train.alphabet)
    data_test = Dataset(args.data_test, data_train.alphabet)


    def evaluate_hyper_parameters(embedding, rnn_cell, rnn_cell_dim):
        # Construct the network
        expname = "uppercase-letters-{}{}-{}-bs{}-epochs{}".format(rnn_cell, rnn_cell_dim, embedding,
                                                                   args.batch_size, args.epochs)
        network = Network(embedding, alphabet_size=len(data_train.alphabet), rnn_cell=rnn_cell,
                          rnn_cell_dim=rnn_cell_dim,
                          logdir=args.logdir, expname=expname, threads=args.threads)
        # Train
        for epoch in range(args.epochs):
            print("Training epoch {}".format(epoch))
            while not data_train.epoch_finished():
                sentences, sentence_lens, labels = data_train.next_batch(args.batch_size)
                network.train(sentences, sentence_lens, labels)

            network.evaluate(data_dev.sentences, data_dev.sentence_lens, data_dev.labels, "dev")
            network.evaluate(data_test.sentences, data_test.sentence_lens, data_test.labels, "test")

        return network


    network = None
    performance = 0
    for rnn_cell in ('GRU', 'LSTM'):
        for rnn_cell_dim in (96, 64):
            for embedding in (('lookup', 64), ('lookup', 16), 'one-hot'):
                n = evaluate_hyper_parameters(embedding, rnn_cell, rnn_cell_dim)
                p = n.evaluate(data_dev.sentences, data_dev.sentence_lens, data_dev.labels, "dev")
                if performance < p:
                    network = n
                    performance = p

    print("TF ACCURACY\nAccuracy: {0}\n".format(network.evaluate(data_test.sentences, data_test.sentence_lens, data_test.labels, "test")))

    # Manually verify the correctness...
    predictions = network.compute(data_test.sentences, data_test.sentence_lens)
    errors = 0
    for i in range(len(data_test.sentence_lens)):
        length = data_test.sentence_lens[i]
        prediction = predictions[i][0:length]
        label = data_test.labels[i][0:length]
        errors += numpy.sum((prediction - label) ** 2)
    lens = numpy.sum(data_test.sentence_lens)
    print("MANUAL ACCURACY\nErrors: {0}, Lens: {1}, Accuracy: {2}".format(errors, lens, 1 - errors / lens))
