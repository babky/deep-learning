#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import numpy.random as np_random
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.metrics as tf_metrics
import tensorflow.contrib.losses as tf_losses

import morpho_dataset

from word_embeddings import WordEmbeddings

LANGUAGES = {
    'en': {
        'dataset': {
            'dev': 'en-dev.txt',
            'train': 'en-train.txt',
            'test': 'en-test.txt'
        },
        'embeddings': 'word_embedding-en.txt'
    },
    'cs': {
        'dataset': {
            'dev': 'cs-dev.txt',
            'train': 'cs-train.txt',
            'test': 'cs-test.txt'
        },
        'embeddings': 'word_embedding-cs.txt'
    }
}


class Network:
    TAG_COUNT = 19
    PRETRAINED_EMBEDDING_SIZE = 100

    def learned_we(self, words):
        embedding_size = 128
        self.embedding = tf.Variable(tf.random_uniform([words, embedding_size], -1.0, 1.0))
        represented_sentences = tf.nn.embedding_lookup(self.embedding, self.forms)
        return represented_sentences

    def only_pretrained_we(self, words, language):
        print('Loading the embeddings from {0}'.format(LANGUAGES[language]['embeddings']), file=sys.stderr)
        embedding_size = self.PRETRAINED_EMBEDDING_SIZE
        we = WordEmbeddings(LANGUAGES[language]['embeddings'])
        embedding = np.concatenate((we.we, np.zeros([words - len(we.we), embedding_size])), axis=0)
        embedding = embedding.astype(np.float32)
        self.embedding = tf.Variable(embedding, name='embedding', trainable=False)
        represented_sentences = tf.nn.embedding_lookup(self.embedding, self.forms)
        return represented_sentences

    def updated_pretrained_we(self, words, language):
        self.started_embedding_training = False
        return self.only_pretrained_we(words, language)

    def __init__(self, rnn_cell, rnn_cell_dim, method, words, logdir, expname, threads=1, seed=42, language='cs'):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.summary.FileWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.forms = tf.placeholder(tf.int32, [None, None])
            self.tags = tf.placeholder(tf.int32, [None, None])

            self.method = method

            if method == 'learned_we':
                represented_sentences = self.learned_we(words)
            elif method == 'only_pretrained_we':
                represented_sentences = self.only_pretrained_we(words, language)
            elif method == 'updated_pretrained_we':
                represented_sentences = self.updated_pretrained_we(words, language)
            else:
                NotImplemented("Method {0} not implemented".format(method))

            # Go back and forth.
            rnn_out = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, represented_sentences, self.sentence_lens,
                                                      dtype=tf.float32)
            (rnn_outputs), (state_bw, state_fw) = rnn_out
            combined_outputs = tf.concat(2, rnn_outputs)

            # Linear activation
            outputs = tf_layers.fully_connected(combined_outputs, self.TAG_COUNT, activation_fn=None)

            # Predictions
            softmax = tf.nn.softmax(outputs)
            self.predictions = tf.cast(tf.argmax(softmax, 2), dtype=tf.int32)

            # Mask so that a valid loss is computed
            mask = tf.sequence_mask(self.sentence_lens)
            tags_masked = tf.boolean_mask(self.tags, mask)

            # Accuracy
            predictions_masked = tf.boolean_mask(self.predictions, mask)
            self.accuracy = tf_metrics.accuracy(predictions_masked, tags_masked)

            # Training & loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, self.tags)
            loss_masked = tf.boolean_mask(loss, mask)
            self.training = tf.train.AdamOptimizer().minimize(loss_masked, global_step=self.global_step)

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.merge_summary(
                [tf.scalar_summary(self.dataset_name + "/loss", tf_losses.compute_weighted_loss(loss_masked)),
                 tf.scalar_summary(self.dataset_name + "/accuracy", self.accuracy)])

            # Initialize variables
            self.session.run(tf.global_variables_initializer())

        self.session.graph.finalize()
        self.summary_writer.add_graph(self.session.graph)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentence_lens, forms, tags, epoch):
        if self.method == 'updated_pretrained_we' and epoch > 1 and not self.started_embedding_training:
            print("Started the embedding training", file=sys.stderr)
            self.started_embedding_training = True
            tf.trainable_variables().append(self.embedding)

        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentence_lens: sentence_lens, self.forms: forms,
                                       self.tags: tags, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, forms, tags):
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                             {self.sentence_lens: sentence_lens, self.forms: forms,
                                              self.tags: tags, self.dataset_name: "dev"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, forms):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens, self.forms: forms})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--language", default=10, type=str, help="Language")
    parser.add_argument("--method", default="learned_we", type=str, help="Which method of word embeddings to use.")
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--rnn_cell", default="GRU", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_cell_dim", default=100, type=int, help="RNN cell dimension.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Load the data
    print("Loading the data for language {0}.".format(args.language), file=sys.stderr)
    data_train = morpho_dataset.MorphoDataset(LANGUAGES[args.language]['dataset']['train'], add_bow_eow=True)
    data_dev = morpho_dataset.MorphoDataset(LANGUAGES[args.language]['dataset']['dev'], add_bow_eow=True,
                                            train=data_train)
    data_test = morpho_dataset.MorphoDataset(LANGUAGES[args.language]['dataset']['test'], add_bow_eow=True,
                                             train=data_train)

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "tagger-{}{}-m{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.method, args.batch_size,
                                                     args.epochs)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim, method=args.method,
                      words=len(data_train.factors[data_train.FORMS]['words']),
                      logdir=args.logdir, expname=expname, threads=args.threads, language=args.language)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, word_ids = data_train.next_batch(args.batch_size)
            network.train(sentence_lens, word_ids[data_train.FORMS], word_ids[data_train.TAGS], epoch)
            # To use character-level embeddings, pass including_charseqs=True to next_batch
            # and instead of word_ids[data_train.FORMS] use charseq_ids[data_train.FORMS],
            # charseqs[data_train.FORMS] and charseq_lens[data_train.FORMS]

        dev_sentence_lens, dev_word_ids = data_dev.whole_data_as_batch()
        dev_accuracy = network.evaluate(dev_sentence_lens, dev_word_ids[data_dev.FORMS], dev_word_ids[data_dev.TAGS])
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            test_sentence_lens, test_word_ids = data_test.whole_data_as_batch()
            test_predictions = network.predict(test_sentence_lens, test_word_ids[data_test.FORMS])

    # Print test predictions
    test_forms = data_test.factors[data_test.FORMS][
        'strings']  # We use strings instead of words, because words can be <unk>
    test_tags = data_test.factors[data_test.TAGS]['words']
    for i in range(len(data_test.sentence_lens)):
        for j in range(data_test.sentence_lens[i]):
            print("{}\t_\t{}".format(test_forms[i][j], test_tags[test_predictions[i, j]]))
        print()
