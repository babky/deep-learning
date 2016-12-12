#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
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


class Method(object):
    def __init__(self, uses_character_data):
        self._uses_character_data = uses_character_data

    def create_embedding(self, network, words, language):
        pass

    def get_training(self, network, epoch):
        return network.training

    def initialize_training(self, network, loss_masked):
        network.training = tf.train.AdamOptimizer().minimize(loss_masked, global_step=network.global_step)

    def update_forms(self, network, forms):
        return forms

    @property
    def uses_character_data(self):
        return self._uses_character_data


class LearnedWe(Method):
    EMBEDDING_SIZE = 128

    def __init__(self):
        super(LearnedWe, self).__init__(False)

    def create_embedding(self, network, words, language):
        network.embedding = tf.Variable(tf.random_uniform([len(words), self.EMBEDDING_SIZE], -1.0, 1.0))
        represented_sentences = tf.nn.embedding_lookup(network.embedding, network.forms)
        return represented_sentences


class OnlyPreTrainedWe(Method):
    def __init__(self):
        super(OnlyPreTrainedWe, self).__init__(False)

    def create_embedding(self, network, words, language):
        print('Loading the embeddings from {0}'.format(LANGUAGES[language]['embeddings']), file=sys.stderr)
        we = WordEmbeddings(LANGUAGES[language]['embeddings'])

        pad = len(we.words)
        unk = pad + 1

        perm = np.zeros([len(words)], dtype=int)
        for i in range(len(words)):
            if words[i] == '<pad>':
                perm[i] = pad
            elif words[i].lower() in we.words_map:
                perm[i] = we.words_map[words[i].lower()]
            else:
                # Unk
                perm[i] = unk
                # unk += 1

        embedding = np.concatenate(
            (we.we, np.random.uniform(low=-1.0, high=1.0, size=[1, we.dimension]), np.zeros([1, we.dimension])), axis=0)
        embedding = embedding.astype(dtype=np.float32)
        network.embedding_perm = perm
        network.embedding = tf.Variable(embedding, name='embedding', trainable=False)
        represented_sentences = tf.nn.embedding_lookup(network.embedding, network.forms)
        return represented_sentences

    def update_forms(self, network, forms):
        f = np.vectorize(lambda x: network.embedding_perm[x])
        return f(forms)


class UpdatedPreTrainedWe(OnlyPreTrainedWe):
    def create_embedding(self, network, words, language):
        network.started_embedding_training = False
        return super(UpdatedPreTrainedWe, self).create_embedding(network, words, language)

    def get_training(self, network, epoch):
        training = network.training
        if epoch > 1:
            if not network.started_embedding_training:
                print("Started the embedding training", file=sys.stderr)
                network.started_embedding_training = True
            training = network.training_with_embedding
        return training

    def initialize_training(self, network, loss_masked):
        super(UpdatedPreTrainedWe, self).initialize_training(network, loss_masked)
        trainable = tf.trainable_variables() + [network.embedding]
        training = tf.train.AdamOptimizer().minimize(loss_masked, global_step=network.global_step, var_list=trainable)
        network.training_with_embedding = training


class CharRnn(Method):
    def __init__(self):
        super(CharRnn, self).__init__(True)

    def create_embedding(self, network, words, language):
        pass


class CharConv(Method):
    def __init__(self):
        super(CharConv, self).__init__(True)

    def create_embedding(self, network, words, language):
        pass


class Network(object):
    TAG_COUNT = 19

    def learned_we(self):
        self.method = LearnedWe()

    def __init__(self, rnn_cell, rnn_cell_dim, method, words, logdir, expname, threads=1, seed=42, language='en',
                 layers=1):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed

        self.training = None
        self.embedding = None
        self.embedding_perm = None
        self.started_embedding_training = None
        self.training_with_embedding = None

        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.summary.FileWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            if rnn_cell == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
            elif rnn_cell == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell
                rnn_cell_fw = rnn_cell(rnn_cell_dim)
                rnn_cell_bw = rnn_cell(rnn_cell_dim)
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.forms = tf.placeholder(tf.int32, [None, None])
            self.tags = tf.placeholder(tf.int32, [None, None])

            self.method_name = method

            if method == 'learned_we':
                self.method = LearnedWe()
            elif method == 'only_pretrained_we':
                self.method = OnlyPreTrainedWe()
            elif method == 'updated_pretrained_we':
                self.method = UpdatedPreTrainedWe()
            else:
                NotImplemented("Method {0} not implemented".format(method))

            represented_sentences = self.method.create_embedding(self, words, language)

            # Go back and forth.
            combined_outputs = represented_sentences
            for layer in range(layers):
                scope_name = "layer_{0}".format(layer)
                with tf.variable_scope(scope_name):
                    rnn_out = tf.nn.bidirectional_dynamic_rnn(rnn_cell(rnn_cell_dim), rnn_cell(rnn_cell_dim),
                                                              combined_outputs, self.sentence_lens, dtype=tf.float32,
                                                              scope=scope_name)
                    (rnn_outputs), _ = rnn_out
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
            self.method.initialize_training(self, loss_masked)

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
        training = self.method.get_training(self, epoch)
        forms = self.method.update_forms(self, forms)

        _, summary = self.session.run([training, self.summary],
                                      {self.sentence_lens: sentence_lens, self.forms: forms,
                                       self.tags: tags, self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, forms, tags):
        forms = self.method.update_forms(self, forms)
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                             {self.sentence_lens: sentence_lens, self.forms: forms,
                                              self.tags: tags, self.dataset_name: "dev"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, forms):
        forms = self.method.update_forms(self, forms)
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens, self.forms: forms})


def main():
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
    parser.add_argument("--layers", default=1, type=int, help="Bidirectional layer count.")
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
                      words=data_train.factors[data_train.FORMS]['words'], logdir=args.logdir, expname=expname,
                      threads=args.threads, language=args.language, layers=args.layers)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            # To use character-level embeddings, pass including_charseqs=True to next_batch
            # and instead of word_ids[data_train.FORMS] use charseq_ids[data_train.FORMS],
            # charseqs[data_train.FORMS] and charseq_lens[data_train.FORMS]
            if network.method.uses_character_data:
                batch = data_train.next_batch(args.batch_size, including_charseqs=True)
                sentence_lens, word_ids, batch_charseq_ids, batch_charseqs, batch_charseq_lens = batch
                network.train(sentence_lens, word_ids[data_train.FORMS], word_ids[data_train.TAGS], epoch)
            else:
                sentence_lens, word_ids = data_train.next_batch(args.batch_size)
                network.train(sentence_lens, word_ids[data_train.FORMS], word_ids[data_train.TAGS], epoch)

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


if __name__ == "__main__":
    main()
