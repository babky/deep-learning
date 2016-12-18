#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import sys
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from contrib_seq2seq import dynamic_rnn_decoder

import morpho_dataset

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


class Network(object):
    CHAR_EMBEDDING_SIZE = 32
    MAX_LENGTH = 512

    def __init__(self, rnn_cell, rnn_cell_dim, num_chars, bow_char, eow_char, logdir, expname, threads=1, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.summary_writer = tf.summary.FileWriter("{}/{}-{}".format(logdir, timestamp, expname), flush_secs=10)

        # Construct the graph
        with self.session.graph.as_default():
            self.alphabet_size = num_chars
            if rnn_cell == "LSTM":
                rnn_cell_type = tf.nn.rnn_cell.LSTMCell
            elif rnn_cell == "GRU":
                rnn_cell_type = tf.nn.rnn_cell.GRUCell
            else:
                raise ValueError("Unknown rnn_cell {}".format(rnn_cell))

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.sentence_lens = tf.placeholder(tf.int32, [None])
            self.form_ids = tf.placeholder(tf.int32, [None, None])
            self.forms = tf.placeholder(tf.int32, [None, None])
            self.form_lens = tf.placeholder(tf.int32, [None])
            self.lemma_ids = tf.placeholder(tf.int32, [None, None])
            self.lemmas = tf.placeholder(tf.int32, [None, None])
            self.lemma_lens = tf.placeholder(tf.int32, [None])

            # The tensor of [sentence, word, word_embedding_feature] shape.
            word_embeddings = self._create_embedding(rnn_cell_type, rnn_cell_dim)
            cell_fw = rnn_cell_type(rnn_cell_dim)
            rnn_outputs, rnn_state = tf.nn.dynamic_rnn(cell_fw, word_embeddings, self.sentence_lens, dtype=tf.float32)
            # rnn_outputs = tf.concat(2, rnn_outputs)
            # rnn_states = tf.concat(1, rnn_states)

            cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
            with tf.variable_scope("rnn_decoding"):
                df = self._decoder_fn_train(rnn_outputs)
                inputs = tf.nn.embedding_lookup(self.lemmas, self.lemma_ids)
                training_logits, states = dynamic_rnn_decoder(cell, df, inputs=self.lemmas, sequence_length=self.lemma_lens)

            with tf.variable_scope("rnn_decoding", reuse=True):
                df = self._decoder_fn_inference(rnn_state)
                inference_logits, states = dynamic_rnn_decoder(cell, df)

            print(inference_logits)
                # output = tf.argmax(inference_logits, 2)

            labels_masked = tf.boolean_mask(self.labels, boolean_mask)

            # Accuracy
            predictions_masked = tf.boolean_mask(self.predictions, boolean_mask)
            self.accuracy = tf_metrics.accuracy(predictions_masked, labels_masked)

            loss_unmasked = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(training_logits, lemma_embeddings))
            mask = tf.sequence_mask(self.sentence_lens)

            self.loss = tf.boolean_mask(loss_unmasked, mask)
            self.training = tf.train.AdamOptimizer().minimize(self.loss)
            self.accuracy = None

            self.dataset_name = tf.placeholder(tf.string, [])
            self.summary = tf.merge_summary([tf.scalar_summary(self.dataset_name + "/loss", loss_unmasked),
                                             tf.scalar_summary(self.dataset_name + "/accuracy", self.accuracy)])

            # Initialize variables
            self.session.run(tf.initialize_all_variables())

    def _create_embedding(self, rnn_cell_type, rnn_cell_dim):
        self.char_embeddings = tf.Variable(tf.random_uniform([self.alphabet_size, self.CHAR_EMBEDDING_SIZE], -1.0, 1.0))
        represented_forms = tf.nn.embedding_lookup(self.char_embeddings, self.forms)
        _rnn_out, rnn_states = tf.nn.bidirectional_dynamic_rnn(rnn_cell_type(rnn_cell_dim),
                                                               rnn_cell_type(rnn_cell_dim),
                                                               represented_forms, self.form_lens,
                                                               dtype=tf.float32,
                                                               scope="rnn_embedding")
        word_embeddings = tf.concat(1, rnn_states)
        self.embedding = tf.nn.embedding_lookup(word_embeddings, self.form_ids)
        return self.embedding

    def _decoder_fn_train(self, rnn_outputs):
        def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
            cell_output = self._output_fn(cell_output)
            if cell_state is None:  # first call, return encoder_state
                cell_state = rnn_outputs[time]
            cell_input = self._input_fn(cell_input)
            return None, cell_state, cell_input, cell_output, context_state

        return decoder_fn

    def _decoder_fn_inference(self, encoder_state, bow_char, pad_char=0):
        batch_size = tf.shape(encoder_state)[0]
        max_length = self.MAX_LENGTH

        def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
            cell_output = self._output_fn(cell_output)
            if cell_state is None:
                cell_state = encoder_state
                done = tf.zeros([batch_size], dtype=tf.bool)
                cell_input = tf.tile([bow_char], [batch_size])
            else:
                cell_input = tf.argmax(cell_output, 1)
                done = tf.equal(cell_input, pad_char)
                done = tf.cond(tf.greater_equal(time, max_length),  # return true if time >= maxlen
                               lambda: tf.ones([batch_size], dtype=tf.bool),
                               lambda: done)

            cell_input = self._input_fn(cell_input)
            return done, cell_state, cell_input, cell_output, context_state

        return decoder_fn

    def _input_fn(self, cell_input):
        return tf.nn.embedding_lookup(self.char_embeddings, cell_input)

    # Output function (makes logits out of rnn outputs)
    def _output_fn(self, cell_output):
        if cell_output is None:
            return tf.zeros([self.alphabet_size], tf.float32)  # only used for shape inference
        else:
            return tf_layers.linear(cell_output, num_outputs=self.alphabet_size, scope="rnn_output")

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, sentence_lens, form_ids, forms, form_lens, lemma_ids, lemmas, lemma_lens):
        _, summary = self.session.run([self.training, self.summary],
                                      {self.sentence_lens: sentence_lens,
                                       self.form_ids: form_ids, self.forms: forms, self.form_lens: form_lens,
                                       self.lemma_ids: lemma_ids, self.lemmas: lemmas, self.lemma_lens: lemma_lens,
                                       self.dataset_name: "train"})
        self.summary_writer.add_summary(summary, self.training_step)

    def evaluate(self, sentence_lens, form_ids, forms, form_lens, lemma_ids, lemmas, lemma_lens):
        accuracy, summary = self.session.run([self.accuracy, self.summary],
                                             {self.sentence_lens: sentence_lens,
                                              self.form_ids: form_ids, self.forms: forms, self.form_lens: form_lens,
                                              self.lemma_ids: lemma_ids, self.lemmas: lemmas,
                                              self.lemma_lens: lemma_lens,
                                              self.dataset_name: "dev"})
        self.summary_writer.add_summary(summary, self.training_step)
        return accuracy

    def predict(self, sentence_lens, form_ids, forms, form_lens):
        return self.session.run(self.predictions,
                                {self.sentence_lens: sentence_lens,
                                 self.form_ids: form_ids, self.forms: forms, self.form_lens: form_lens})


if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--language", default='en', type=str, help="Language")
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

    test_forms = data_test.factors[data_test.FORMS]['strings']
    bow_char = data_train.alphabet.index("<bow>")
    eow_char = data_train.alphabet.index("<eow>")

    # Construct the network
    print("Constructing the network.", file=sys.stderr)
    expname = "lemmatizer-{}{}-bs{}-epochs{}".format(args.rnn_cell, args.rnn_cell_dim, args.batch_size, args.epochs)
    network = Network(rnn_cell=args.rnn_cell, rnn_cell_dim=args.rnn_cell_dim,
                      num_chars=len(data_train.alphabet), bow_char=bow_char, eow_char=eow_char,
                      logdir=args.logdir, expname=expname, threads=args.threads)

    # Train
    best_dev_accuracy = 0
    test_predictions = None

    for epoch in range(args.epochs):
        print("Training epoch {}".format(epoch + 1), file=sys.stderr)
        while not data_train.epoch_finished():
            sentence_lens, form_ids, charseq_ids, charseqs, charseq_lens = \
                data_train.next_batch(args.batch_size, including_charseqs=True)
            network.train(sentence_lens, charseq_ids[data_train.FORMS], charseqs[data_train.FORMS],
                          charseq_lens[data_train.FORMS], charseq_ids[data_train.LEMMAS], charseqs[data_train.LEMMAS],
                          charseq_lens[data_train.LEMMAS])

        sentence_lens, form_ids, charseq_ids, charseqs, charseq_lens = data_dev.whole_data_as_batch(
            including_charseqs=True)
        dev_accuracy = network.evaluate(sentence_lens, charseq_ids[data_train.FORMS], charseqs[data_train.FORMS],
                                        charseq_lens[data_train.FORMS], charseq_ids[data_train.LEMMAS],
                                        charseqs[data_train.LEMMAS], charseq_lens[data_train.LEMMAS])
        print("Development accuracy after epoch {} is {:.2f}.".format(epoch + 1, 100. * dev_accuracy), file=sys.stderr)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            sentence_lens, form_ids, charseq_ids, charseqs, charseq_lens = data_test.whole_data_as_batch(
                including_charseqs=True)
            test_predictions = network.predict(sentence_lens,
                                               charseq_ids[data_train.FORMS], charseqs[data_train.FORMS],
                                               charseq_lens[data_train.FORMS])

    # Print test predictions
    # We use strings instead of words, because words can be <unk>
    test_forms = data_test.factors[data_test.FORMS]['strings']
    for i in range(len(data_test.sentence_lens)):
        for j in range(data_test.sentence_lens[i]):
            lemma = ''
            for k in range(len(test_predictions[i][j])):
                if test_predictions[i][j][k] == eow_char:
                    break
                lemma += data_test.alphabet[test_predictions[i][j][k]]
            print("{}\t{}\t_".format(test_forms[i][j], lemma))
        print()
