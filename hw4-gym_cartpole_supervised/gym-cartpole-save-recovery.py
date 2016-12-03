from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tensorflow.contrib.losses as tf_losses
import tensorflow.contrib.metrics as tf_metrics
import tensorflow.contrib.layers as tf_layers

class Network:
    OBSERVATIONS = 4

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

    def construct(self):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.actions = tf.placeholder(tf.int64, [None], name="actions")

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")
            self.dropout_keep_probability = tf.placeholder_with_default(1.0, ())

            # The real model should be here, currently only a random guess
            batch_size = tf.shape(self.observations)[0]

            hidden_layer_1 = tf_layers.fully_connected(self.observations, num_outputs=4, activation_fn=tf.nn.relu)
            dropout_layer_1 = tf.nn.dropout(hidden_layer_1, self.dropout_keep_probability)
            hidden_layer_2 = tf_layers.fully_connected(dropout_layer_1, num_outputs=4, activation_fn=tf.nn.relu)
            dropout_layer_2 = tf.nn.dropout(hidden_layer_2, self.dropout_keep_probability)
            output_layer = tf_layers.fully_connected(dropout_layer_2, num_outputs=2, activation_fn=tf.nn.relu, scope="output_layer")
            self.predictions = tf.argmax(output_layer, 1)

            # Loss function
            self.loss = tf_losses.sparse_softmax_cross_entropy(output_layer, self.actions, scope="loss")
            self.training = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
            self.accuracy = tf_metrics.accuracy(self.predictions, self.actions)

            # Summaries
            self.summaries = {"training": tf.merge_summary([tf.scalar_summary("train/loss", self.loss),
                                          tf.scalar_summary("train/accuracy", self.accuracy)])}

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/action", self.predictions)
            self.saver = tf.train.Saver(max_to_keep=None)

            # Initialize the variables
            self.session.run(tf.initialize_all_variables())

        # Finalize graph and log it if requested
        self.session.graph.finalize()
        if self.summary_writer:
            self.summary_writer.add_graph(self.session.graph)

    def train(self, observations, actions, summaries=False, run_metadata=False):
        if (summaries or run_metadata) and not self.summary_writer:
            raise ValueError("Logdir is required for summaries or run_metadata.")

        args = {"feed_dict": {self.observations: observations, self.actions: actions, self.dropout_keep_probability: 0.5}}
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

    # Save the graph
    def save(self, path):
        self.saver.save(self.session, path)

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def predict(self, observations):
        return self.session.run(self.predictions, {self.observations: [observations]})[0]

def train_network(network, observations, actions, epochs, summaries=False):
    observations = np.array(observations)
    actions = np.array(actions)
    batch_size = 50

    for e in range(epochs):
        permutation = np.random.permutation(len(observations))
        for batch in np.split(permutation, len(observations) / batch_size):
            network.train(observations[batch], actions[batch], network.training_step, summaries and (network.training_step == 0))

if __name__ == "__main__":
    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logs", type=str, help="Logdir name.")
    parser.add_argument("--exp", default="1-gym-save", type=str, help="Experiment name.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Construct the network
    network = Network(threads=args.threads, logdir=args.logdir, expname=args.exp)
    network.construct()

    # Load the data
    import csv
    csv.register_dialect('ssv', delimiter=' ', quoting=csv.QUOTE_NONE)
    observations = []
    actions = []
    with open('gym-cartpole-data.txt') as f:
        reader = csv.reader(f, 'ssv')
        for row in reader:
            observations.append(row[0:4])
            actions.append(row[4])

    import gym
    # Create the environment
    env = gym.make('CartPole-v1')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for epoch in range(10):
        print("Epoch {0}.".format(epoch))

        print("Training network.")
        # Train the network
        ## train_network(network, observations, actions, 5000 if epoch == 0 else 3, epoch == 0)
        train_network(network, observations, actions, 5000 if epoch == 0 else 1, epoch == 0)

        # Evaluate the episodes
        total_score = 0
        episodes = 1000 
        print("Evaluating the trained network.")
        best_observations = None
        best_score = 0
        best_actions = None

        for episode in range(episodes):
            observation = env.reset()
            score = 0
        
            my_observations = []
            my_actions = []

            for i in range(5000):
                action = network.predict(observation)
                observation, reward, done, info = env.step(action)
                my_observations.append(observation)
                my_actions.append(action)

                score += reward
                if done:
                    break

            if best_score < score:
                best_score = score
                best_actions = my_actions
                best_observations = my_observations

            total_score += score

        # Add the observation and learn it.
        if best_score >= 2000:
            print("Adding new observations.")
            observations += best_observations[0:200]
            actions += best_actions[0:200]

        print("Avg score: {0}, Best score: {1}".format(total_score / episodes, best_score))
        # Save the network
        network.save("1-gym-babka-{1}-epoch-{0}".format(epoch, timestamp))

    train_network(network, observations, actions, 3)
    network.save("1-gym-babka-{0}-completed".format(timestamp))

