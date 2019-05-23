import tensorflow as tf
import numpy as np


class Mnetwork:
    def __init__(self, sess, config):

        self.sess = sess

        self.fc_size1 = config.fc_size1
        self.learning_rate = config.mnet_lr

        self.gamma = config.gamma
        self.tau = config.tau

        self.nStates = config.nStates
        self.nActions = config.nActions

        # create network
        self.input_state, self.logits, self.prediction = self.build_network(scope_name='mnet')

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mnet')

        self.target_predictions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target_predictions))
        self.updateModel = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            input_state = tf.placeholder(tf.float32, shape=(None, self.nActions))

            net = tf.contrib.layers.fully_connected(input_state, self.fc_size1, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))
            logits = tf.contrib.layers.fully_connected(net, self.nActions, activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))

            prediction = tf.argmax(logits, axis=1)

        return input_state, logits, prediction

    def get_prediction(self, state):

        prediction = self.sess.run(self.prediction, feed_dict={
            self.input_state: [state]
        })
        return prediction

    def update(self, train_batch):
        # state_batch, action_batch, reward_batch, next_state_batch, termination_batch, label_batch = train_batch

        # state_batch = train_batch[0]
        # action_batch = train_batch[1]
        # reward_batch = train_batch[2]
        next_state_batch = train_batch[3]
        # termination_batch = train_batch[4]
        label_batch = train_batch[5]

        self.sess.run(self.updateModel, feed_dict={
            self.input_state: next_state_batch,
            self.target_predictions: label_batch
        })

    def save_network(self, save_dir):
        self.saver.save(self.sess, '{}_mnet_dt'.format(save_dir))

    def restore_network(self, load_dir):
        self.saver.restore(self.sess, '{}_mnet_dt'.format(load_dir))
