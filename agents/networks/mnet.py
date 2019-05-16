import tensorflow as tf
import numpy as np

class Mnetwork:
    def __init__(self, sess, config):

        self.sess = sess

        # before lstm
        self.fc_size1 = config.fc_size1
        self.fc_size2 = config.fc_size2
        self.h_size = config.h_size

        self.learning_rate = config.mnet_lr
        self.gamma = config.gamma
        self.tau = config.tau
        self.nStates = config.nStates  # 21
        self.nActions = config.nActions  # 10

        # create network
        self.input_obs, self.input_rnn_state, self.current_rnn_state, \
            self.batch_size, self.train_length, \
            self.prediction = self.build_network(scope_name='mnet')

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mnet')

        self.target_prediction = tf.placeholder(shape=[None], dtype=tf.int32)
        target_prediction_onehot = tf.one_hot(self.target_prediction, self.nStates, dtype=tf.float32)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_prediction_onehot, logits=self.prediction))
        self.updateModel = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            input_obs = tf.placeholder(tf.float32, shape=(None, self.nStates + self.nActions))

            # 4 fc layers (1 more layer than qnet)
            net = tf.contrib.layers.fully_connected(input_obs, self.fc_size1, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))
            net = tf.contrib.layers.fully_connected(net, self.fc_size2, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))
            net = tf.contrib.layers.fully_connected(net, self.h_size, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))

            # lstm layer
            batch_size = tf.placeholder(dtype=tf.int32, shape=[])
            train_length = tf.placeholder(dtype=tf.int32)

            # The input must be reshaped into [batch x trace x units] for rnn processing,
            # and then returned to [batch x units] when sent through the upper levels.
            net = tf.reshape(net, [batch_size, train_length, self.h_size])

            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.h_size, state_is_tuple=True)
            input_rnn_state = lstm_cell.zero_state(batch_size, tf.float32)
            net, current_rnn_state = tf.nn.dynamic_rnn(inputs=net, cell=lstm_cell, dtype=tf.float32, initial_state=input_rnn_state)
            net = tf.reshape(net, shape=[-1, self.h_size])

            prediction = tf.contrib.layers.fully_connected(net, self.nStates, activation_fn=None,
                                                           weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                               factor=1.0, mode="FAN_IN", uniform=True),
                                                           weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                           biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                               factor=1.0, mode="FAN_IN", uniform=True))
            # prediction = tf.contrib.layers.fully_connected(net, self.nStates, activation_fn=tf.nn.softmax)

        return input_obs, input_rnn_state, current_rnn_state, batch_size, train_length, prediction

    def get_prediction(self, obs, input_rnn_state):

        train_length = 1
        batch_size = 1

        prediction, rnn_state = self.sess.run([self.prediction, self.current_rnn_state], feed_dict={
            self.input_obs: obs,
            self.input_rnn_state: input_rnn_state,
            self.train_length: train_length,
            self.batch_size: batch_size
        })

        return prediction, rnn_state

    def update(self, train_batch, trace_length, batch_size):
        # trace_length : 4
        # batch_size : 4

        state_train = (np.zeros([batch_size, self.h_size]), np.zeros([batch_size, self.h_size]))

        # obs_batch = train_batch[:, 0]
        # action_batch = train_batch[:, 1]
        # reward_batch = train_batch[:, 2]
        next_obs_batch = train_batch[:, 3]
        # termination_batch = train_batch[:, 4]
        true_state_batch = train_batch[:, 5]

        self.sess.run(self.updateModel, feed_dict={
            self.input_obs: np.vstack(next_obs_batch),
            self.input_rnn_state: state_train,
            self.target_prediction: true_state_batch,
            self.train_length: trace_length,
            self.batch_size: batch_size})

