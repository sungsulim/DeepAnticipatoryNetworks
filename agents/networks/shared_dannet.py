import tensorflow as tf
import numpy as np


class SharedDANNetwork:
    def __init__(self, sess, config):

        self.sess = sess

        self.fc_size1 = config.fc_size1
        self.fc_size2 = config.fc_size2
        self.h_size = config.h_size

        self.qnet_lr = config.qnet_lr
        self.mnet_lr = config.mnet_lr

        self.gamma = config.gamma
        self.tau = config.tau
        self.nStates = config.nStates  # 21
        self.nActions = config.nActions  # 10

        # create network
        self.input_obs, self.input_rnn_state, self.current_rnn_state, \
            self.batch_size, self.train_length, self.salience, \
            self.Qout, self.argmaxQ, self.prediction = self.build_network(scope_name='shared_dannet')

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='shared_dannet')

        # create target network
        self.target_input_obs, self.target_input_rnn_state, self.target_current_rnn_state,  \
            self.target_batch_size, self.target_train_length, self.target_salience, \
            self.target_Qout, self.target_argmaxQ, _ = self.build_network(scope_name='target_shared_dannet')

        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_shared_dannet')

        # update target network Ops
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

        # init target network Ops
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in
                                       range(len(self.target_net_params))]

        ### Q Loss operator
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        action_onehot = tf.one_hot(self.action, self.nActions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, action_onehot), axis=1)
        td_error = tf.squared_difference(self.targetQ, self.Q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        maskA = tf.zeros([self.batch_size, self.train_length // 2])
        maskB = tf.ones([self.batch_size, self.train_length // 2])
        self.mask = tf.concat([maskA, maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.q_loss = tf.reduce_mean(td_error * self.mask)
        self.updateQ = tf.train.AdamOptimizer(learning_rate=self.qnet_lr).minimize(self.q_loss)

        ### M Loss operator
        self.target_prediction = tf.placeholder(shape=[None], dtype=tf.int32)
        target_prediction_onehot = tf.one_hot(self.target_prediction, self.nStates, dtype=tf.float32)

        self.m_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_prediction_onehot, logits=self.prediction))
        self.updateM = tf.train.AdamOptimizer(learning_rate=self.mnet_lr).minimize(self.m_loss)

        self.saver = tf.train.Saver()

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            input_obs = tf.placeholder(tf.float32, shape=(None, self.nStates + self.nActions))

            # 3 fc layers
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

            # Q part
            # The output from the recurrent player is then split into separate Value and Advantage streams
            streamA, streamV = tf.split(net, num_or_size_splits=2, axis=1)

            Advantage = tf.contrib.layers.fully_connected(streamA, self.nActions, activation_fn=None,
                                                          weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                              factor=1.0, mode="FAN_IN", uniform=True),
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                          biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                              factor=1.0, mode="FAN_IN", uniform=True))
            Value = tf.contrib.layers.fully_connected(streamV, 1, activation_fn=None,
                                                      weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          factor=1.0, mode="FAN_IN", uniform=True),
                                                      weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                      biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                          factor=1.0, mode="FAN_IN", uniform=True))

            # Salience is not used
            salience = tf.gradients(Advantage, input_obs)
            # Then combine them together to get our final Q-values.
            Qout = Value + tf.subtract(Advantage, tf.reduce_mean(Advantage, axis=1, keepdims=True))
            argmaxQ = tf.argmax(Qout, axis=1)
            # maxQ = tf.reduce_max(Qout, axis=1)


            # M part
            prediction = tf.contrib.layers.fully_connected(net, self.nStates, activation_fn=None,
                                                           weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                               factor=1.0, mode="FAN_IN", uniform=True),
                                                           weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                           biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                               factor=1.0, mode="FAN_IN", uniform=True))

        return input_obs, input_rnn_state, current_rnn_state, batch_size, train_length, salience, Qout, argmaxQ, prediction  # , maxQ

    def get_greedy_action(self, obs, input_rnn_state):

        ### during take action (e-greedy)
        # input_rnn_state : np.zeros only at the beginning of episode
        # train_length : 1
        # batch_size : 1

        ### during update
        # input_rnn_state : np.zeros
        # train_length: trace_length: 4
        # batch_size = 4

        train_length = 1
        batch_size = 1

        action, rnn_state = self.sess.run([self.argmaxQ, self.current_rnn_state], feed_dict={
            self.input_obs: obs,
            self.input_rnn_state: input_rnn_state,
            self.train_length: train_length,
            self.batch_size: batch_size
        })
        # print('Qval', Qval)

        return action, rnn_state

    def get_greedy_action_target(self, obs, input_rnn_state):

        ### during take action (e-greedy)
        # input_rnn_state : np.zeros only at the beginning of episode
        # train_length : 1
        # batch_size : 1

        ### during update
        # input_rnn_state : np.zeros
        # train_length: trace_length: 4
        # batch_size = 4

        train_length = 1
        batch_size = 1

        action, rnn_state = self.sess.run([self.target_argmaxQ, self.target_current_rnn_state], feed_dict={
            self.target_input_obs: obs,
            self.target_input_rnn_state: input_rnn_state,
            self.target_train_length: train_length,
            self.target_batch_size: batch_size
        })

        return action, rnn_state

    def get_Qval(self, obs, input_rnn_state):

        train_length = 1
        batch_size = 1

        Qval, rnn_state = self.sess.run([self.Qout, self.current_rnn_state], feed_dict={
            self.input_obs: obs,
            self.input_rnn_state: input_rnn_state,
            self.train_length: train_length,
            self.batch_size: batch_size
        })
        return Qval, rnn_state

    def update_q(self, train_batch, trace_length, batch_size):
        # trace_length : 4
        # batch_size : 4

        state_train = (np.zeros([batch_size, self.h_size]), np.zeros([batch_size, self.h_size]))

        obs_batch = train_batch[:, 0]
        action_batch = train_batch[:, 1]
        reward_batch = train_batch[:, 2]
        next_obs_batch = train_batch[:, 3]
        termination_batch = train_batch[:, 4]
        # true_state_batch = train_batch[:, 5]

        # get argmaxQ1
        argmaxQ1 = self.sess.run(self.argmaxQ, feed_dict={
            self.input_obs: np.vstack(next_obs_batch),
            self.input_rnn_state: state_train,
            self.train_length: trace_length,
            self.batch_size: batch_size
        })

        # get Q2
        Q2 = self.sess.run(self.target_Qout, feed_dict={
            self.target_input_obs: np.vstack(next_obs_batch),
            self.target_input_rnn_state: state_train,
            self.target_train_length: trace_length,
            self.target_batch_size: batch_size
        })

        # termination
        end_multiplier = -(termination_batch - 1)

        doubleQ = Q2[range(batch_size * trace_length), argmaxQ1]
        targetQ = reward_batch + (self.gamma * doubleQ * end_multiplier)

        # print('avg. doubleQ: {}'.format(np.mean(doubleQ)))

        # Update the network with our target values.
        self.sess.run(self.updateQ, feed_dict={
            self.input_obs: np.vstack(obs_batch),
            self.targetQ: targetQ,
            self.action: action_batch,
            self.train_length: trace_length,
            self.input_rnn_state: state_train,
            self.batch_size: batch_size
        })
        return

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

    def update_m(self, train_batch, trace_length, batch_size):
        # trace_length : 4
        # batch_size : 4

        state_train = (np.zeros([batch_size, self.h_size]), np.zeros([batch_size, self.h_size]))

        # obs_batch = train_batch[:, 0]
        # action_batch = train_batch[:, 1]
        # reward_batch = train_batch[:, 2]
        next_obs_batch = train_batch[:, 3]
        # termination_batch = train_batch[:, 4]
        true_state_batch = train_batch[:, 5]

        self.sess.run(self.updateM, feed_dict={
            self.input_obs: np.vstack(next_obs_batch),
            self.input_rnn_state: state_train,
            self.target_prediction: true_state_batch,
            self.train_length: trace_length,
            self.batch_size: batch_size})

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def save_network(self, save_dir, xory):
        self.saver.save(self.sess, '{}_shared_dannet{}'.format(save_dir, xory))

    def restore_network(self, load_dir, xory):
        self.saver.restore(self.sess, '{}_shared_dannet{}'.format(load_dir, xory))