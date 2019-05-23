import tensorflow as tf
import numpy as np


class Qnetwork:
    def __init__(self, sess, config):

        self.sess = sess

        self.fc_size1 = config.fc_size1
        self.learning_rate = config.qnet_lr

        self.gamma = config.gamma
        self.tau = config.tau

        self.nStates = config.nStates
        self.nActions = config.nActions

        # create network
        self.input_state, self.Qout, self.argmaxQ, self.maxQ = self.build_network(scope_name='qnet')

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='qnet')

        # create target network
        self.target_input_state, self.target_Qout, self.target_argmaxQ, self.target_maxQ = self.build_network(scope_name='target_qnet')

        self.target_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_qnet')

        # update target network Ops
        self.update_target_net_params = [
            tf.assign_add(self.target_net_params[idx], self.tau * (self.net_params[idx] - self.target_net_params[idx]))
            for idx in range(len(self.target_net_params))]

        # init target network Ops
        self.init_target_net_params = [tf.assign(self.target_net_params[idx], self.net_params[idx]) for idx in
                                       range(len(self.target_net_params))]

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        action_onehot = tf.one_hot(self.action, self.nActions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, action_onehot), axis=1)
        td_error = tf.squared_difference(self.targetQ, self.Q)

        self.loss = tf.reduce_mean(td_error)
        self.updateModel = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def build_network(self, scope_name):
        with tf.variable_scope(scope_name):

            input_state = tf.placeholder(tf.float32, shape=(None, self.nActions))

            # 3 fc layers
            net = tf.contrib.layers.fully_connected(input_state, self.fc_size1, activation_fn=tf.nn.relu,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))

            Qout = tf.contrib.layers.fully_connected(net, self.nActions, activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True),
                                                    weights_regularizer=tf.contrib.layers.l2_regularizer(0.01),
                                                    biases_initializer=tf.contrib.layers.variance_scaling_initializer(
                                                        factor=1.0, mode="FAN_IN", uniform=True))

            argmaxQ = tf.argmax(Qout, axis=1)
            maxQ = tf.reduce_max(Qout, axis=1)

        return input_state, Qout, argmaxQ, maxQ

    def get_argmaxQ(self, state, avec):

        Qval = self.sess.run(self.Qout, feed_dict={
            self.input_state: [state]
        })

        Qval[0][np.where(avec == 1)[0]] = -np.inf

        action = np.argmax(Qval)
        return action

    def get_argmaxQ_target(self, state):

        action = self.sess.run(self.target_argmaxQ, feed_dict={
            self.target_input_state: state
        })
        return action

    def get_Qval(self, state):

        Qval = self.sess.run(self.Qout, feed_dict={
            self.input_state: state
        })
        return Qval

    def get_maxQ_target(self, state):

        maxQ_target = self.sess.run(self.target_maxQ, feed_dict={
            self.target_input_state: state
        })

        return maxQ_target

    def update(self, train_batch, batch_size):

        # state_batch, action_batch, reward_batch, next_state_batch, termination_batch, label_batch = train_batch
        state_batch = train_batch[0]
        action_batch = train_batch[1]
        reward_batch = train_batch[2]
        next_state_batch = train_batch[3]
        termination_batch = train_batch[4]
        # label_batch = train_batch[5]

        # get argmaxQ1
        argmaxQ1 = self.sess.run(self.argmaxQ, feed_dict={
            self.input_state: next_state_batch
        })

        # get Q2
        Q2 = self.sess.run(self.target_Qout, feed_dict={
            self.target_input_state: next_state_batch
        })

        # termination
        end_multiplier = -(termination_batch - 1)

        doubleQ = Q2[range(batch_size), argmaxQ1]
        targetQ = reward_batch + (self.gamma * doubleQ * end_multiplier)

        # print('avg. doubleQ: {}'.format(np.mean(doubleQ)))

        # Update the network with our target values.
        self.sess.run(self.updateModel, feed_dict={
            self.input_state: state_batch,
            self.targetQ: targetQ,
            self.action: action_batch
        })

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def save_network(self, save_dir):
        self.saver.save(self.sess, '{}_qnet_dt'.format(save_dir))

    def restore_network(self, load_dir):
        self.saver.restore(self.sess, '{}_qnet_dt'.format(load_dir))