import numpy as np
import tensorflow as tf
from utils.utils import ExperienceBuffer
from agents.networks.qnet import Qnetwork
from agents.networks.mnet import Mnetwork


class DAN:
    def __init__(self, config):

        self.rng = np.random.RandomState(config.random_seed)

        self.h_size = config.h_size  # The size of the final recurrent layer before splitting it into Advantage and Value streams.
        self.batch_size = config.batch_size
        self.trace_length = config.trace_length

        self.pre_train_steps = config.pre_train_steps
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.tau = config.tau

        self.rnn_state = None
        self.graph = tf.Graph()

        # create Network
        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.qnet = Qnetwork(self.sess, config)
            self.mnet = Mnetwork(self.sess, config)

            self.sess.run(tf.global_variables_initializer())
            self.qnet.init_target_network()
            self.mnet.init_target_network()

    def start(self, obs, is_train):
        # obs: (1,31) np.zero observation

        self.rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))

        return

    def step(self, is_train):
        return

    def update(self):
        # add to replay buffer

        # perform update

        return

    def end(self, is_train):
        raise NotImplementedError("This shouldn't be used. The environment never terminates for SSEnv")


