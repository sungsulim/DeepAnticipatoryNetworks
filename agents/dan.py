import numpy as np
import tensorflow as tf
from utils.utils import ExperienceBuffer
from agents.networks.qnet import Qnetwork
from agents.networks.mnet import Mnetwork


class DAN:
    def __init__(self, config, xory):

        # 'x' or 'y'
        self.xory = xory

        self.rng = np.random.RandomState(config.random_seed)
        self.h_size = config.h_size  # The size of the final recurrent layer before splitting it into Advantage and Value streams.
        self.batch_size = config.batch_size
        self.trace_length = config.trace_length

        self.pre_train_steps = config.pre_train_steps
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.tau = config.tau

        self.replay_buffer = ExperienceBuffer(config.buffer_size, config.random_seed)
        self.rnn_stateQ = None
        self.rnn_stateM = None

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

    def reset(self):
        # TODO: Why is this a tuple?
        # reset rnn_state
        self.qnet.reset_rnn_state()
        self.mnet.reset_rnn_state()
        # self.rnn_stateQ = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))
        # self.rnn_stateM = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))

    def start(self, raw_obs, is_train):
        # obs: (1,31) np.zero observation
        obs = self.process_obs(raw_obs)

        # get Q-val
        self.qnet.get_qval(obs, self.)
        # find greedy action

        # if is_train, do e-greedy
        # else do greedy

        return

    def step(self, raw_obs, is_train):
        # obs: (1, 31)
        obs = self.process_obs(raw_obs)

        return

    def update(self):
        # add to replay buffer

        # perform update

        return

    def end(self, is_train):
        raise NotImplementedError("This shouldn't be used. The environment never terminates for SSEnv")

    def process_obs(self, obs):
        if self.xory == 'x':
            return obs[0]
        elif self.xory == 'y':
            return obs[1]
        else:
            return ValueError("Wrong value in self.xory")

