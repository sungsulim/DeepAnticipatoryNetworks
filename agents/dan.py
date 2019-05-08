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

        # self.pre_train_steps = config.pre_train_steps
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.tau = config.tau
        self.nActions = config.nActions
        self.nStates = config.nStates

        self.replay_buffer = ExperienceBuffer(config.buffer_size, config.random_seed)

        self.graph = tf.Graph()

        self.qnet_current_rnn_state = None
        self.mnet_current_rnn_state = None
        # create Network
        with self.graph.as_default():
            tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.qnet = Qnetwork(self.sess, config)
            self.mnet = Mnetwork(self.sess, config)

            self.sess.run(tf.global_variables_initializer())
            self.qnet.init_target_network()

    def start(self, raw_obs, is_pretraining, is_train):
        # obs: (1,31) np.zero observation
        obs = self.select_xy(raw_obs)

        # reset qnet, mnet current rnn state
        self.qnet_current_rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))
        self.mnet_current_rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))

        greedy_action, rnn_state = self.qnet.get_greedy_action(obs, self.qnet_current_rnn_state)
        self.qnet_current_rnn_state = rnn_state

        if is_train:
            if is_pretraining or self.rng.rand() < self.epsilon:
                # random action
                action = self.rng.randint(0, self.nActions)
            else:
                action = greedy_action[0]

        return action

    def step(self, raw_obs, is_pretraining, is_train):
        # obs: (1, 31)
        obs = self.select_xy(raw_obs)

        greedy_action, rnn_state = self.qnet.get_greedy_action(obs, self.qnet_current_rnn_state)
        self.qnet_current_rnn_state = rnn_state

        if is_train:
            if is_pretraining or self.rng.rand() < self.epsilon:
                # random action
                action = self.rng.randint(0, self.nActions)
            else:
                # greedy action
                action = greedy_action[0]

        return action

    def predict(self, raw_obs, raw_state):
        obs = self.select_xy(raw_obs)
        state = self.select_xy(raw_state)

        prediction, rnn_state = self.mnet.get_prediction(obs, self.mnet_current_rnn_state)
        self.mnet_current_rnn_state = rnn_state

        reward = self.get_prediction_reward(prediction[0], state)

        return reward

    def get_prediction_reward(self, pred_s, true_s):
        # true_s : 0~21
        # pred_s : an array of size (21,) containing prediction values with highest being most probable
        if np.argmax(pred_s) == true_s:
            reward = 1.0
        else:
            reward = 0.0

        return reward

    def update(self):

        # Get a random batch of experiences.
        train_batch = self.replay_buffer.sample(self.batch_size, self.trace_length)

        # Select x or y obs/next_obs, true_state
        for i in range(len(train_batch)):
            # obs
            train_batch[i][0] = self.select_xy(train_batch[i][0])
            # next_obs
            train_batch[i][3] = self.select_xy(train_batch[i][3])
            train_batch[i][5] = self.select_xy(train_batch[i][5])

        # perform update
        self.qnet.update(train_batch, self.trace_length, self.batch_size)
        self.mnet.update(train_batch, self.trace_length, self.batch_size)

        return

    def select_xy(self, xy_tuple):
        if self.xory == 'x':
            return xy_tuple[0]
        elif self.xory == 'y':
            return xy_tuple[1]
        else:
            return ValueError("Wrong value in self.xory")

