import numpy as np
import tensorflow as tf
from utils.utils import ExperienceBuffer
from agents.networks.qnet_dt import Qnetwork
from agents.networks.mnet_dt import Mnetwork


class DAN:
    def __init__(self, config):

        # self.rng = np.random.RandomState(config.random_seed)

        self.agent_type = config.agent_type  # 'dan', 'randomAction'

        self.batch_size = config.batch_size

        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.tau = config.tau

        self.nStates = config.nStates
        self.nActions = config.nActions

        self.replay_buffer = ExperienceBuffer(config.buffer_size, config.random_seed)

        self.graph = tf.Graph()

        # create Network
        with self.graph.as_default():
            # tf.set_random_seed(config.random_seed)
            self.sess = tf.Session()
            self.qnet = Qnetwork(self.sess, config)
            self.mnet = Mnetwork(self.sess, config)

            self.sess.run(tf.global_variables_initializer())
            self.qnet.init_target_network()

        self.avec = None

    def start(self, state, is_pretraining, is_train):
        # Check if starting or previously full
        assert (self.avec is None or np.array_equal(self.avec, np.ones(self.nActions)))
        # History of actions taken
        self.avec = np.array([0] * self.nActions)

        greedy_action = self.qnet.get_argmaxQ(state, self.avec)

        if is_train:

            if self.agent_type == 'dan':
                # if is_pretraining or self.rng.rand() < self.epsilon:
                if is_pretraining or np.random.rand() < self.epsilon:
                    # random action
                    # action = self.rng.randint(0, self.nActions)
                    # action = np.random.randint(0, self.nActions)
                    action = np.random.choice(np.where(self.avec == 0)[0])
                else:
                    action = greedy_action

            elif self.agent_type == 'randomAction':
                # action = self.rng.randint(0, self.nActions)
                # action = np.random.randint(0, self.nActions)
                action = np.random.choice(np.where(self.avec == 0)[0])
            else:
                raise ValueError("Invalid self.agent_type")

        else:
            action = greedy_action

        self.avec[action] = 1
        return action

    def step(self, state, is_pretraining, is_train):

        greedy_action = self.qnet.get_argmaxQ(state, self.avec)

        if is_train:

            if self.agent_type == 'dan':
                # if is_pretraining or self.rng.rand() < self.epsilon:
                if is_pretraining or np.random.rand() < self.epsilon:
                    # random action
                    # action = self.rng.randint(0, self.nActions)
                    # action = np.random.randint(0, self.nActions)
                    action = np.random.choice(np.where(self.avec == 0)[0])
                else:
                    # greedy action
                    action = greedy_action

            elif self.agent_type == 'randomAction':
                # action = self.rng.randint(0, self.nActions)
                # action = np.random.randint(0, self.nActions)
                action = np.random.choice(np.where(self.avec == 0)[0])
            else:
                raise ValueError("Invalid self.agent_type")

        else:
            action = greedy_action

        self.avec[action] = 1
        return action

    def predict(self, state, label):

        prediction = self.mnet.get_prediction(state)
        reward = self.get_prediction_reward(prediction[0], label)

        return reward

    def update(self):

        # Get a random batch of experiences.
        state_batch, action_batch, reward_batch, next_state_batch, termination_batch, label_batch = self.replay_buffer.sample(self.batch_size)
        train_batch = [state_batch, action_batch, reward_batch, next_state_batch, termination_batch, label_batch]
        # perform update

        # QNet update
        if self.agent_type == 'dan':
            self.qnet.update(train_batch, self.batch_size)
            self.qnet.update_target_network()

        # MNet update
        if self.agent_type == 'dan' or self.agent_type == 'randomAction':
            self.mnet.update(train_batch)
        return

    def get_prediction_reward(self, pred_s, true_s):
        # print("pred_s: {}, true_s: {}".format(pred_s, true_s))
        if pred_s == true_s:
            reward = 1.0
        else:
            reward = 0.0
        return reward

    # def start_getQ(self, raw_obs, rnn_state, is_train):
    #     assert(is_train is False)
    #     # obs: (1,31) np.zero observation
    #     obs = self.select_xy(raw_obs)
    #
    #     # reset qnet, mnet current rnn state
    #     # self.test_qnet_current_rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))
    #     # self.test_mnet_current_rnn_state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))
    #
    #     Qval, new_rnn_state = self.qnet.get_Qval(obs, rnn_state)
    #     # self.test_qnet_current_rnn_state = rnn_state
    #
    #     return Qval, new_rnn_state
    #
    # def step_getQ(self, raw_obs, rnn_state, is_train):
    #     assert (is_train is False)
    #     # obs: (1, 31)
    #     obs = self.select_xy(raw_obs)
    #
    #     Qval, new_rnn_state = self.qnet.get_Qval(obs, rnn_state)
    #     # self.test_qnet_current_rnn_state = rnn_state
    #
    #     return Qval, new_rnn_state
    #
    # def predict_test(self, raw_obs, raw_state, rnn_state):
    #     # print("raw_obs", raw_obs)
    #     obs = self.select_xy(raw_obs)
    #     state = self.select_xy(raw_state)
    #
    #     # prediction, rnn_state = self.mnet.get_prediction(obs, self.test_mnet_current_rnn_state)
    #     prediction, new_rnn_state = self.mnet.get_prediction(obs, rnn_state)
    #     # self.test_mnet_current_rnn_state = rnn_state
    #
    #     if self.agent_type == 'dan' or self.agent_type == 'randomAction' or self.agent_type == 'dan_coverage':
    #         reward = self.get_prediction_reward(prediction[0], state)
    #
    #     elif self.agent_type == 'coverage':
    #         reward = self.get_coverage_reward(obs)
    #
    #     else:
    #         raise ValueError("Invalid self.agent_type")
    #
    #     return np.argmax(prediction[0]), reward, new_rnn_state

    def save_network(self, save_dir):
        self.qnet.save_network(save_dir)
        self.mnet.save_network(save_dir)

    def restore_network(self, load_dir):
        self.qnet.restore_network(load_dir)
        self.mnet.restore_network(load_dir)
