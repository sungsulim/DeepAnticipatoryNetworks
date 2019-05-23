import numpy as np
import scipy.io as sio
import pickle as pk


class DecisionTreeEnv:

    def __init__(self, data, label, config):

        # self.rng = np.random.RandomState(config.random_seed)
        self.nStates = config.nStates
        self.nActions = config.nActions
        self.max_ep_length = config.max_ep_length

        self.data = data
        self.label = label

        self.num_samples = len(self.data)
        self.current_data = None
        self.revealed_data = None
        self.current_label = None
        self.current_step = None

    def start(self, selected_idx):

        # select track randomly
        if selected_idx is None:
            # self.current_track = self.tracks[self.rng.choice(self.track_idx)]
            ind = np.random.randint(self.num_samples)
            self.current_data = self.data[ind]
            self.current_label = self.label[ind]
            self.current_step = 0

        # For testing purpose
        else:
            self.current_data = self.data[selected_idx]
            self.current_label = self.label[selected_idx]
            self.current_step = 0

        self.revealed_data = [-1] * self.nActions
        # return empty feature vector : All -1
        return self.revealed_data

    def step(self, action):
        self.current_step += 1

        if self.current_step == self.max_ep_length:
            done = True
        else:
            done = False

        # print("=======")
        # print("agent_step")
        # print("self.revealed_data: {}".format(self.revealed_data))
        # print("action: {}".format(action))

        self.revealed_data[action] = self.current_data[action]

        # print("update revealed_data: {}".format(self.revealed_data))
        # print("done: {}".format(done))
        # print("=======")

        return self.revealed_data, done

    def get_label(self):
        return self.current_label

