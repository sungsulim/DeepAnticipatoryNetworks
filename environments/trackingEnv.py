import numpy as np
import scipy.io as sio
import pickle as pk


class SSenvReal:

    def __init__(self, config, idx_arr):

        self.nStates = 21  # (these are the number of cells possible in the x space)
        self.nActions = 10  # (0 to 9)
        self.h_size = config.h_size

        self.max_ep_length = config.max_ep_length  # 16

        self.tracks = pk.load(open('data/sampled_tracksX', 'rb'))

        # sets idx to be chosen from. (For train/test split)
        self.track_idx = idx_arr

        # to add noise to observation (arbitrarily chosen)
        self.obsmat = [0.96, 0.93, 0.91, 0.88, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7]

        self.k = 0
        self.index = 0
        self.track = self.getNewTrack()
        self.state = self.get_next_state(self.track, self.k)

        self.cc_by_sensors = self.get_covered_cells_by_sensors()

    def start(self):
        # observation shape: (1,31)
        obs = np.zeros(shape=(1, self.nStates + self.nActions))  # np.reshape([0] * (self.nStates + self.nActions), [1, 31])

        # Previous RNN state
        # state = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))

        return obs

    def step(self):
        return None

    def discretize_cell(self, contstate):

        xcell = int((contstate[0]) * 2 / 15)
        ycell = int((contstate[1]) * 2 / 15)
        discstate = [xcell, ycell]
        return discstate

    def get_next_state(self, track, index):

        index = index + 1
        if index < len(track[0]):
            return [track[0][index], track[1][index]]
        else:
            return [21, 21]

    def getNewTrack(self):

        return self.load_track()

    def load_track(self):

        if self.xory == 'x':
            sx = pk.load(open('/content/drive/My Drive/Dancode/sampled_tracksX', 'rb'))
        elif self.xory == 'y':
            sx = pk.load(open('/content/drive/My Drive/Dancode/sampled_tracksX', 'rb'))
        tracknu = np.random.randint(0, len(sx))
        trck = sx[tracknu]

        return trck

    def get_cell(self, xypoint):

        xp = xypoint[0] + 50
        yp = xypoint[1] + 50
        xcell = xp // 1
        ycell = yp // 1
        return [xcell, ycell]

    def get_covered_cells_by_sensors(self):

        mat_content = sio.loadmat('/content/drive/My Drive/Dancode/dandc.mat')
        cell_info = mat_content['c2']
        covered_cells_by_sensors = {}
        for i in range(10):
            covered_cells_by_sensors[i] = cell_info[0][i][0].tolist()

        for i in range(10):
            covered_cells_by_sensors[i].extend(cell_info[0][i + 10][0].tolist())

        return covered_cells_by_sensors

    def get_obs(self, cellstate, action):

        cc_by_s = self.cc_by_sensors[action]

        if cellstate in cc_by_s:
            temp = self.discretize_cell(cellstate)
            return temp
        else:
            return [21, 21]

    def get_obsX(self, cellstate, action):

        temp = self.get_obs(cellstate, action)
        if np.random.rand(1) < self.obsmat[action]:
            return temp[0]
        else:
            return 21

    def get_obsY(self, cellstate, action):
        temp = self.get_obs(cellstate, action)
        if np.random.rand(1) < self.obsmat[action]:
            return temp[1]
        else:
            return 21