import numpy as np
import scipy.io as sio
import pickle as pk


class SSenvReal:

    def __init__(self, config, file_loc, idx_arr):

        # self.rng = np.random.RandomState(config.random_seed)

        self.nStates = config.nStates  # (these are the number of cells possible in the x space)
        self.h_size = config.h_size

        self.max_ep_length = config.max_ep_length

        self.tracks = pk.load(open(file_loc, 'rb'))

        # sets idx to be chosen from. (For train/test split)
        self.track_idx = idx_arr

        self.cc_by_sensors = self.get_covered_cells_by_sensors()
        self.current_track = None
        self.current_track_idx = None

    def start(self, selected_track_idx):

        # select track randomly
        if selected_track_idx is None:
            # self.current_track = self.tracks[self.rng.choice(self.track_idx)]
            self.current_track = self.tracks[np.random.choice(self.track_idx)]
        # For testing purpose
        else:
            self.current_track = self.tracks[self.track_idx[selected_track_idx]]

        self.current_track_idx = 0
        next_state = None

        # first observation: np.zeros (1,21)
        next_obs_one_hot = np.zeros(shape=(1, self.nStates))

        return next_state, (next_obs_one_hot, next_obs_one_hot)

    def step(self, action):

        next_state_coord = self.process_coords(self.get_next_state(self.current_track, self.current_track_idx))  # tuple of (x,y) : 0.0 ~ 150.0, 160.0 if out of range

        # discretized into 50 bins : 0 ~ 49, idx 50 would only occur if value exactly 150.0
        next_state = self.discretize_cell(next_state_coord)

        self.current_track_idx += 8

        # next_obs currently returns the discretized cells
        next_obs = self.get_next_obs(next_state_coord, action)  # (2,1)

        # process into one-hot-vector of (1,31)
        next_obs_one_hotX = self.format_one_hot(next_obs[0])
        next_obs_one_hotY = self.format_one_hot(next_obs[1])

        done = False

        # next_state_cell: (2,1) : discretized cell
        # (next_obsX, next_obsY) : (2,1,21) : formatted observations one-hot
        return next_state, (next_obs_one_hotX, next_obs_one_hotY), done

    def format_one_hot(self, cell):
        cell_one_hot = np.reshape(np.array([int(i == cell) for i in range(self.nStates)]), (1, self.nStates))
        assert(np.shape(cell_one_hot) == (1, self.nStates))

        return cell_one_hot

    def get_next_state(self, track, idx):

        if idx < len(track[0]):
            return [track[0][idx], track[1][idx]]
        else:
            # if not observed, setting the discretized cell to be last idx.
            return [100, 100]  # manually selecting out-of-range xy_coord

    # previously get_obs
    def get_next_obs(self, xy_coord, action):

        # print(xy_coord, action)
        cc_by_s = self.cc_by_sensors[action]

        if xy_coord in cc_by_s:
            xy_cell = self.discretize_cell(xy_coord)
            return xy_cell
        else:
            return [self.nStates - 1, self.nStates - 1]

    # convert x,y coordinates to discretized coordinates scaled to be positive from range 0~150
    def process_coords(self, xypoint):

        # add 50 to make positive
        xp = xypoint[0] + 50  # 0~150, 150 if out of range
        yp = xypoint[1] + 50  # 0~150, 150 if out of range
        x_floored = xp // 1  # floor function
        y_floored = yp // 1  # floor function

        return [x_floored, y_floored]

    # convert discretized x,y coordinates to cell idx
    def discretize_cell(self, contstate):

        # discretize into (nStates - 1) cells
        xcell = int((contstate[0]) * (self.nStates-1) / 150)
        ycell = int((contstate[1]) * (self.nStates-1) / 150)
        discstate = [xcell, ycell]
        return discstate

    def get_covered_cells_by_sensors(self):

        mat_content = sio.loadmat('data/dandc.mat')
        cell_info = mat_content['c2']
        covered_cells_by_sensors = {}
        for i in range(10):
            covered_cells_by_sensors[i] = cell_info[0][i][0].tolist()

        for i in range(10):
            covered_cells_by_sensors[i].extend(cell_info[0][i + 10][0].tolist())

        return covered_cells_by_sensors

    def multitest_start(self):

        next_state = None

        # first observation: np.zeros (1, nStates)
        next_obs_one_hot = np.zeros(shape=(1, self.nStates))

        # next_state_cell: None
        # (next_obs, next_obs) : (2, 1, nStates) -- for x and y
        return next_state, (next_obs_one_hot, next_obs_one_hot)

    def multitest_step(self, selected_track_idx, step_num, action):

        current_track = self.tracks[self.track_idx[selected_track_idx]]
        current_track_idx = step_num * 8

        next_state_coord = self.process_coords(self.get_next_state(current_track, current_track_idx))  # tuple of (x,y)

        # discretized into 50 bins
        next_state = self.discretize_cell(next_state_coord)

        # next_obs currently returns the discretized cells
        next_obs = self.get_next_obs(next_state_coord, action)

        # process into one-hot-vector
        next_obs_one_hotX = self.format_one_hot(next_obs[0])
        next_obs_one_hotY = self.format_one_hot(next_obs[1])

        done = False

        # next_state_cell: (2,1) : discretized cell
        # (next_obsX, next_obsY) : formatted observations one-hot
        return next_state, (next_obs_one_hotX, next_obs_one_hotY), done
