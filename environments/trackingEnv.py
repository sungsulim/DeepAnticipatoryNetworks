import numpy as np
import scipy.io as sio
import pickle as pk


class SSenvReal:

    def __init__(self, config, file_loc, idx_arr):

        self.rng = np.random.RandomState(config.random_seed)

        self.nStates = 21  # (these are the number of cells possible in the x space)
        # self.nActions = 10  # (0 to 9) # not used anymore.
        self.h_size = config.h_size

        self.max_ep_length = config.max_ep_length  # 16

        self.tracks = pk.load(open(file_loc, 'rb'))

        # sets idx to be chosen from. (For train/test split)
        self.track_idx = idx_arr

        # to add noise to observation (arbitrarily chosen)
        # Currently disabled
        # self.obsmat = [0.96, 0.93, 0.91, 0.88, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7]

        self.cc_by_sensors = self.get_covered_cells_by_sensors()
        self.current_track = None
        self.current_track_idx = None

    def start(self, selected_track_idx):

        # select track randomly
        if selected_track_idx is None:
            self.current_track = self.tracks[self.rng.choice(self.track_idx)]
        # For testing purpose
        else:
            self.current_track = self.tracks[selected_track_idx]

        self.current_track_idx = 0

        # print("current_track")
        # print(self.current_track)
        next_state = None

        # first observation: np.zeros (1,21)
        next_obs_one_hot = np.zeros(shape=(1, self.nStates))

        # next_state_cell: None
        # (next_obs, next_obs) : (2, 1, 21) -- for x and y
        return next_state, (next_obs_one_hot, next_obs_one_hot)

    def step(self, action):

        next_state_coord = self.process_coords(self.get_next_state(self.current_track, self.current_track_idx))  # tuple of (x,y) : 0.0 ~ 150.0, 160.0 if out of range

        # discretized into 20 bins : 0 ~ 19, idx 20 would only occur if value exactly 150.0 (idx 20 never really used)
        next_state = self.discretize_cell(next_state_coord)  # (2,1)

        self.current_track_idx += 1

        # next_obs currently returns the discretized cells
        next_obs = self.get_next_obs(next_state_coord, action)  # (2,1)

        # process into one-hot-vector of (1,31)
        # print('next_obs in env', next_obs)
        next_obs_one_hotX = self.format_one_hot(next_obs[0])
        next_obs_one_hotY = self.format_one_hot(next_obs[1])
        # print('onehot shape', np.shape(next_obs_one_hotX), np.shape(next_obs_one_hotY))

        done = False

        # next_state_cell: (2,1) : discretized cell
        # (next_obsX, next_obsY) : (2,1,21) : formatted observations one-hot
        return next_state, (next_obs_one_hotX, next_obs_one_hotY), done

    def format_one_hot(self, cell):
        cell_one_hot = np.reshape(np.array([int(i == cell) for i in range(self.nStates)]), (1, self.nStates))

        assert(np.shape(cell_one_hot) == (1, self.nStates))

        # (1, 21)
        return cell_one_hot

    def get_next_state(self, track, idx):

        if idx < len(track[0]):
            return [track[0][idx], track[1][idx]]
        else:
            # TODO: Temp. method of setting the discretized cell to be 20.
            return [100, 100]  # [20, 20] : manually selecting out-of-range xy_coord

    # previously get_obs
    def get_next_obs(self, xy_coord, action):

        # print(xy_coord, action)
        cc_by_s = self.cc_by_sensors[action]

        if xy_coord in cc_by_s:
            xy_cell = self.discretize_cell(xy_coord)
            return xy_cell
        else:
            return [20, 20]

    # previously get_cell()
    def process_coords(self, xypoint):

        # add 50 to make positive
        xp = xypoint[0] + 50  # 0~150, 160 if out of range
        yp = xypoint[1] + 50  # 0~150, 160 if out of range
        x_floored = xp // 1  # floor function
        y_floored = yp // 1  # floor function

        return [x_floored, y_floored]

    def discretize_cell(self, contstate):

        # discretize into 20 cells
        xcell = int((contstate[0]) * 2 / 15)
        ycell = int((contstate[1]) * 2 / 15)
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



    # def get_obsX(self, cellstate, action):
    #
    #     temp = self.get_obs(cellstate, action)
    #
    #     # TODO: disabled adding random noise
    #     # if np.random.rand(1) < self.obsmat[action]:
    #     #     return temp[0]
    #     # else:
    #     #     return 21
    #     return temp[0]
    #
    # def get_obsY(self, cellstate, action):
    #
    #     temp = self.get_obs(cellstate, action)
    #
    #     # TODO: disabled adding random noise
    #     # if np.random.rand(1) < self.obsmat[action]:
    #     #     return temp[1]
    #     # else:
    #     #     return 21
    #     return temp[1]
