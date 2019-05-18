import numpy as np
from datetime import datetime
import time


class Experiment(object):
    def __init__(self, train_env, test_env, agent, config):

        # Env / Agent
        # self.test_rng = None
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent

        # self.config = config
        self.agent_type = config.agent_type

        self.total_train_steps = config.total_train_steps
        self.test_interval = config.test_interval
        self.test_ep_num = config.test_ep_num

        self.agent_pre_train_steps = config.agent_pre_train_steps
        self.agent_update_freq = config.agent_update_freq
        self.print_ep_freq = config.print_ep_freq

        # results
        self.train_return_per_episodeX = []
        self.train_return_per_episodeY = []

        self.test_mean_return_per_episode = []
        # self.test_mean_return_per_episodeY = []

        self.train_step_count = {'x': 0, 'y': 0}
        self.cum_train_time = 0.0
        self.cum_test_time = 0.0

        self.nStates = config.nStates
        self.nActions = config.nActions

    def run(self):

        episode_count = 0

        # For total time
        start_run = datetime.now()
        print("Start run at: " + str(start_run) + '\n')

        # TODO: Disabled evaluation
        # test once at beginning
        first_test_session_time, first_mean_return_per_episode = self.test()
        self.cum_test_time += first_test_session_time
        self.test_mean_return_per_episode.append(first_mean_return_per_episode)

        while (self.train_step_count['x'] + self.train_step_count['y'])/2 < self.total_train_steps:
            # runs a single episode and returns the accumulated return for that episode
            train_start_time = time.time()
            # print('running episode x')
            episode_returnX, num_stepsX, force_terminatedX = self.run_episode_train('x')

            # print('running episode y')
            episode_returnY, num_stepsY, force_terminatedY = self.run_episode_train('y')

            train_end_time = time.time()

            train_ep_time = train_end_time - train_start_time

            self.cum_train_time += train_ep_time
            # self.cum_test_time += test_session_time
            if episode_count % self.print_ep_freq == 0:
                print("Train:: ep: {}, returnX: {}, returnY: {}, stepsX: {}, stepsY: {},  elapsed: {}".format(
                    episode_count, episode_returnX, episode_returnY, num_stepsX, num_stepsY,
                    time.strftime("%H:%M:%S", time.gmtime(train_ep_time))))

            # Test
            if (self.train_step_count['x'] + self.train_step_count['y'])/2 % self.test_interval == 0:
                test_session_time, mean_return_per_episode = self.test()
                self.cum_test_time += test_session_time
                self.test_mean_return_per_episode.append(mean_return_per_episode)

            # force_terminated if total_train_steps reached and episode is truncated
            assert force_terminatedX == force_terminatedY
            if not force_terminatedX:
                self.train_return_per_episodeX.append(episode_returnX)
                self.train_return_per_episodeY.append(episode_returnY)

            episode_count += 1

        end_run = datetime.now()
        print("End run at: " + str(end_run) + '\n')
        print("Total Time taken: " + str(end_run - start_run) + '\n')
        print("Train Time: " + time.strftime("%H:%M:%S", time.gmtime(self.cum_train_time)))
        print("Test Time: " + time.strftime("%H:%M:%S", time.gmtime(self.cum_test_time)))

        return (self.train_return_per_episodeX, self.train_return_per_episodeY), self.test_mean_return_per_episode

    # Runs a single episode (TRAIN)
    def run_episode_train(self, xory):

        train_env = self.train_env[xory]
        agent = self.agent[xory]

        episode_return = 0.
        episode_step_count = 0

        _, obs = train_env.start(selected_track_idx=None)
        action_one_hot = np.zeros((1, agent.nActions))

        obs = np.array([np.concatenate((o, action_one_hot), axis=1) for o in obs])

        # print("env_start", np.shape(obs),obs)
        force_terminated = False

        episode_buffer = []

        # Episode is always fixed length
        for i in range(0, train_env.max_ep_length):

            if self.train_step_count[xory] == self.total_train_steps:
                force_terminated = True
                print("force terminated during training. This shouldn't normally happen")
                break

            self.train_step_count[xory] += 1
            episode_step_count += 1

            is_pretraining = (self.train_step_count[xory] <= self.agent_pre_train_steps)

            # Agent start/step
            # first step
            if i == 0:
                action = agent.start(obs, is_pretraining, is_train=True)
                # print("agent start", np.shape(action), action)

            # also take action in last step, because we are manually truncating the episode
            else:
                action = agent.step(obs, is_pretraining, is_train=True)
                # print("agent step", np.shape(action), action)

            # Env step: gives next_state, next_obs
            next_state, next_obs, done = train_env.step(action)

            # print("env step: ns", np.shape(next_state), next_state)
            # print("env step: n_obs", np.shape(next_obs), next_obs)

            # Augment next_obs
            action_one_hot = np.reshape(np.array([int(i == action) for i in range(agent.nActions)]), (1, agent.nActions))
            next_obs = np.array([np.concatenate((o, action_one_hot), axis=1) for o in next_obs])
            # print("env step: augmented n_obs", np.shape(next_obs), next_obs)

            # Agent predict
            reward = agent.predict(next_obs, next_state)
            # print("reward", np.shape(reward), reward)

            # Agent update
            if (not is_pretraining) and \
                    (self.train_step_count[xory] % self.agent_update_freq == 0):
                agent.update()

            episode_buffer.append(np.reshape(np.array([obs, action, reward, next_obs, False, next_state]), [1, 6]))

            episode_return += reward
            obs = next_obs

        # save to ReplayBuffer
        agent.replay_buffer.add(list(zip(np.array(episode_buffer))))

        return episode_return, episode_step_count, force_terminated

    def test(self):
        temp_return_per_episode = []

        test_session_time = 0.0

        for i in range(self.test_ep_num):
            test_start_time = time.time()
            episode_return, num_steps = self.run_episode_test(track_idx=i)
            test_end_time = time.time()

            temp_return_per_episode.append(episode_return)
            # temp_return_per_episodeY.append(episode_returnY)

            test_elapsed_time = test_end_time - test_start_time

            test_session_time += test_elapsed_time

            if i % self.print_ep_freq == 0:
                print("Test:: ep: {}, return: {}, steps: {}, elapsed: {}".format(i, episode_return, num_steps, time.strftime("%H:%M:%S",time.gmtime(test_elapsed_time))))

        mean_return_per_episode = np.mean(temp_return_per_episode)

        return test_session_time, mean_return_per_episode

    # Runs a single episode (TEST)
    def run_episode_test(self, track_idx):

        # if self.train_step_count['x'] > 24000:
        #     print("start episode test")
        # self.test_rng = np.random.RandomState(0)

        test_env = self.test_env

        agentX = self.agent['x']
        agentY = self.agent['y']

        episode_return = 0.
        episode_step_count = 0

        _, obs = test_env.start(selected_track_idx=track_idx)
        action_one_hot = np.zeros((1, self.nActions))

        obs = np.array([np.concatenate((o, action_one_hot), axis=1) for o in obs])

        for i in range(0, test_env.max_ep_length):
            episode_step_count += 1

            # Agent take action
            # first step
            if i == 0:
                Qx = agentX.start_getQ(obs, is_train=False)
                Qy = agentY.start_getQ(obs, is_train=False)

                # print('Qx', np.shape(Qx), Qx)
                # print('Qy', np.shape(Qy), Qy)

                if self.agent_type == 'dan' or self.agent_type == 'coverage' or self.agent_type == 'dan_coverage':
                    # TODO: implement true argmax (break ties)
                    action = np.argmax(Qx + Qy)
                elif self.agent_type == 'randomAction':
                    # action = self.test_rng.randint(0, agentX.nActions)
                    action = np.random.randint(0, agentX.nActions)
                else:
                    raise ValueError("invalid self.agent_type", self.agent_type)

            # also take action in last step, because we are manually truncating the episode
            else:
                Qx = agentX.step_getQ(obs, is_train=False)
                Qy = agentY.step_getQ(obs, is_train=False)

                # print('Qx', np.shape(Qx), Qx)
                # print('Qy', np.shape(Qy), Qy)

                if self.agent_type == 'dan' or self.agent_type == 'coverage' or self.agent_type == 'dan_coverage':
                    # TODO: implement true argmax (break ties)
                    action = np.argmax(Qx + Qy)
                elif self.agent_type == 'randomAction':
                    # action = self.test_rng.randint(0, agentX.nActions)
                    action = np.random.randint(0, agentX.nActions)
                else:
                    raise ValueError("invalid self.agent_type")

            # Env gives obs_n, reward
            next_state, next_obs, done = test_env.step(action)

            # Augment next_obs
            action_one_hot = np.reshape(np.array([int(i == action) for i in range(agentX.nActions)]),
                                        (1, agentX.nActions))
            next_obs = np.array([np.concatenate((o, action_one_hot), axis=1) for o in next_obs])

            # Agent predict
            _, rewardX = agentX.predict_test(next_obs, next_state)
            _, rewardY = agentY.predict_test(next_obs, next_state)

            if rewardX and rewardY:
                reward = 1.0
            else:
                reward = 0.0

            # if self.train_step_count['x'] > 24000:
            #     print("ns: {}".format(next_state))
            #     print("pred: {}, {}".format(predictionX, predictionY))
            #
            #     foundX = True
            #     foundY = True
            #
            #     if next_obs[0][0][self.nStates-1] == 1:
            #         foundX = False
            #     if next_obs[1][0][self.nStates-1] == 1:
            #         foundY = False
            #
            #     print("found: {}, {}".format(foundX, foundY))
            #     print('rewardX: {}'.format(rewardX))
            #     print('rewardY: {}'.format(rewardY))
            #     print('reward: {}'.format(reward))
            #     print()
                # input()
            episode_return += reward
            obs = next_obs

        return episode_return, episode_step_count

    def multiperson_test(self):

        test_env = self.test_env

        agentX = self.agent['x']
        agentY = self.agent['y']

        episode_return = np.zeros(test_env.max_ep_length)
        episode_step_count = 0

        # same for all tracks
        _, obs = test_env.multitest_start()
        action_one_hot = np.zeros((1, self.nActions))

        # (2, 1, nStates + nActions)
        obs = np.array([np.concatenate((o, action_one_hot), axis=1) for o in obs])

        stacked_obs = [obs for i in range(self.test_ep_num)]

        assert(len(stacked_obs) == self.test_ep_num)
        assert(np.shape(stacked_obs[0]) == (2, 1, self.nStates + self.nActions))

        # i = 0, ..., 11
        for i in range(0, test_env.max_ep_length):
            episode_step_count += 1
            print("current step: {}".format(episode_step_count))
            if self.agent_type == 'dan' or self.agent_type == 'coverage' or self.agent_type == 'dan_coverage':

                # For each person in test tracks, get Q val
                Qsum = np.zeros((1, self.nActions))

                for p in range(0, self.test_ep_num):

                    # Agent take action
                    # first step
                    if i == 0:
                        Qx = agentX.start_getQ(stacked_obs[p], is_train=False)  # (1, nActions)
                        Qy = agentY.start_getQ(stacked_obs[p], is_train=False)  # (1, nActions)

                    # also take action in last step, because we are manually truncating the episode
                    else:
                        Qx = agentX.step_getQ(stacked_obs[p], is_train=False)
                        Qy = agentY.step_getQ(stacked_obs[p], is_train=False)

                    Qsum += (Qx + Qy) / 2

                action = np.argmax(Qsum)

            elif self.agent_type == 'randomAction':
                # action = self.test_rng.randint(0, self.nActions)
                action = np.random.randint(0, self.nActions)
            else:
                raise ValueError("invalid self.agent_type", self.agent_type)

            action_one_hot = np.reshape(np.array([int(i == action) for i in range(agentX.nActions)]),
                                        (1, agentX.nActions))

            # Env gives obs_n, reward
            for p in range(0, self.test_ep_num):
                next_state, next_obs, done = test_env.multitest_step(selected_track_idx=p, step_num=i, action=action)

                # Augment next_obs
                next_obs = np.array([np.concatenate((o, action_one_hot), axis=1) for o in next_obs])

                # Agent predict
                _, rewardX = agentX.predict_test(next_obs, next_state)
                _, rewardY = agentY.predict_test(next_obs, next_state)

                if rewardX and rewardY:
                    reward = 1.0
                else:
                    reward = 0.0

                episode_return[i] += reward
                stacked_obs[p] = next_obs

        # episode_return: np array (max_ep_length, )
        # episode_step_count: int
        return episode_return, episode_step_count

