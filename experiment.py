import numpy as np
from datetime import datetime
import time


class Experiment(object):
    def __init__(self, train_env, test_env, agent, config):

        # Env / Agent
        self.train_env = train_env
        self.test_env = test_env
        self.agent = agent

        # self.config = config
        self.total_train_steps = config.total_train_steps
        self.test_interval = config.test_interval
        self.test_ep_num = config.test_ep_num

        self.agent_pre_train_steps = config.agent_pre_train_steps
        self.agent_update_freq = config.agent_update_freq

        # results
        self.train_return_per_episodeX = []
        self.train_return_per_episodeY = []
        self.test_mean_return_per_episode = []
        self.test_std_return_per_episode = []

        self.train_step_count = 0
        self.cum_train_time = 0.0
        self.cum_test_time = 0.0

    def run(self):

        episode_count = 0

        # For total time
        start_run = datetime.now()
        print("Start run at: " + str(start_run) + '\n')

        # TODO: Disabled evaluation
        # test once at beginning
        # self.cum_test_time += self.test()

        while self.train_step_count < self.total_train_steps:

            # runs a single episode and returns the accumulated return for that episode
            train_start_time = time.time()
            episode_returnX, num_stepsX, force_terminatedX, test_session_timeX = self.run_episode_train('x')
            episode_returnY, num_stepsY, force_terminatedY, test_session_timeY = self.run_episode_train('y')

            train_end_time = time.time()

            # TODO: combine test_session_timeX and Y
            test_session_time = test_session_timeX + test_session_timeY
            train_ep_time = train_end_time - train_start_time - test_session_time

            self.cum_train_time += train_ep_time
            self.cum_test_time += test_session_time
            # print("Train:: ep: " + str(episode_count) + ", returnX: " + str(episode_returnX) + ", n_steps: " + str(num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(train_ep_time)))
            print("Train:: ep: {}, returnX: {}, returnY: {}, stepsX: {}, stepsY: {},  elapsed: {}".format(episode_count, episode_returnX, episode_returnY, num_stepsX, num_stepsY, time.strftime("%H:%M:%S", time.gmtime(train_ep_time))))

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

        return (self.train_return_per_episodeX, self.train_return_per_episodeY), self.test_mean_return_per_episode, self.test_std_return_per_episode

    # Runs a single episode (TRAIN)
    def run_episode_train(self, xory):

        train_env = self.train_env[xory]
        agent = self.agent[xory]

        test_session_time = 0.0
        episode_return = 0.
        episode_step_count = 0

        # TODO: reset rnn state. No, it should be reset at agent.start() and when starting training
        # agent.reset()  # Need to be careful in Agent not to reset the weight

        # TODO: Use track_idx to choose idx (only for test)
        _, obs = train_env.start(selected_track_idx=None)

        force_terminated = False

        episode_buffer = []

        # Episode is always fixed length
        for i in range(0, train_env[xory].max_ep_length):

            if self.train_step_count == self.total_train_steps:
                force_terminated = True
                print("force terminated during training. This shouldn't normally happen")
                break

            self.train_step_count += 1
            episode_step_count += 1

            is_pretraining = (self.train_step_count <= self.agent_pre_train_steps)

            # Agent start/step
            # first step
            if i == 0:
                action = agent.start(obs, is_pretraining, is_train=True)

            # also take action in last step, because we are manually truncating the episode
            else:
                action = agent.step(obs, is_pretraining, is_train=True)

            # Env step: gives next_state, next_obs
            next_state, next_obs, done = train_env.step(action)

            # Agent predict
            reward = agent.predict(next_obs, next_state)

            # Agent update
            if (not is_pretraining) and \
                    (self.train_step_count % self.agent_update_freq == 0):
                agent.update()

            episode_buffer.append(np.reshape(np.array([obs, action, reward, next_obs, False, next_state]), [1, 6]))

            episode_return += reward
            obs = next_obs

            # Temporarily disabled
            # if self.train_step_count % self.test_interval == 0:
            #     test_session_time += self.test()

        # save to ReplayBuffer
        agent.replay_buffer.add(list(zip(np.array(episode_buffer))))

        return episode_return, episode_step_count, force_terminated, test_session_time

    def test(self):
        temp_return_per_episode = []

        test_session_time = 0.0

        for i in range(self.test_ep_num):
            test_start_time = time.time()
            episode_return, num_steps = self.run_episode_test(track_idx=i)
            test_end_time = time.time()
            temp_return_per_episode.append(episode_return)

            test_elapsed_time = test_end_time - test_start_time

            test_session_time += test_elapsed_time
            print("=== TEST :: ep: " + str(i) + ", r: " + str(episode_return) + ", n_steps: " + str(
                num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(test_elapsed_time)))

        self.test_mean_return_per_episode.append(np.mean(temp_return_per_episode))
        self.test_std_return_per_episode.append(np.std(temp_return_per_episode))

        self.cum_test_time += test_session_time

        return test_session_time

    # Runs a single episode (TEST)
    def run_episode_test(self, track_idx):

        episode_return = 0.
        episode_step_count = 0

        self.agent.reset()
        obs = self.test_env.start(track_idx=track_idx)

        for i in range(0, self.test_env.max_ep_length):
            episode_step_count += 1

            # Agent take action
            # first step
            if i == 0:
                action = self.agent.start(obs, is_train=False)

            # also take action in last step, because we are manually truncating the episode
            else:
                action = self.agent.step(obs, is_train=False)

            # Env gives obs_n, reward
            obs_n, reward, done, info = self.test_env.step(action)

            episode_return += reward
            obs = obs_n

        return episode_return, episode_step_count




