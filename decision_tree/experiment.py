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
        self.train_return_per_episode = []
        self.test_mean_return_per_episode = []

        self.cum_train_time = 0.0
        self.cum_test_time = 0.0
        self.train_step_count = 0

        self.nStates = config.nStates
        self.nActions = config.nActions

    def run(self):

        self.test_count = 0
        self.train_step_count = 0
        episode_count = 0

        # For total time
        start_run = datetime.now()
        print("Start run at: " + str(start_run) + '\n')

        # test once at beginning
        first_test_session_time, first_mean_return_per_episode = self.test()
        self.cum_test_time += first_test_session_time
        self.test_mean_return_per_episode.append(first_mean_return_per_episode)

        while self.train_step_count < self.total_train_steps:
            # runs a single episode and returns the accumulated return for that episode
            train_start_time = time.time()

            episode_return, num_steps, force_terminated = self.run_episode_train()

            train_end_time = time.time()
            train_ep_time = train_end_time - train_start_time

            self.cum_train_time += train_ep_time
            # self.cum_test_time += test_session_time

            # if episode_count % self.print_ep_freq == 0:
            #     print("Train:: ep: {}, return: {}, steps: {}, elapsed: {}".format(episode_count, episode_return, num_steps, time.strftime("%H:%M:%S", time.gmtime(train_ep_time))))

            # Test
            if self.train_step_count % self.test_interval == 0:
                test_session_time, mean_return_per_episode = self.test()
                self.cum_test_time += test_session_time
                self.test_mean_return_per_episode.append(mean_return_per_episode)

            if not force_terminated:
                self.train_return_per_episode.append(episode_return)

            episode_count += 1

        end_run = datetime.now()
        print("End run at: " + str(end_run) + '\n')
        print("Total Time taken: " + str(end_run - start_run) + '\n')
        print("Train Time: " + time.strftime("%H:%M:%S", time.gmtime(self.cum_train_time)))
        print("Test Time: " + time.strftime("%H:%M:%S", time.gmtime(self.cum_test_time)))

        return self.train_return_per_episode, self.test_mean_return_per_episode

    # Runs a single episode (TRAIN)
    def run_episode_train(self):

        train_env = self.train_env
        agent = self.agent

        episode_return = 0.
        episode_step_count = 0

        # empty feature vector
        state = train_env.start(selected_idx=None)

        force_terminated = False

        # Episode is always fixed length
        for i in range(0, train_env.max_ep_length):

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
                action = agent.start(state, is_pretraining, is_train=True)
                # print("agent start", np.shape(action), action)

            # also take action in last step, because we are manually truncating the episode
            else:
                action = agent.step(state, is_pretraining, is_train=True)
                # print("agent step", np.shape(action), action)

            # Env step: gives next_state, next_obs
            next_state, done = train_env.step(action)

            # print("env step: ns", np.shape(next_state), next_state)
            # print("env step: n_obs", np.shape(next_obs), next_obs)

            # Agent predict
            reward = agent.predict(next_state, train_env.current_label)
            # print("reward", np.shape(reward), reward)

            # save to ReplayBuffer
            agent.replay_buffer.add((state, action, reward, next_state, False, train_env.current_label))

            # Agent update
            if (not is_pretraining) and \
                    (self.train_step_count % self.agent_update_freq == 0):
                agent.update()

            episode_return += reward
            state = next_state

        return episode_return, episode_step_count, force_terminated

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

            # if i % self.print_ep_freq == 0:
            #     print("Test:: ep: {}, return: {}, steps: {}, elapsed: {}".format(i, episode_return, num_steps, time.strftime("%H:%M:%S",time.gmtime(test_elapsed_time))))

        mean_return_per_episode = np.mean(temp_return_per_episode)

        print("Test iter: {}, train_steps: {}, test_mean_return: {}, elapsed: {}".format(self.test_count, self.train_step_count, mean_return_per_episode,
                                                                        time.strftime("%H:%M:%S", time.gmtime(
                                                                            test_session_time))))
        self.test_count += 1

        return test_session_time, mean_return_per_episode

    # Runs a single episode (TEST)
    def run_episode_test(self, track_idx):

        # if self.train_step_count['x'] > 24000:
        #     print("start episode test")
        # self.test_rng = np.random.RandomState(0)

        test_env = self.test_env
        agent = self.agent

        episode_return = 0.
        episode_step_count = 0

        state = test_env.start(selected_idx=track_idx)

        for i in range(0, test_env.max_ep_length):
            episode_step_count += 1

            # Agent take action
            # first step
            if i == 0:
                action = agent.start(state, is_pretraining=False, is_train=False)

            # also take action in last step, because we are manually truncating the episode
            else:
                action = agent.step(state, is_pretraining=False, is_train=False)

            # Env gives obs_n, reward
            next_state, done = test_env.step(action)

            # Agent predict
            reward = agent.predict(next_state, test_env.current_label)

            episode_return += reward
            state = next_state

        return episode_return, episode_step_count