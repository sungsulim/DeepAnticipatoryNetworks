import numpy as np

from config import Config
from experiment import Experiment
from environments.trackingEnv import SSenvReal
from agents.dan import DAN


def main():

    config = Config()
    rng_state = np.random.RandomState(config.random_seed)

    # create environment
    dummy_env = SSenvReal(config, [])

    # split train/test tracks
    track_idx = list(range(len(dummy_env.tracks))) # 0~16406
    rng_state.shuffle(track_idx)

    train_track_idx = track_idx[config.test_ep_num:]
    test_track_idx = track_idx[:config.test_ep_num]

    print("train track num: {}".format(len(train_track_idx)))  # 15407
    print("test track num: {}".format(len(test_track_idx)))  # 1000

    train_envX = SSenvReal(config, train_track_idx)
    train_envY = SSenvReal(config, train_track_idx)

    test_env = SSenvReal(config, test_track_idx)

    # create agent
    agentX = DAN(config, 'x')
    agentY = DAN(config, 'y')

    # create experiment
    experiment = Experiment(train_env={'x': train_envX, 'y': train_envY}, test_env=test_env, agent={'x': agentX, 'y': agentY}, config=config)

    # run experiment
    train_return_per_episode, test_mean_return_per_episode, test_std_return_per_episode = experiment.run()
    train_return_per_episodeX, train_return_per_episodeY = train_return_per_episode

    # save results


if __name__ == '__main__':
    main()