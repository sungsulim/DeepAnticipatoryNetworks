import numpy as np
import argparse

from config import Config
from experiment import Experiment
from environments.trackingEnv import SSenvReal
from agents.dan import DAN


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_type', type=str)
    parser.add_argument('--random_seed', type=str)
    args = parser.parse_args()
    arg_params = {
        "agent_type": args.agent_type,
        "random_seed": int(args.random_seed)
    }

    config = Config()
    config.merge_config(arg_params)

    rng_state = np.random.RandomState(config.random_seed)

    # create environment
    dummy_envX = SSenvReal(config, 'data/sampled_tracksX', [])
    dummy_envY = SSenvReal(config, 'data/sampled_tracksY', [])

    # split train/test tracks
    trackX_idx = list(range(len(dummy_envX.tracks)))  # 0~16406
    trackY_idx = list(range(len(dummy_envY.tracks)))  # 0~12825

    rng_state.shuffle(trackX_idx)
    rng_state.shuffle(trackY_idx)

    train_trackX_idx = trackX_idx[config.test_ep_num:]
    test_trackX_idx = trackX_idx[:config.test_ep_num]

    train_trackY_idx = trackY_idx[config.test_ep_num:]
    test_trackY_idx = trackY_idx[:config.test_ep_num]

    print("train track X num: {}".format(len(train_trackX_idx)))  # 15907
    print("test track X num: {}".format(len(test_trackX_idx)))  # 500

    print("train track Y num: {}".format(len(train_trackY_idx)))  # 12326
    print("test track Y num: {}".format(len(test_trackY_idx)))  # 500

    train_envX = SSenvReal(config, 'data/sampled_tracksX', train_trackX_idx)
    train_envY = SSenvReal(config, 'data/sampled_tracksY', train_trackY_idx)

    test_envX = SSenvReal(config, 'data/sampled_tracksX', test_trackX_idx)
    test_envY = SSenvReal(config, 'data/sampled_tracksY', test_trackY_idx)

    # create agent
    agentX = DAN(config, 'x')
    agentY = DAN(config, 'y')

    # create experiment
    experiment = Experiment(train_env={'x': train_envX, 'y': train_envY},
                            test_env={'x': test_envX, 'y': test_envY},
                            agent={'x': agentX, 'y': agentY},
                            config=config)

    # run experiment
    train_return_per_episode, test_mean_return_per_episode, test_std_return_per_episode = experiment.run()
    train_return_per_episodeX, train_return_per_episodeY = train_return_per_episode

    # save results
    np.array(train_return_per_episodeX).tofile("results/{}_{}_train_return_per_episodeX.txt".format(config.agent_type, config.random_seed), sep=',', format='%15.8f')
    np.array(train_return_per_episodeY).tofile("results/{}_{}_train_return_per_episodeY.txt".format(config.agent_type, config.random_seed), sep=',', format='%15.8f')


if __name__ == '__main__':
    main()