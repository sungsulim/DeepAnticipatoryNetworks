import numpy as np
import argparse
import os
import json
from collections import OrderedDict

from config import Config
from experiment import Experiment
from environments.trackingEnv import SSenvReal
from agents.dan_tracking import DAN
from agents.dan_shared_tracking import DANShared


def get_sweep_parameters(parameters, index):
    out = OrderedDict()
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        out[key] = parameters[key][int(index / accum) % num]
        accum *= num
    return (out, accum)


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--agent_type', type=str)
    # parser.add_argument('--random_seed', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--agent_json', type=str)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()

    # arg_params = {
    #     "agent_type": args.agent_type,
    #     "random_seed": int(args.random_seed)
    # }

    with open(args.agent_json, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    agent_params, total_num_sweeps = get_sweep_parameters(agent_json, args.index)

    print('Agent setting: ', agent_params)

    # get run idx and setting idx
    RUN_NUM = int(args.index / total_num_sweeps)
    SETTING_NUM = args.index % total_num_sweeps

    config = Config()
    # config.merge_config(arg_params)
    config.merge_config(agent_params)

    # save results
    save_dir = 'results/{}'.format(args.result_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Fixed random state for train/test split
    rng_state = np.random.RandomState(9999)

    ####### New data
    dummy_env = SSenvReal(config, 'data/sampled_tracks_new', [])

    track_idx = list(range(len(dummy_env.tracks)))
    rng_state.shuffle(track_idx)

    train_track_idx = track_idx[int(config.test_ep_num):]
    test_track_idx = track_idx[:int(config.test_ep_num)]

    print("train track num: {}".format(len(train_track_idx)))  # 15907
    print("test track X num: {}".format(len(test_track_idx)))  # 50

    train_envX = SSenvReal(config, 'data/sampled_tracks_new', train_track_idx)
    train_envY = SSenvReal(config, 'data/sampled_tracks_new', train_track_idx)

    test_env = SSenvReal(config, 'data/sampled_tracks_new', test_track_idx)
    # test_envY = SSenvReal(config, 'data/sampled_tracks_new', test_track_idx)

    ####### Previous data

    # # create environment
    # dummy_envX = SSenvReal(config, 'data/sampled_tracksX', [])
    # dummy_envY = SSenvReal(config, 'data/sampled_tracksY', [])
    #
    # # Remove short tracks
    # # X < 32: 4736
    # # Y < 32: 3311
    # trackX_idx = []
    # trackY_idx = []
    #
    # cutoff = config.max_ep_length
    # for i in range(len(dummy_envX.tracks)):
    #     if len(dummy_envX.tracks[i][0]) >= cutoff:
    #         trackX_idx.append(i)
    #
    # for j in range(len(dummy_envY.tracks)):
    #     if len(dummy_envY.tracks[j][1]) >= cutoff:
    #         trackY_idx.append(j)
    #
    # # split train/test tracks
    # # trackX_idx = list(range(len(dummy_envX.tracks)))  # 0~16406
    # # trackY_idx = list(range(len(dummy_envY.tracks)))  # 0~12825
    # rng_state.shuffle(trackX_idx)
    # rng_state.shuffle(trackY_idx)
    #
    # train_trackX_idx = trackX_idx[int(config.test_ep_num):]
    # test_trackX_idx = trackX_idx[:int(config.test_ep_num)]
    #
    # train_trackY_idx = trackY_idx[int(config.test_ep_num):]
    # test_trackY_idx = trackY_idx[:int(config.test_ep_num)]
    #
    # print("train track X num: {}".format(len(train_trackX_idx)))  # 15907
    # print("train track Y num: {}".format(len(train_trackY_idx)))  # 12326
    #
    # print("test track X num: {}".format(len(test_trackX_idx)))  # 500
    # print("test track Y num: {}".format(len(test_trackY_idx)))  # 500
    #
    # train_envX = SSenvReal(config, 'data/sampled_tracksX', train_trackX_idx)
    # train_envY = SSenvReal(config, 'data/sampled_tracksY', train_trackY_idx)
    #
    # test_envX = SSenvReal(config, 'data/sampled_tracksX', test_trackX_idx)
    # test_envY = SSenvReal(config, 'data/sampled_tracksY', test_trackY_idx)
    ##############


    # create agent


    if config.agent_type == 'dan_shared' or config.agent_type == 'shared_attention':
        agentX = DANShared(config, 'x')
        agentY = DANShared(config, 'y')

    else:
        assert(config.agent_type == 'dan' or config.agent_type == 'coverage' or config.agent_type == 'dan_coverage' or config.agent_type == 'randomAction')
        agentX = DAN(config, 'x')
        agentY = DAN(config, 'y')

    # create experiment
    experiment = Experiment(train_env={'x': train_envX, 'y': train_envY},
                            test_env=test_env,
                            agent={'x': agentX, 'y': agentY},
                            config=config)

    # run experiment
    train_return_per_episode, test_mean_return_per_episode = experiment.run()
    train_return_per_episodeX, train_return_per_episodeY = train_return_per_episode

    # save model
    save_prefix = '{}/{}_setting_{}_run_{}'.format(save_dir, config.agent_type, SETTING_NUM, RUN_NUM)
    agentX.save_network(save_prefix, 'x')
    agentY.save_network(save_prefix, 'y')

    # Train result
    np.array(train_return_per_episodeX).tofile("{}_train_return_per_episodeX.txt".format(save_prefix), sep=',', format='%15.8f')
    np.array(train_return_per_episodeY).tofile("{}_train_return_per_episodeY.txt".format(save_prefix), sep=',', format='%15.8f')

    # Test result
    np.array(test_mean_return_per_episode).tofile("{}_test_mean_return_per_episode.txt".format(save_prefix), sep=',', format='%15.8f')
    # np.array(test_mean_return_per_episodeY).tofile("{}/{}_setting_{}_run_{}_test_mean_return_per_episodeY.txt".format(save_dir, config.agent_type, SETTING_NUM, RUN_NUM), sep=',', format='%15.8f')

    config_string = ""
    for key in config.__dict__:
        config_string += '{}: {}\n'.format(key, config.__getattribute__(key))
    with open("{}/{}_setting_{}_run_{}_experiment_params.txt".format(save_dir, config.agent_type, SETTING_NUM, RUN_NUM), "w") as config_file:
        config_file.write(config_string)


if __name__ == '__main__':
    main()