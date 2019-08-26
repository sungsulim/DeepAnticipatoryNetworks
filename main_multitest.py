import numpy as np
import argparse
import os
import json
from collections import OrderedDict

from datetime import datetime
import time

from config import Config
from experiment import Experiment
from environments.trackingEnv import SSenvReal
from agents.dan_tracking import DAN

def get_sweep_parameters(parameters, index):
    out = OrderedDict()
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        out[key] = parameters[key][int(index / accum) % num]
        accum *= num
    return (out, accum)


def main_multitest():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--agent_json', type=str)
    parser.add_argument('--index', type=int)
    parser.add_argument('--num_runs', type=int)
    parser.add_argument('--test_batch_size', type=int)
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

    print("Testing {} Setting {}".format(config.agent_type, SETTING_NUM))

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
    test_track_idx = track_idx[:int(config.test_ep_num)]
    print("test track num: {}".format(len(test_track_idx)))  # 50

    # (10, 12)
    multiperson_mean_return_per_run = np.zeros((args.num_runs, config.max_ep_length))

    # create env
    test_env = SSenvReal(config, 'data/sampled_tracks_new', test_track_idx)

    # create agent
    agentX = DAN(config, 'x')
    agentY = DAN(config, 'y')

    for r in range(args.num_runs):
        print("Run {}".format(r))


        # Restore model
        model_dir = 'results/{}'.format(args.model_dir)
        model_prefix = '{}/{}_setting_{}_run_{}'.format(model_dir, config.agent_type, SETTING_NUM, r)

        agentX.restore_network(model_prefix, 'x')
        agentY.restore_network(model_prefix, 'y')
        print("loaded model...")

        # create experiment
        experiment = Experiment(train_env={'x': None, 'y': None},
                                test_env=test_env,
                                agent={'x': agentX, 'y': agentY},
                                config=config)

        # Verify loaded model
        # Test once (single person tracking)
        # print("Verifying loaded model...")
        # test_session_time, mean_return_per_episode = experiment.test()
        #
        # saved_result_filename = '{}_test_mean_return_per_episode.txt'.format(model_prefix)
        # saved_result = np.loadtxt(saved_result_filename, delimiter=',')
        #
        # print("Single person Test Time: " + time.strftime("%H:%M:%S", time.gmtime(test_session_time)))
        # print("Mean return per episode: {}, saved result: {}".format(mean_return_per_episode, saved_result[-1]))
        # if config.agent_type != 'randomAction':
        #     assert(mean_return_per_episode == saved_result[-1])

        # Test multiperson (all test tracks simultaneously)
        episode_return_arr, episode_step_count = experiment.multiperson_test(args.test_batch_size)

        multiperson_mean_return_per_run[r] += episode_return_arr
        print("Run {} multiperson test result".format(r))
        print(episode_return_arr)

    print("Testing {} Setting {} multiperson mean return per run:".format(config.agent_type, SETTING_NUM))
    multiperson_return_mean = np.mean(multiperson_mean_return_per_run, axis=0)
    multiperson_return_stderr = np.std(multiperson_mean_return_per_run, axis=0) / np.sqrt(args.num_runs)

    print(multiperson_return_mean)

    total_return_per_run = np.sum(multiperson_mean_return_per_run, axis=1)
    mean_return = np.mean(total_return_per_run, axis=0)
    stderr_return = np.std(total_return_per_run, axis=0) / np.sqrt(args.num_runs)
    print("Total return mean: {}, stderr: {}".format(mean_return, stderr_return))

    # save result
    save_prefix = '{}/{}_setting_{}'.format(save_dir, config.agent_type, SETTING_NUM)
    np.array(multiperson_return_mean).tofile("{}_multiperson_test_return_mean.txt".format(save_prefix), sep=',',
                                                  format='%15.8f')
    np.array(multiperson_return_stderr).tofile("{}_multiperson_test_return_stderr.txt".format(save_prefix), sep=',',
                                             format='%15.8f')

    with open("{}_multiperson_test_total_mean_stderr.txt".format(save_prefix), "w") as writefile:
        writefile.write("{}, {}".format(mean_return, stderr_return))


if __name__ == '__main__':
    main_multitest()
