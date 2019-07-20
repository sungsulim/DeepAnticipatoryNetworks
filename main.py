import os
from sklearn.datasets import load_iris
from sklearn import tree

import numpy as np
from decision_tree import dataset_loader
import argparse

from decision_tree.config import Config
from environments.decisiontreeEnv import DecisionTreeEnv
from agents.dan_dt import DAN
from decision_tree.experiment import Experiment

def runDecisionTree(train_data, train_label, test_data, test_label, env_params):

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_label)

    # DT in text format
    # r = export_text(clf, feature_names=iris['feature_names'])
    # print(r)

    print("Test accuracy: {}".format(clf.score(test_data, test_label)))


def runDAN(train_data, train_label, test_data, test_label, env_params):

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--agent_type', type=str)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()

    arg_params = {
        "agent_type": args.agent_type
    }

    config = Config()
    config.merge_config(env_params)
    config.merge_config(arg_params)

    # # create env
    train_env = DecisionTreeEnv(train_data, train_label, config)
    test_env = DecisionTreeEnv(test_data, test_label, config)


    # create agent
    agent = DAN(config)

    # create experiment
    experiment = Experiment(train_env=train_env,
                            test_env=test_env,
                            agent=agent,
                            config=config)

    # run experiment
    train_return_per_episode, test_mean_return_per_episode = experiment.run()

    # save results
    save_dir = 'results/{}'.format(args.result_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_prefix = '{}/{}_run_{}'.format(save_dir, config.agent_type, args.index)
    # Train result
    np.array(train_return_per_episode).tofile("{}_train_return_per_episode.txt".format(save_prefix), sep=',', format='%15.8f')

    # Test result
    np.array(test_mean_return_per_episode).tofile("{}_test_mean_return_per_episode.txt".format(save_prefix), sep=',', format='%15.8f')

    # save model
    agent.save_network(save_prefix)

    config_string = ""
    for key in config.__dict__:
        config_string += '{}: {}\n'.format(key, config.__getattribute__(key))
    with open("{}_experiment_params.txt".format(save_prefix), "w") as config_file:
        config_file.write(config_string)


def main():

    dataset_name = 'iris'
    method_name = 'dan'

    test_ratio = 0.2

    rng_state = np.random.RandomState(9999)

    ##### Load dataset

    # Iris dataset
    if dataset_name == 'iris':
        data, target = load_iris(return_X_y=True)

        env_params = {
            "nStates": 3,
            "nActions": 4,
            "max_ep_length": 4
        }

    # Titanic dataset
    elif dataset_name == 'titanic':
        data, target = dataset_loader.load_titanic()

        # nStates = 3
        # nActions = 4

        # pass
    elif dataset_name == 'pokerhand':
        train_data, train_label, test_data, test_label = dataset_loader.load_pokerhand()

        env_params = {
            "nStates": 10,
            "nActions": 10,
            "max_ep_length": 10
        }

    else:
        raise ValueError("Dataset not found.")

    # Divide into train/test split
    idx_arr = list(range(len(data)))
    rng_state.shuffle(idx_arr)
    
    num_test = int(len(idx_arr) * test_ratio)
    
    train_idx = idx_arr[num_test:]
    test_idx = idx_arr[:num_test]
    
    train_data = data[train_idx]
    train_label = target[train_idx]
    
    test_data = data[test_idx]
    test_label = target[test_idx]

    # Run Decision Tree
    if method_name == 'decision_tree':
        runDecisionTree(train_data, train_label, test_data, test_label, env_params)
    elif method_name == 'dan':
        runDAN(train_data, train_label, test_data, test_label, env_params)
    else:
        raise ValueError("Method not found.")


if __name__ == '__main__':
    main()
