import matplotlib.pyplot as plt
import numpy as np
import argparse

use_label = False
num_runs = 25

agent_type_arr = ['dan', 'dan_coverage', 'coverage', 'randomAction']
agent_best_setting = [0, 0, 0, 0]

ma_window = 100

yloc_arr = list(range(0, 16, 2))

ep_length = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()

    dir_name = args.result_dir

    # Multi-person test Result
    # plt.figure()

    handle_arr = []
    label_arr = []

    # mean_per_step_arr = {} # np.zeros((len(agent_type_arr), ep_length))
    # stderr_per_step_arr = {} # np.zeros((len(agent_type_arr), ep_length))

    # create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(5)
    bar_width = 0.18
    opacity = 0.8


    for idx, agent_type in enumerate(agent_type_arr):

        best_idx = agent_best_setting[idx]

        # results

        # Each step
        # mean_filename = '{}/{}_setting_{}_multiperson_test_return_mean.txt'.format(dir_name, agent_type, best_idx)
        # stderr_filename = '{}/{}_setting_{}_multiperson_test_return_stderr.txt'.format(dir_name, agent_type, best_idx)

        # Total return per episode
        mean_arr = []
        stderr_arr = []

        for batchsize in [1,2,5,10,20]:
            mean_stderr_filename = '{}/{}_setting_{}_batchsize_{}_multiperson_test_total_mean_stderr.txt'.format(dir_name, agent_type, best_idx, batchsize)
            mean, stderr = np.loadtxt(mean_stderr_filename, delimiter=',')

            mean_arr.append(mean * batchsize/500.0)
            stderr_arr.append(stderr)

        # x_range = list(range(len(mean_arr)))

        # Line graph

        # handle1, = plt.plot(x_range, mean)
        # plt.fill_between(x_range, mean - stderr, mean + stderr, alpha=0.2)
        #
        # handle_arr.append(handle1)
        # if use_label:
        #     label_arr.append(agent_type + ' setting: ' + str(best_idx))

        # Bar graph
        # data to plot
        plt.bar(index + idx * bar_width, mean_arr, bar_width,
                alpha=opacity,
                # color='b',
                label=agent_type)

    # xloc_arr = list(range(0, 12, 1))
    yloc_arr = list(range(0,120,20))
    xval_arr = [1,2,5,10,20]

    plt.ylim([0, 110])

    if use_label:
        plt.xlabel('Num. Tracked People')
        plt.ylabel('Correct Predictions per Episode')
        plt.title('Correct Predictions in Multi-person Tracking ({} Runs)'.format(num_runs))

        plt.xticks(index + 1.5 * bar_width, xval_arr)
        plt.yticks(yloc_arr, yloc_arr)
        plt.legend()

    else:
        plt.xticks(index + 1.5 * bar_width, [])
        plt.yticks(yloc_arr, [])

    # plt.legend(handle_arr, label_arr)

    # if use_label:
    #     plt.xlabel("Step")
    #     plt.ylabel("Return")
    #     plt.title('Mean Return per Step (Multi-person Tracking): {} Runs, {} Window'.format(num_runs, ma_window))
    #     plt.xticks(xloc_arr, xval_arr)
    #     plt.yticks(yloc_arr, yloc_arr)
    # else:
    #     plt.xticks(xloc_arr, [])
    #     plt.yticks(yloc_arr, [])

    plt.show()
    plt.close()

    # Bar chart
    # handle_arr = []
    # label_arr = []
    #
    #
    # for idx, agent_type in enumerate(agent_type_arr):
    #
    #     best_idx = agent_best_setting[idx]
    #
    #     # results
    #     mean_filename = '{}/{}_setting_{}_multiperson_test_return_mean.txt'.format(dir_name, agent_type, best_idx)
    #     stderr_filename = '{}/{}_setting_{}_multiperson_test_return_stderr.txt'.format(dir_name, agent_type, best_idx)
    #
    #     mean = np.loadtxt(mean_filename, delimiter=',')
    #     stderr = np.loadtxt(stderr_filename, delimiter=',')
    #
    #     x_range = list(range(len(mean)))
    #
    #     handle1, = plt.plot(x_range, mean)
    #     plt.fill_between(x_range, mean - stderr, mean + stderr, alpha=0.2)
    #
    #     handle_arr.append(handle1)
    #
    #     if use_label:
    #         label_arr.append(agent_type + ' setting: ' + str(best_idx))
    #
    # plt.legend(handle_arr, label_arr)
    #
    # xloc_arr = list(range(len(agent_type_arr)))
    # # yloc_arr = [100, 150, 200, 250]
    #
    # if use_label:
    #     plt.xticks(xloc_arr, agent_type_arr)
    #     plt.yticks(yloc_arr, yloc_arr)
    #
    #     plt.xlabel("Step")
    #     plt.ylabel("Return")
    #
    #     plt.title('Mean Return per Step (Multi-person Tracking): {} Runs, {} Window'.format(num_runs, ma_window))
    # else:
    #     plt.xticks(xloc_arr, [])
    #     # plt.yticks(yloc_arr, [])
    #
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main()
