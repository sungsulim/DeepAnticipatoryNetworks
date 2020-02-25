import matplotlib.pyplot as plt
import numpy as np
import argparse


use_label = True
num_runs = 25

agent_type_arr = ['dan', 'dan_shared', 'dan_coverage', 'coverage', 'random_policy']
agent_best_setting = [4, 4, 4, 1, 1]

ma_window = 100

yloc_arr = list(range(0, 16, 2))

ep_length = 12

# black-n-white compatible
color_arr = ['#16454E', '#2B6F39', '#747A32', '#D38FC5', '#C2C1F2', '#C6E2E6'] # '#C1796F',


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()

    dir_name = args.result_dir

    # create plot
    # fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(5)
    bar_width = 0.16
    opacity = 0.8

    for idx, agent_type in enumerate(agent_type_arr):

        best_idx = agent_best_setting[idx]

        # Total return per episode
        mean_arr = []
        stderr_arr = []

        for batchsize in [1,2,5,10,20]:
            mean_stderr_filename = '{}/{}_setting_{}_batchsize_{}_multiperson_test_total_mean_stderr.txt'.format(dir_name, agent_type, best_idx, batchsize)
            mean, stderr = np.loadtxt(mean_stderr_filename, delimiter=',')

            mean_arr.append(mean * batchsize/500.0)
            stderr_arr.append(stderr)

        # Bar graph
        # data to plot
        plt.bar(index + idx * bar_width, mean_arr, bar_width,
                alpha=opacity,
                color=color_arr[idx],
                label=agent_type)

    yloc_arr = list(range(0,120,20))
    xval_arr = [1,2,5,10,20]

    plt.ylim([0, 110])

    if use_label:
        plt.xlabel('Num. Tracked People')
        plt.ylabel('Correct Predictions per Episode')
        plt.title('Correct Predictions in Multi-person Tracking ({} Runs)'.format(num_runs))

        plt.xticks(index + 2 * bar_width, xval_arr)
        plt.yticks(yloc_arr, yloc_arr)
        plt.legend()

    else:
        plt.xticks(index + 2 * bar_width, [])
        plt.yticks(yloc_arr, [])

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
