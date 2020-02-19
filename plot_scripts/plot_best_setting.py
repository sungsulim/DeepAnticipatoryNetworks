import matplotlib.pyplot as plt
import numpy as np
import argparse


train_plot_idx_arr = []
test_plot_idx_arr = []

agent_type_arr = ['shared_attention']  # , 'randomAction', 'coverage', 'dan_coverage']
num_settings = 9
num_runs = 25

ma_window = 100


# xloc_arr = [0, 101,]
# xval_arr = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
yloc_arr = list(range(0, 16, 2))

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()

    dir_name = args.result_dir

    # Train Result
    plt.figure(figsize=(12, 6))
    plt.xlabel("Episodes")
    plt.ylabel("Return")

    handle_arr = []
    label_arr = []

    for agent_type in agent_type_arr:

        trainX_settings_mean = []
        trainY_settings_mean = []
        trainX_settings_stderr = []
        trainY_settings_stderr = []

        for s in range(num_settings):

            trainX_runs = []
            trainY_runs = []

            for i in range(num_runs):

                # Train results
                trainX_filename = '{}/{}_setting_{}_run_{}_train_return_per_episodeX.txt'.format(dir_name, agent_type, s, i)
                trainY_filename = '{}/{}_setting_{}_run_{}_train_return_per_episodeY.txt'.format(dir_name, agent_type, s, i)

                trainX = np.loadtxt(trainX_filename, delimiter=',')
                trainY = np.loadtxt(trainY_filename, delimiter=',')

                trainX = movingaverage(trainX, ma_window)
                trainY = movingaverage(trainY, ma_window)

                trainX_runs.append(trainX)
                trainY_runs.append(trainY)

                # Test results
                # test_filename = 'results/{}/{}_{}_test_return_per_episode.txt'.format(dir_name, agent_type, i)

            trainX_mean = np.mean(trainX_runs, axis=0)
            trainY_mean = np.mean(trainY_runs, axis=0)
            trainX_stderr = np.std(trainX_runs, axis=0) / np.sqrt(num_runs)
            trainY_stderr = np.std(trainY_runs, axis=0) / np.sqrt(num_runs)

            trainX_settings_mean.append(trainX_mean)
            trainY_settings_mean.append(trainY_mean)

            trainX_settings_stderr.append(trainX_stderr)
            trainY_settings_stderr.append(trainY_stderr)

        trainX_settings_mean = np.array(trainX_settings_mean)
        trainY_settings_mean = np.array(trainY_settings_mean)

        trainX_settings_stderr = np.array(trainX_settings_stderr)
        trainY_settings_stderr = np.array(trainY_settings_stderr)


        if len(train_plot_idx_arr) == 0:
            train_sum = np.sum(trainX_settings_mean + trainY_settings_mean, axis=1)
            # descending order
            sorted_idx = np.flip(np.argsort(train_sum))

            for idx in sorted_idx:
                print("setting {}: {}".format(idx, train_sum[idx]))

            best_idx = sorted_idx[0]
            print("Best train setting: {}".format(best_idx))

            with open("{}/{}_setting_{}_run_{}_experiment_params.txt".format(dir_name, agent_type, best_idx, 0), "r") as readfile:
                for line in readfile:
                    print(line[:-1])

            train_plot_idx_arr.append(best_idx)

        for idx in train_plot_idx_arr:
            x_range = list(range(len(trainX_settings_mean[idx])))

            handle1, = plt.plot(x_range, trainX_settings_mean[idx])
            handle2, = plt.plot(x_range, trainY_settings_mean[idx])
            plt.fill_between(x_range, trainX_settings_mean[idx] - trainX_settings_stderr[idx], trainX_settings_mean[idx] + trainX_settings_stderr[idx], alpha=0.2)
            plt.fill_between(x_range, trainY_settings_mean[idx] - trainY_settings_stderr[idx], trainY_settings_mean[idx] + trainY_settings_stderr[idx], alpha=0.2)

            handle_arr.append(handle1)
            handle_arr.append(handle2)

            label_arr.append(agent_type + 'X' + ' setting '+str(idx))
            label_arr.append(agent_type + 'Y' + ' setting '+str(idx))

    plt.legend(handle_arr, label_arr)
    plt.title('Return per Episode (Training): {} Runs, {} Window'.format(num_runs, ma_window))

    # plt.xticks(xloc_arr, xval_arr)
    plt.yticks(yloc_arr, yloc_arr)

    plt.show()
    plt.close()

    print()

    # Test Result
    plt.figure(figsize=(12, 6))
    plt.xlabel("Episodes")
    plt.ylabel("Return")

    handle_arr = []
    label_arr = []

    for agent_type in agent_type_arr:

        testX_settings_mean = []
        # testY_settings_mean = []
        testX_settings_stderr = []
        # testY_settings_stderr = []

        for s in range(num_settings):

            testX_runs = []
            # testY_runs = []

            for i in range(num_runs):

                # Train results
                testX_filename = '{}/{}_setting_{}_run_{}_test_mean_return_per_episode.txt'.format(dir_name, agent_type, s, i)
                # testY_filename = '{}/{}_setting_{}_run_{}_test_mean_return_per_episodeY.txt'.format(dir_name, agent_type, s, i)

                testX = np.loadtxt(testX_filename, delimiter=',')
                # testY = np.loadtxt(testY_filename, delimiter=',')

                testX_runs.append(testX)
                # testY_runs.append(testY)

                # Test results
                # test_filename = 'results/{}/{}_{}_test_return_per_episode.txt'.format(dir_name, agent_type, i)

            testX_mean = np.mean(testX_runs, axis=0)
            # testY_mean = np.mean(testY_runs, axis=0)
            testX_stderr = np.std(testX_runs, axis=0) / np.sqrt(num_runs)
            # testY_stderr = np.std(testY_runs, axis=0) / np.sqrt(num_runs)

            testX_settings_mean.append(testX_mean)
            # testY_settings_mean.append(testY_mean)

            testX_settings_stderr.append(testX_stderr)
            # testY_settings_stderr.append(testY_stderr)

        testX_settings_mean = np.array(testX_settings_mean)
        # testY_settings_mean = np.array(testY_settings_mean)

        testX_settings_stderr = np.array(testX_settings_stderr)
        # testY_settings_stderr = np.array(testY_settings_stderr)


        if len(test_plot_idx_arr) == 0:
            test_sum = np.sum(testX_settings_mean, axis=1)
            # descending order
            sorted_idx = np.flip(np.argsort(test_sum))

            for idx in sorted_idx:
                print("setting {}: {}".format(idx, test_sum[idx]))

            best_idx = sorted_idx[0]
            print("Best test setting: {}".format(best_idx))

            with open("{}/{}_setting_{}_run_{}_experiment_params.txt".format(dir_name, agent_type, best_idx, 0), "r") as readfile:
                for line in readfile:
                    print(line[:-1])

            test_plot_idx_arr.append(best_idx)

        for idx in test_plot_idx_arr:
            x_range = list(range(len(testX_settings_mean[idx])))

            handle1, = plt.plot(x_range, testX_settings_mean[idx])
            # handle2, = plt.plot(x_range, testY_settings_mean[idx])
            plt.fill_between(x_range, testX_settings_mean[idx] - testX_settings_stderr[idx], testX_settings_mean[idx] + testX_settings_stderr[idx], alpha=0.2)
            # plt.fill_between(x_range, testY_settings_mean[idx] - testY_settings_stderr[idx], testY_settings_mean[idx] + testY_settings_stderr[idx], alpha=0.2)

            handle_arr.append(handle1)
            # handle_arr.append(handle2)

            label_arr.append(agent_type + ' setting '+str(idx))
            # label_arr.append(agent_type + 'Y' + ' setting '+str(idx))

    plt.legend(handle_arr, label_arr)
    plt.title('Return per Episode (Testing): {} Runs, {} Window'.format(num_runs, ma_window))

    # plt.xticks(xloc_arr, xval_arr)
    plt.yticks(yloc_arr, yloc_arr)

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
