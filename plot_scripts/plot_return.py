import matplotlib.pyplot as plt
import numpy as np
import argparse

num_runs = 7

agent_type_arr = ['dan', 'randomAction', 'coverage', 'dan_coverage']
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

        trainX_arr = []
        trainY_arr = []

        for i in range(num_runs):

            # Train results
            trainX_filename = '{}/{}_{}_train_return_per_episodeX.txt'.format(dir_name, agent_type, i)
            trainY_filename = '{}/{}_{}_train_return_per_episodeY.txt'.format(dir_name, agent_type, i)

            trainX = np.loadtxt(trainX_filename, delimiter=',')
            trainY = np.loadtxt(trainY_filename, delimiter=',')

            trainX = movingaverage(trainX, ma_window)
            trainY = movingaverage(trainY, ma_window)

            trainX_arr.append(trainX)
            trainY_arr.append(trainY)

            # Test results
            # test_filename = 'results/{}/{}_{}_test_return_per_episode.txt'.format(dir_name, agent_type, i)

        trainX_mean = np.mean(trainX_arr, axis=0)
        trainY_mean = np.mean(trainY_arr, axis=0)
        trainX_stderr = np.std(trainX_arr, axis=0) / np.sqrt(num_runs)
        trainY_stderr = np.std(trainY_arr, axis=0) / np.sqrt(num_runs)

        x_range = list(range(len(trainX_mean)))

        handle1, = plt.plot(x_range, trainX_mean)
        handle2, = plt.plot(x_range, trainY_mean)
        plt.fill_between(x_range, trainX_mean - trainX_stderr, trainX_mean + trainX_stderr, alpha=0.2)
        plt.fill_between(x_range, trainY_mean - trainY_stderr, trainY_mean + trainY_stderr, alpha=0.2)

        handle_arr.append(handle1)
        handle_arr.append(handle2)

        label_arr.append(agent_type + 'X')
        label_arr.append(agent_type + 'Y')

    plt.legend(handle_arr, label_arr)
    plt.title('Return per Episode (Training): {} Runs, {} Window'.format(num_runs, ma_window))

    # plt.xticks(xloc_arr, xval_arr)
    plt.yticks(yloc_arr, yloc_arr)

    plt.show()
    plt.close()

    # Test Result
    plt.figure(figsize=(12, 6))
    plt.xlabel("Episodes")
    plt.ylabel("Return")

    handle_arr = []
    label_arr = []

    for agent_type in agent_type_arr:

        testX_arr = []
        testY_arr = []

        for i in range(num_runs):
            # Train results
            testX_filename = '{}/{}_{}_test_mean_return_per_episodeX.txt'.format(dir_name, agent_type, i)
            testY_filename = '{}/{}_{}_test_mean_return_per_episodeY.txt'.format(dir_name, agent_type, i)

            testX = np.loadtxt(testX_filename, delimiter=',')
            testY = np.loadtxt(testY_filename, delimiter=',')

            # testX = movingaverage(testX, ma_window)
            # testY = movingaverage(testY, ma_window)

            testX_arr.append(testX)
            testY_arr.append(testY)

            # Test results
            # test_filename = 'results/{}/{}_{}_test_return_per_episode.txt'.format(dir_name, agent_type, i)

        testX_mean = np.mean(testX_arr, axis=0)
        testY_mean = np.mean(testY_arr, axis=0)
        testX_stderr = np.std(testX_arr, axis=0) / np.sqrt(num_runs)
        testY_stderr = np.std(testY_arr, axis=0) / np.sqrt(num_runs)

        x_range = list(range(len(testX_mean)))

        handle1, = plt.plot(x_range, testX_mean)
        handle2, = plt.plot(x_range, testY_mean)
        plt.fill_between(x_range, testX_mean - testX_stderr, testX_mean + testX_stderr, alpha=0.2)
        plt.fill_between(x_range, testY_mean - testY_stderr, testY_mean + testY_stderr, alpha=0.2)

        handle_arr.append(handle1)
        handle_arr.append(handle2)

        label_arr.append(agent_type + 'X')
        label_arr.append(agent_type + 'Y')

    plt.legend(handle_arr, label_arr)
    plt.title('Return per Episode (Testing): {} Runs, {} Window'.format(num_runs, ma_window))

    # plt.xticks(xloc_arr, xval_arr)
    plt.yticks(yloc_arr, yloc_arr)

    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
