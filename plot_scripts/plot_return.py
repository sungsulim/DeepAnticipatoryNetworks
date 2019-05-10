import matplotlib.pyplot as plt
import numpy as np

agent_type_arr = ['normal', 'randomAction', 'coverage']

num_runs = 8
ma_window = 100


# xloc_arr = [0, 101,]
# xval_arr = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
yloc_arr = list(range(0, 16, 2))

def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def main():
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
            trainX_filename = 'results/{}_{}_train_return_per_episodeX.txt'.format(agent_type, i)
            trainY_filename = 'results/{}_{}_train_return_per_episodeY.txt'.format(agent_type, i)

            trainX = np.loadtxt(trainX_filename, delimiter=',')
            trainY = np.loadtxt(trainY_filename, delimiter=',')

            trainX_arr.append(trainX)
            trainY_arr.append(trainY)

            # Test results
            # test_filename = 'results/{}_{}_test_return_per_episode.txt'.format(agent_type, i)

        trainX_mean = np.mean(trainX_arr, axis=0)
        trainY_mean = np.mean(trainY_arr, axis=0)


        trainX_mean = movingaverage(trainX_mean, ma_window)
        trainY_mean = movingaverage(trainY_mean, ma_window)

        x_range = list(range(len(trainX_mean)))
        handle1, = plt.plot(x_range, trainX_mean)
        handle2, = plt.plot(x_range, trainY_mean)

        handle_arr.append(handle1)
        handle_arr.append(handle2)

        label_arr.append(agent_type + 'X')
        label_arr.append(agent_type + 'Y')

    plt.legend(handle_arr, label_arr)
    plt.title('Return per Episode (Training): {} Runs, {} Window'.format(num_runs, ma_window))

    # plt.xticks(xloc_arr, xval_arr)
    plt.yticks(yloc_arr, yloc_arr)

    plt.show()


if __name__ == '__main__':
    main()
