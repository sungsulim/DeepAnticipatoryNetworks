import matplotlib.pyplot as plt
import numpy as np

prefix_arr = ['normal', 'randomAction', 'coverage']


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma

plt.xlabel("Episodes")
plt.ylabel("Return")


handle_arr = []
label_arr = []

for prefix in prefix_arr:

    trainX_filename = 'results/{}_train_return_per_episodeX.txt'.format(prefix)
    trainY_filename = 'results/{}_train_return_per_episodeY.txt'.format(prefix)
    test_filename = 'results/test_return_per_episode.txt'

    trainX = np.loadtxt(trainX_filename, delimiter=',')
    trainY = np.loadtxt(trainY_filename, delimiter=',')

    window = 100
    trainX = movingaverage(trainX, window)
    trainY = movingaverage(trainY, window)

    x_range = list(range(len(trainX)))
    handle1, = plt.plot(x_range, trainX, label=prefix+'X')
    handle2, = plt.plot(x_range, trainY, label=prefix+'Y')

    handle_arr.append(handle1)
    handle_arr.append(handle2)

    label_arr.append(prefix+'X')
    label_arr.append(prefix + 'Y')

plt.legend(handle_arr, label_arr)
plt.show()
