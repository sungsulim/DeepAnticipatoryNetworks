import matplotlib.pyplot as plt
import numpy as np


trainX_filename = 'results/train_return_per_episodeX.txt'
trainY_filename = 'results/train_return_per_episodeY.txt'
test_filename = 'results/test_return_per_episode.txt'

trainX = np.loadtxt(trainX_filename, delimiter=',')
trainY = np.loadtxt(trainY_filename, delimiter=',')


plt.xlabel("Episodes")
plt.ylabel("Return")

x_range = list(range(len(trainX)))
handle1, = plt.plot(x_range, trainX, color='red')
handle2, = plt.plot(x_range, trainY, color='blue')

plt.show()