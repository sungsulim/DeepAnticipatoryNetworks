import numpy as np


def load_titanic():
    raise NotImplementedError
    # Some data is missing. Needs additional preprocessing
    # np.loadtxt("data/titanic/", delimiter=',')

def load_pokerhand():
    # training: 25010
    # testing: 1000     # 1000000
    # train_data = np.loadtxt("data/pokerhand/poker-hand-training-true.data", delimiter=',')
    # test_data = np.loadtxt("data/pokerhand/poker-hand-testing.data", delimiter=',')
    #
    # train_data = np.loadtxt("poker-hand-training-true.data", delimiter=',')
    # test_data = np.loadtxt("poker-hand-testing.data", delimiter=',')

    train_data = np.load("data/pokerhand/pokerhand_train_data.npy")
    train_label = np.load("data/pokerhand/pokerhand_train_label.npy")
    test_data = np.load("data/pokerhand/pokerhand_test_data.npy")[:1000]
    test_label = np.load("data/pokerhand/pokerhand_test_label.npy")[:1000]

    return train_data, train_label, test_data, test_label

