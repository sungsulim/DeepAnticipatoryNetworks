import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import pickle as pk
import tensorflow.contrib.slim as slim

import random




def updateTargetGraph(tfVars,tau=0.9):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+12].assign((var.value()*tau) + ((1-tau)*tfVars[idx+12].value())))
    return op_holder


def updateTargetMGraph(tfVars, tau=0.9):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+14].assign((var.value()*tau)) + (1-tau)*tfVars[idx+14].value())
    return op_holder


def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


class SSenvReal:

    def __init__(self, xory):

        self.nStates = 21  # (these are the number of cells possible in the x space)
        self.nActions = 10  # (0 to 9)
        self.testOrTrain = 'train'
        self.k = 0
        self.index = 0
        self.xory = xory
        self.max_epLength = 16
        self.track = self.getNewTrack()
        self.state = self.get_next_state(self.track, self.k)
        self.cc_by_sensors = self.get_covered_cells_by_sensors()
        self.obsmat = [0.96, 0.93, 0.91, 0.88, 0.85, 0.81, 0.78, 0.75, 0.72, 0.7]

    def discretize_cell(self, contstate):

        xcell = int((contstate[0]) * 2 / 15)
        ycell = int((contstate[1]) * 2 / 15)
        discstate = [xcell, ycell]
        return discstate

    def get_next_state(self, track, index):

        index = index + 1
        if index < len(track[0]):
            return [track[0][index], track[1][index]]
        else:
            return [21, 21]

    def getNewTrack(self):

        return self.load_track()

    def load_track(self):

        if self.xory == 'x':
            sx = pk.load(open('/content/drive/My Drive/Dancode/sampled_tracksX', 'rb'))
        elif self.xory == 'y':
            sx = pk.load(open('/content/drive/My Drive/Dancode/sampled_tracksX', 'rb'))
        tracknu = np.random.randint(0, len(sx))
        trck = sx[tracknu]

        return trck

    def get_cell(self, xypoint):

        xp = xypoint[0] + 50
        yp = xypoint[1] + 50
        xcell = xp // 1
        ycell = yp // 1
        return [xcell, ycell]

    def get_covered_cells_by_sensors(self):

        mat_content = sio.loadmat('/content/drive/My Drive/Dancode/dandc.mat')
        cell_info = mat_content['c2']
        covered_cells_by_sensors = {}
        for i in range(10):
            covered_cells_by_sensors[i] = cell_info[0][i][0].tolist()

        for i in range(10):
            covered_cells_by_sensors[i].extend(cell_info[0][i + 10][0].tolist())

        return covered_cells_by_sensors

    def get_obs(self, cellstate, action):

        cc_by_s = self.cc_by_sensors[action]

        if cellstate in cc_by_s:
            temp = self.discretize_cell(cellstate)
            return temp
        else:
            return [21, 21]

    def get_obsX(self, cellstate, action):

        temp = self.get_obs(cellstate, action)
        if np.random.rand(1) < self.obsmat[action]:
            return temp[0]
        else:
            return 21

    def get_obsY(self, cellstate, action):
        temp = self.get_obs(cellstate, action)
        if np.random.rand(1) < self.obsmat[action]:
            return temp[1]
        else:
            return 21


class Qnetwork():

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __init__(self, h_size, nStates, nActions, rnn_cell, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.inpSize = nStates + nActions
        self.scalarInput = tf.placeholder(shape=[None, self.inpSize], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, self.inpSize])

        with tf.variable_scope(myScope + "_fclayers"):
            self.weight1 = self.weight_variable([self.inpSize, 40])
            self.bias1 = self.bias_variable([40])

            self.weight11 = self.weight_variable([40, 20])
            self.bias11 = self.bias_variable([20])

            self.weight2 = self.weight_variable([20, h_size])
            self.bias2 = self.bias_variable([h_size])

            self.wfin = self.weight_variable([h_size, nActions])
            self.bfin = self.bias_variable([nActions])

        self.hstate1 = (tf.matmul(self.scalarInput, self.weight1) + self.bias1)
        self.hstate11 = tf.matmul(self.hstate1, self.weight11) + self.bias11
        self.hstate2 = tf.matmul(self.hstate11, self.weight2) + self.bias2

        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.reshape(slim.flatten(self.hstate2), [self.batch_size, self.trainLength, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        # The output from the recurrent player is then split into separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(self.rnn, 2, 1)
        with tf.variable_scope(myScope + "_fclayers"):
            self.AW = tf.Variable(tf.random_normal([h_size // 2, nActions]))
            self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.salience = tf.gradients(self.Advantage, self.imageIn)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, nActions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, self.trainLength // 2])
        self.maskB = tf.ones([self.batch_size, self.trainLength // 2])
        self.mask = tf.concat([self.maskA, self.maskB], 1)
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class Mnetwork():

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __init__(self, h_size, nStates, nActions, rnn_cell, myScope):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.inpSize = nStates + nActions
        self.scalarInput = tf.placeholder(shape=[None, inpSize], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 1, inpSize])
        # self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        # self.actions_onehot = tf.one_hot(self.actions,nStates,dtype=tf.float32)
        # self.newScalarInput = tf.concat([self.scalarInput, self.actions_onehot],1)
        with tf.variable_scope(myScope + "_fclayers"):
            self.weight1 = self.weight_variable([inpSize, 40])
            self.bias1 = self.bias_variable([40])

            self.weight11 = self.weight_variable([40, 20])
            self.bias11 = self.bias_variable([20])

            self.weight2 = self.weight_variable([20, 10])
            self.bias2 = self.bias_variable([10])

            self.weight3 = self.weight_variable([10, h_size])
            self.bias3 = self.bias_variable([h_size])

            self.weight4 = self.weight_variable([h_size, 40])
            self.bias4 = self.bias_variable([40])

            self.wfin = self.weight_variable([40, nStates])
            self.bfin = self.bias_variable([nStates])

        self.hstate1 = (tf.matmul(self.scalarInput, self.weight1) + self.bias1)
        self.hstate11 = tf.matmul(self.hstate1, self.weight11) + self.bias11
        self.hstate2 = tf.matmul(self.hstate11, self.weight2) + self.bias2
        self.hstate3 = tf.matmul(self.hstate2, self.weight3) + self.bias3

        self.trainLength = tf.placeholder(dtype=tf.int32)
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.convFlat = tf.reshape(slim.flatten(self.hstate3), [self.batch_size, self.trainLength, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32, initial_state=self.state_in, scope=myScope + '_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])

        self.prepred = tf.matmul(self.rnn, self.weight4) + self.bias4
        self.prediction = tf.matmul(self.prepred, self.wfin) + self.bfin

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetP = tf.placeholder(shape=[None], dtype=tf.int32)
        self.targetP_onehot = tf.one_hot(self.targetP, nStates, dtype=tf.float32)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.targetP_onehot, logits=self.prediction))

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)


class experience_buffer():
    def __init__(self, buffer_size=20000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 6])



tf.reset_default_graph()
sse = SSenvReal('x')

nActions = sse.nActions
nStates = sse.nStates
inpSize = nStates + nActions
file = open('/content/drive/My Drive/Dancode/testfileX.txt', 'w')

# We define the cells for the primary and target q-networks for x coordinates
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = Qnetwork(h_size, sse.nStates, sse.nActions, cell, 'mainqx')
targetQN = Qnetwork(h_size, sse.nStates, sse.nActions, cellT, 'targetqx')

mcell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mcellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainMN = Mnetwork(h_size, sse.nStates, sse.nActions, mcell, 'mmainx')
targetMN = Mnetwork(h_size, sse.nStates, sse.nActions, mcellT, 'mtargetx')

# We define the same thing for Y
cellY = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellTY = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQNY = Qnetwork(h_size, sse.nStates, sse.nActions, cellY, 'mainqy')
targetQNY = Qnetwork(h_size, sse.nStates, sse.nActions, cellTY, 'targetqy')

mcellY = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mcellTY = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainMNY = Mnetwork(h_size, sse.nStates, sse.nActions, mcellY, 'mmainy')
targetMNY = Mnetwork(h_size, sse.nStates, sse.nActions, mcellTY, 'mtargety')



def get_pred_reward(prediction, ns_cell):
    if np.argmax(prediction) == ns_cell:
        r = 1
    else:
        r = 0
    return r


def get_cover_reward(obs):
    if obs < 21:
        r = 1
    else:
        r = 0
    return r


def format_obs(obs, nStates, a_onehot):
    obs_onehot = np.reshape(np.array([int(i == obs) for i in range(nStates)]), [1, nStates])
    fin_obs = np.reshape(np.append(obs_onehot, a_onehot), [1, inpSize])
    return fin_obs


def get_Qaction(mainQN, state, sess, prev_obs):
    if len(prev_obs) == 1:
        stateQx = state
        a = np.random.randint(0, sse.nActions)
        return a, stateQx
    else:
        Q, stateQ = sess.run([mainQN.Qout, mainQN.rnn_state], feed_dict={mainQN.scalarInput: prev_obs, mainQN.trainLength: 1, mainQN.state_in: state,mainQN.batch_size: 1})
        a = np.argmax(Q, 1)
        a = a[0]
        return a, stateQ


def run_episode(sse, mainQN, mainMN, sess, xory):
    prev_obsx = np.reshape([0] * (sse.nStates + sse.nActions), [1, 31])
    state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    stateQx = state
    stateMx = state
    episodeBuffer = []
    d = False
    rew = 0
    for i in range(0, sse.max_epLength):

        a, stateQx = get_Qaction(mainQN, stateQx, sess, prev_obsx)
        a_onehot = np.reshape(np.array([int(i == a) for i in range(sse.nActions)]), [1, sse.nActions])

        ns = (sse.get_cell(sse.get_next_state(sse.track, sse.index)))
        ns_cell = sse.discretize_cell(ns)
        x_cell = ns_cell[0]
        y_cell = ns_cell[1]
        if xory == 'x':
            obsx = np.reshape(sse.get_obsX(ns, a), [1, 1])
            curr_obsx = format_obs(obsx, sse.nStates, a_onehot)
            prediction, stateMx = sess.run([mainMN.prediction, mainMN.rnn_state],
                                           feed_dict={mainMN.scalarInput: curr_obsx, mainMN.trainLength: 1,
                                                      mainMN.state_in: stateMx, mainMN.batch_size: 1})
            r = get_pred_reward(prediction, x_cell)
            episodeBuffer.append(np.reshape(np.array([prev_obsx, a, r, curr_obsx, d, x_cell]), [1, 6]))
        elif xory == 'y':
            obsx = np.reshape(sse.get_obsY(ns, a), [1, 1])
            curr_obsx = format_obs(obsx, sse.nStates, a_onehot)
            prediction, stateMx = sess.run([mainMN.prediction, mainMN.rnn_state],
                                           feed_dict={mainMN.scalarInput: curr_obsx, mainMN.trainLength: 1,
                                                      mainMN.state_in: stateMx, mainMN.batch_size: 1})
            r = get_pred_reward(prediction, y_cell)
            episodeBuffer.append(np.reshape(np.array([prev_obsx, a, r, curr_obsx, d, y_cell]), [1, 6]))
        rew = rew + r
        prev_obsx = curr_obsx

    return episodeBuffer, rew


def updateQandM(myBuffer, mainQN, mainMN):
    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))
    trainBatch = myBuffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
    # Below we perform the Double-DQN update to the target Q-values
    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3]), mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size})
    Q2 = sess.run(targetQN.Qout, feed_dict={ targetQN.scalarInput: np.vstack(trainBatch[:, 3]), targetQN.trainLength: trace_length, targetQN.state_in: state_train, targetQN.batch_size: batch_size})
    end_multiplier = -(trainBatch[:, 4] - 1)
    doubleQ = Q2[range(batch_size * trace_length), Q1]
    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
    sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length,
                                            mainQN.state_in: state_train, mainQN.batch_size: batch_size})

    sess.run(mainMN.updateModel, feed_dict={mainMN.scalarInput: np.vstack(trainBatch[:, 3]), mainMN.targetP: trainBatch[:, 5],
                                            mainMN.trainLength: trace_length,
                                            mainMN.state_in: state_train, mainMN.batch_size: batch_size})
    return mainQN, mainMN


init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)
trainables = tf.trainable_variables()
variables_names = [v.name for v in tf.trainable_variables()]

targetOps = updateTargetGraph(trainables[0:24],tau)
targetOpsM = updateTargetMGraph(trainables[24:52], tau)
targetOpsY = updateTargetGraph(trainables[52:76],tau)
targetOpsMY = updateTargetMGraph(trainables[76:104],tau)




## Here we are learning mainQ and mainM over the x-coordinates.

myBuffer = experience_buffer()
myBufferY = experience_buffer()
total_steps = 0
total_eps = 0

e = startE
stepDrop = (startE - endE)/anneling_steps

jList = []

rList = []
rListY = []

with tf.Session() as sess:
    sess.run(init)
    for ieps in range(200000):

        ### run an episode for x coord
        if ieps % 2 == 0:
            sse = SSenvReal('x')
            sse.max_epLength = max_epLength
            epbf, rew = run_episode(sse, mainQN, mainMN, sess, 'x')
            bufferArray = np.array(epbf)
            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            rList.append(rew)

        else:
            ### run an episode for Y coord
            sse = SSenvReal('y')
            sse.max_epLength = max_epLength
            epbf, rew = run_episode(sse, mainQNY, mainMNY, sess, 'y')
            bufferArray = np.array(epbf)
            episodeBuffer = list(zip(bufferArray))
            myBufferY.add(episodeBuffer)
            rListY.append(rew)

        sse.index = sse.index + 1
        total_steps += 16
        total_eps += 1

        if total_eps % 5 == 0:
            updateTarget(targetOps, sess)
            updateTarget(targetOpsM, sess)
            updateTarget(targetOpsY, sess)
            updateTarget(targetOpsMY, sess)
            # print('target updated')

        if total_eps > 20:
            if total_eps % 2 == 0:
                mainQN, mainMN = updateQandM(myBuffer, mainQN, mainMN)
                mainQNY, mainMNY = updateQandM(myBufferY, mainQNY, mainMNY)
                # print('model updated')

        if (len(rList) % summaryLength) == 0 and len(rList) != 0:
            print(total_steps, np.mean(rList[-summaryLength:]))
            print(total_steps, np.mean(rListY[-summaryLength:]))
