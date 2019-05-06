

class Qnetwork:

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # def __init__(self, h_size, nStates, nActions, rnn_cell, myScope):
    def __init__(self, sess, config):

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