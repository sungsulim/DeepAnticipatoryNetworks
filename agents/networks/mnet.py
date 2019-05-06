

class Mnetwork:

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