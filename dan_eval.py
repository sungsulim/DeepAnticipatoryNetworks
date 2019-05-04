import numpy as np

def get_eval_action(mainQNX, mainQNY, state, sess, prev_obs):
    if len(prev_obs) == 1:
        Qx, stateQx = sess.run([mainQNX.Qout, mainQNX.rnn_state], feed_dict={mainQNX.scalarInput: prev_obs, mainQNX.trainLength: 1,
                                                                             mainQNX.state_in: state, mainQNX.batch_size: 1})
        Qy, stateQy = sess.run([mainQNY.Qout, mainQNY.rnn_state],
                               feed_dict={mainQNY.scalarInput: prev_obs, mainQNY.trainLength: 1,
                                          mainQNY.state_in: state, mainQNY.batch_size: 1})
        Q = Qx + Qy
        a = np.argmax(Q, 1)
        a = a[0]
        return a, stateQx, stateQy


def run_evaluation_episode(mainQN, mainMN, mainQNY, mainMNY, sess):
    sse1 = SSenvReal()
    sse2 = SSenvReal()

    prev_obsx = np.reshape([0] * (sse.nStates + sse.nActions), [1, 31])
    prev_obsy = np.reshape([0] * (sse.nStates + sse.nActions), [1, 31])
    state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
    stateQx = state
    stateQy = state
    stateMx = state
    stateMy = state
    episodeBuffer = []
    d = False
    rew = 0
    for i in range(0, sse.max_epLength):
        a, stateQx, stateQy = get_Qaction(mainQN, stateQx, sess, prev_obsx)
        a_onehot = np.reshape(np.array([int(i == a) for i in range(sse.nActions)]), [1, sse.nActions])

        ns = (sse.get_cell(sse.get_next_state(sse.track, sse.index)))
        ns_cell = sse.discretize_cell(ns)
        x_cell = ns_cell[0]
        y_cell = ns_cell[1]

        obsx = np.reshape(sse.get_obsX(ns, a), [1, 1])
        curr_obsx = format_obs(obsx, sse.nStates, a_onehot)
        prediction, stateMx = sess.run([mainMN.prediction, mainMN.rnn_state],
                                       feed_dict={mainMN.scalarInput: curr_obsx, mainMN.trainLength: 1,
                                                  mainMN.state_in: stateMx, mainMN.batch_size: 1})
        rx = get_pred_reward(prediction, x_cell)

        obsy = np.reshape(sse.get_obsY(ns, a), [1, 1])
        curr_obsy = format_obs(obsy, sse.nStates, a_onehot)
        prediction, stateMy = sess.run([mainMN.prediction, mainMN.rnn_state],
                                       feed_dict={mainMN.scalarInput: curr_obsy, mainMN.trainLength: 1,
                                                  mainMN.state_in: stateMy, mainMN.batch_size: 1})
        ry = get_pred_reward(prediction, x_cell)

        rew = rew + rx + ry
        sse1.index = sse1.index + 1
        prev_obsx = curr_obsx
        prev_obsy = curr_obsy
    return rew

# Here the idea is to use the trained QX, MX, QY and MY nets to run an evaluation on