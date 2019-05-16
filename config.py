class Config:

    # default setting
    def __init__(self):

        # set by command line args
        self.agent_type = None  # 'dan', 'dan_coverage', 'randomAction', 'coverage'
        self.random_seed = None

        # env params
        self.total_train_steps = 60000  # 60000  # 3000 episode
        self.agent_pre_train_steps = 600  # 100 episodes

        self.test_ep_num = 50  # total 100
        self.test_interval = 600  # every 50 episodes

        self.buffer_size = 1000000  # 1mil episodes
        self.update_reward = False
        self.agent_update_freq = 1

        self.max_ep_length = 12  # The max allowed length of our episode.
        self.batch_size = 8
        self.trace_length = 8

        self.fc_size1 = 60
        self.fc_size2 = 30
        # agent params
        # self.epsilon = 0.3

        # self.qnet_lr = 0.001
        # self.mnet_lr = 0.001
        # self.h_size = 512  # The size of the final recurrent layer before splitting it into Advantage and Value streams.
        self.tau = 0.01
        self.gamma = 0.99

        self.nStates = 151
        self.nActions = 10
        self.print_ep_freq = 50

    # add custom setting
    def merge_config(self, custom_config):
        for key in custom_config.keys():
            setattr(self, key, custom_config[key])

#### Parameters
# batch_size = 4  # How many experience traces to use for each training step.
# trace_length = 4  # How long each experience trace will be when training
# update_freq = 2   # How often to perform a training step.
# update_target = 20
# y = .99  # Discount factor on the target Q-values
# startE = 1  # Starting chance of random action
# endE = 0.1  # Final chance of random action
# anneling_steps = 50000  # How many steps of training to reduce startE to endE.
# num_episodes = 1000000  # How many episodes of game environment to train network with.
# pre_train_steps = 5000  # How many steps of random actions before training begins.
# load_model = False  # Whether to load a saved model.
# path = "./drqn"  # The path to save our model to.
# h_size = 512  # The size of the final recurrent layer before splitting it into Advantage and Value streams.
# max_epLength = 16  # The max allowed length of our episode.
# time_per_step = 1  # Length of each step used in gif creation
# summaryLength = 25  # Number of epidoes to periodically save for analysis
# tau = 0.001
