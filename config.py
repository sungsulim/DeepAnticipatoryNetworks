class Config:

    # default setting
    def __init__(self):

        # set by command line args
        self.agent_type = None  # 'normal', 'randomAction', 'coverage'
        self.random_seed = None

        # experiment params
        self.total_train_steps = 160000  # 5000 episode
        self.test_ep_num = 500
        self.test_interval = 1600  # every 100 episodes

        self.agent_pre_train_steps = 1600  # 100 episodes
        self.agent_update_freq = 4

        # env params
        self.max_ep_length = 12  # The max allowed length of our episode.
        self.nStates = 21
        self.nActions = 10

        # agent params
        self.qnet_lr = 0.0001
        self.mnet_lr = 0.0001

        self.h_size = 128  # The size of the final recurrent layer before splitting it into Advantage and Value streams.

        self.batch_size = 16
        self.trace_length = 8
        self.buffer_size = 1000000  # 1mil episodes

        self.epsilon = 0.1
        self.gamma = 0.99
        self.tau = 0.001

        self.print_ep_freq = 10

        # self.startE = 1.0
        # self.endE = 0.1
        # self.annealing_steps = 50000

        # self.load_model = False  # Whether to load a saved model.
        # self.path = "./drqn"  # The path to save our model to.

        # self.time_per_step = 1  # Length of each step used in gif creation
        # self.summaryLength = 25  # Number of epidoes to periodically save for analysis

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