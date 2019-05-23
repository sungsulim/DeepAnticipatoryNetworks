class Config:

    # default setting
    def __init__(self):

        # set by command line args
        # self.agent_type = 'dan', randomAction
        self.random_seed = None

        # env params
        self.total_train_steps = 60000  # 100 epochs
        self.agent_pre_train_steps = 60  # 0.1 epoch

        self.test_ep_num = 30
        self.test_interval = 600  # every 150 samples (1 epoch)

        self.agent_update_freq = 1

        self.buffer_size = 1000000  # 1mil episodes
        self.batch_size = 4

        self.fc_size1 = 32

        self.qnet_lr = 0.01
        self.mnet_lr = 0.01

        self.epsilon = 0.1

        self.tau = 0.1
        self.gamma = 0.99

        self.print_ep_freq = 1

    # add custom setting
    def merge_config(self, custom_config):
        for key in custom_config.keys():
            setattr(self, key, custom_config[key])
