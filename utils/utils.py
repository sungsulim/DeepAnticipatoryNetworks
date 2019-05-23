import numpy as np
import random
from collections import deque


class ExperienceBuffer_Episode:
    def __init__(self, buffer_size, seed):
        # self.rng = np.random.RandomState(seed)
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        # idx = self.rng.choice(len(self.buffer), batch_size)
        # sampled_episodes = [self.buffer[i] for i in idx]

        sampledTraces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces.append(episode[point:point + trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces, [batch_size * trace_length, 6])


class ExperienceBuffer:
    def __init__(self, buffer_size, seed):
        # self.rng = np.random.RandomState(seed)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, experience):
        # experience: tuple (s, a, r, s', label, done) of size 6
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
