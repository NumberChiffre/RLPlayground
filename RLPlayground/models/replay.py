from RLPlayground.utils.data_structures import Experience
from random import random


class ReplayBuffer:
    def __init__(self, capacity: int):
        """

        :param capacity: maximum number of experience tuple stored in replay
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, experience: Experience):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float):
        """

        :param capacity: maximum number of experience tuple stored in replay
        :param alpha: 0 for no prioritization, 1 for full prioritization
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity=capacity)
        self.alpha = alpha

    def push(self, *args, **kwargs):
        super().push(*args)

    def _sample_proportional(self, batch_size: int):
        pass
