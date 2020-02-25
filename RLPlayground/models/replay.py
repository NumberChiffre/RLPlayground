import random
from collections import deque
from typing import List
import torch

from RLPlayground.utils.data_structures import Transition


class ReplayBuffer:
    def __init__(self, capacity: int, n_step: int):
        """

        :param capacity: maximum number of transition tuple stored in replay
        :param n_step: n step used for replay
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.n_step = n_step
        if n_step > 0:
            self.n_step_memory = deque(maxlen=n_step)

    def push(self, transition: Transition):
        """push Transition into memory for batch sampling and n_step
        computations"""
        if self.n_step > 0:
            self.n_step_memory.append(transition)
        self.memory.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        """samples a batch from memory and concatenates the dimensions of
        observations and convert to torch"""
        batch = random.sample(self.memory, batch_size if len(
            self.memory) > batch_size else len(self.memory))
        observation, action, reward, next_observation, done = zip(*batch)
        observation = torch.cat(tuple(torch.FloatTensor(observation)), dim=0)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.cat(tuple(torch.FloatTensor(next_observation)),
                                     dim=0)
        done = torch.FloatTensor(done)
        return Transition(s0=observation, a=action, r=reward,
                          s1=next_observation, done=done)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float):
        """

        :param capacity: maximum number of transition tuple stored in replay
        :param alpha: 0 for no prioritization, 1 for full prioritization
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity=capacity)
        self.alpha = alpha

    def push(self, *args, **kwargs):
        super().push(*args)

    def _sample_proportional(self, batch_size: int):
        pass
