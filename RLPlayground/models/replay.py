import random
from collections import deque
from typing import List, Tuple
import torch
import numpy as np

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
    def __init__(self, capacity: int, n_step: int, alpha: float, beta: float,
                 beta_inc: float):
        """

        :param capacity: maximum number of transition tuple stored in replay
        :param n_step: n step used for replay
        :param alpha: 0 for no prioritization, 1 for full prioritization
        :param beta:
        :param beta_inc:
        """
        # try with alpha=0.6, beta=0.4, beta_inc=100~network update frequency
        super().__init__(capacity=capacity, n_step=n_step)
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = beta_inc
        self.priorities = np.zeros([capacity], dtype=np.float32)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.array]:
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
        probs = (probs ** self.alpha) / np.sum(probs ** self.alpha)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        if self.beta < 1:
            self.beta += self.beta_inc

        # samples a batch from memory and concatenates the dimensions of
        # observations and convert to torch
        # cannot use super().sample as we need the indices
        batch = [self.memory[idx] for idx in indices]
        observation, action, reward, next_observation, done = zip(*batch)
        observation = torch.cat(tuple(torch.FloatTensor(observation)), dim=0)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.cat(tuple(torch.FloatTensor(next_observation)),
                                     dim=0)
        done = torch.FloatTensor(done)

        # update priorities
        for idx, priority in zip(indices, self.priorities):
            self.priorities[idx] = priority

        # need weights to compute MSE
        weights = (len(self.memory) * probs[indices]) ** -self.beta
        weights = np.array(weights / np.max(weights), dtype=np.float32)
        return Transition(s0=observation, a=action, r=reward,
                          s1=next_observation, done=done), weights
