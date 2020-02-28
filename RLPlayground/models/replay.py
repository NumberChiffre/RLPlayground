import random
from collections import deque
from typing import List, Dict, Tuple
import torch
import numpy as np

from RLPlayground.utils.data_structures import Transition
from RLPlayground.utils.registration import Registrable


class Replay(Registrable):
    @classmethod
    def build(cls, type: str, params: Dict):
        replay = cls.by_name(type)
        return replay.from_params(params)

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError(
            f'from_params not implemented in {cls.__class__.name}')


# TODO: initial version, not optimized with tree structures used in paper
@Replay.register('ExperienceReplay')
class ExperienceReplay(Replay, Registrable):
    def __init__(self,
                 capacity: int,
                 n_step: int,
                 gamma: float):
        """

        :param capacity: maximum number of transition tuple stored in replay
        :param n_step: n step used for replay
        :param gamma: discount factor when computing td-error
        """
        self.replay_type = self.__class__.__name__
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.n_step = n_step
        if self.n_step > 0:
            self.n_step_memory = deque(maxlen=self.n_step)
            self.gamma = gamma

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def push(self, transition: Transition):
        """push Transition into memory for batch sampling and n_step
        computations"""
        if self.n_step > 0:
            self.n_step_memory.append(transition)
            if len(self.n_step_memory) == self.n_step:
                transition = self.generate_n_step_q()
        self.memory.append(transition)

    def generate_n_step_q(self) -> Transition:
        """with s(t), s(t+1), calculate a discounted reward by backtracking
        n_steps prior to t and setting s(t) to s(t-n_steps)"""
        transitions = self.n_step_memory
        reward, next_observation, done = transitions[-1][-3:]
        for i in range(len(transitions) - 1):
            reward = self.gamma * reward * (1 - transitions[i].done) + \
                     transitions[i].r
            next_observation, done = (transitions[i].s1, transitions[i].done) \
                if transitions[i].done else (next_observation, done)
        observation, action = transitions[0][:2]
        return Transition(s0=observation, a=action, r=reward,
                          s1=next_observation, done=done)

    def sample(self, batch_size: int) -> List[Transition]:
        """Uniform sampling with a batch from memory and concatenates the
        dimensions of observations and convert to torch"""
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


@Replay.register('PrioritizedExperienceReplay')
class PrioritizedExperienceReplay(ExperienceReplay, Registrable):
    def __init__(self,
                 capacity: int,
                 n_step: int,
                 alpha: float,
                 beta: float,
                 beta_inc: float,
                 gamma: float,
                 non_zero_variant: float):

        """

        :param capacity: maximum number of transition tuple stored in replay
        :param n_step: n step used for replay
        :param gamma: discount rate for calculating td-error
        :param alpha: 0 for no prioritization, 1 for full prioritization
        :param beta:
        :param beta_inc:
        :param non_zero_variant: small constant to ensure non-zero probabilities
        """
        # try with alpha=0.6, beta=0.4, beta_inc=100~network update frequency
        super().__init__(capacity=capacity, n_step=n_step, gamma=gamma)
        assert alpha + beta == 1.0
        self.alpha = alpha
        self.beta = beta
        self.beta_inc = (1 - beta) / beta_inc
        self.non_zero_variant = non_zero_variant
        self.priorities = np.zeros([self.capacity], dtype=np.float32)
        self.idx = 0
        self.memory = []

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def push(self, transition: Transition):
        max_prior = np.max(self.priorities) if self.memory else 1.0

        # n_step computation
        if self.n_step > 0:
            self.n_step_memory.append(transition)
            if len(self.n_step_memory) == self.n_step:
                transition = self.generate_n_step_q()

        # unlike ExperienceReplay which updates based on FIFO
        # update from start of queue
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.idx] = transition
        self.priorities[self.idx] = max_prior
        self.idx += 1
        self.idx = self.idx % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.array]:
        """use absolute td-error to favor model to optimize"""
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
        # probs = abs(td-error), use probabilities
        probs = (probs ** self.alpha) / np.sum(probs ** self.alpha)
        self.indices = np.random.choice(len(self.memory), batch_size, p=probs)
        if self.beta < 1:
            self.beta += self.beta_inc

        # samples a batch from memory and concatenates the dimensions of
        # observations and convert to torch
        batch = [self.memory[idx] for idx in self.indices]
        observation, action, reward, next_observation, done = zip(*batch)
        observation = torch.cat(tuple(torch.FloatTensor(observation)), dim=0)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.cat(tuple(torch.FloatTensor(next_observation)),
                                     dim=0)
        done = torch.FloatTensor(done)

        # need weights to compute loss
        weights = (len(self.memory) * probs[self.indices]) ** -self.beta
        weights = np.array(weights / np.max(weights), dtype=np.float32)
        return Transition(s0=observation, a=action, r=reward,
                          s1=next_observation, done=done), weights

    def update_priorities(self, losses: np.array):
        """update absolute td-error to compute probabilities"""
        for idx, priority in zip(self.indices, losses):
            self.priorities[idx] = priority
