import torch
import numpy as np
from typing import Dict, Generator, List, Tuple, Union
from gym import Env

from RLPlayground.agents.agent import Agent
from RLPlayground.utils.data_structures import Transition
from RLPlayground.utils.registration import Registrable


@Agent.register('SMDPQLearningAgent')
class SMDPQLearningAgent(Agent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.beta = agent_cfg['beta']
        self.alpha = agent_cfg['alpha']
        self.options = agent_cfg['options']
        self.Q = np.zeros(
            (len(self.env.observation_space), self.env.action_space.n))

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    @torch.no_grad()
    def get_action(self, s0) -> Union[int, Tuple]:
        pass

    def train(self, transition: List[Transition]):
        """
        :param transition: (s0, a, r, s1, done) where s0/s1 are old/new s0s
        """
        self.Q[transition.s0][transition.a] += self.alpha * (
                transition.r + self.gamma * np.max(self.Q[transition.s1]) -
                self.Q[transition.s0][transition.a])
        return self.get_action(s0=transition.s1)

    def learn(self, num_steps: int) -> Generator:
        done = False
        cr, t = 0, 0
        s0 = self.env.reset()
        action = self.get_action(s0=s0)
        while not done and t < num_steps:
            if action >= self.env.action_space.n:
                s1, r, done, info, steps = self.learn_option(s0=s0,
                                                             action=action)
                t += steps
            else:
                s1, r, done, info = self.env.step(action)
                t += 1
            cr += r
            transition = Transition(s0=s0, a=action, r=r, s1=s1, done=done)
            if not done:
                action = self.train(transition)
                s0 = s1
        yield {
            'cum_reward': cr,
            'time_to_solve': t
        }

    def learn_option(self, s0: np.ndarray, action: int):
        done = False
        cr, t = 0, 0
        option = self.options[action - self.env.action_space.n]
        terminated = False

        while not terminated:
            action, terminated = option.step(s0)
            s1, r, done, info = self.env.step(action)  # taking action
            cr += r * (self.gamma ** t)
            t += 1
            if done:
                break
            s0 = s1
        return s0, cr, done, None, t
