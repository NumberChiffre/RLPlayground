import torch
import numpy as np
from typing import Dict, List, Tuple, Union
from collections import deque, defaultdict
from gym import Env

from RLPlayground.agents.agent import Agent
from RLPlayground.utils.data_structures import RLAlgorithm, Transition
from RLPlayground.utils.utils import nested_d
from RLPlayground.utils.registration import Registrable


@Agent.register('MonteCarloAgent')
class MonteCarloAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.eps = agent_cfg['eps']
        self.occurence = agent_cfg['occurence']
        self.alpha = agent_cfg['alpha']
        self.Q = np.zeros(
            (len(self.env.observation_space), self.env.action_space.n))

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, **params)

    def get_action(self, observation: Transition):
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[observation])

    def train(self):
        pass

    def interact(self, num_steps: int):
        done = False
        cr, t = 0, 0
        observation = self.env.reset()
        action = self.get_action(observation=observation)
        transitions = []
        rewards, counts = defaultdict(nested_d), defaultdict(nested_d)
        while not done and t < num_steps:
            next_observation, reward, done, info = self.env.step(action)
            transitions.append(Transition(s0=observation, a=action, r=reward,
                                          s1=next_observation, done=done))
            if not done:
                action = self.get_action(observation=next_observation)
            observation = next_observation

        # TODO: include first-visit
        for i, transition in enumerate(transitions):
            idx = i
            G = sum([self.gamma ** i * x.r for i, x in
                     enumerate(transitions[idx:])])
            self.Q[transition.s0][transition.a] += self.alpha * (
                        G - self.Q[transition.s0][transition.a])
            cr += transition.r
        yield {
            'cum_reward': cr,
            'time_to_solve': t
        }


@Agent.register('TDAgent')
class TDAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.eps = agent_cfg['eps']
        self.gamma = agent_cfg['gamma']
        self.alpha = agent_cfg['alpha']
        self.n_step = agent_cfg['n_step']
        self.Q = np.zeros(
            (len(self.env.observation_space), self.env.action_space.n))

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, **params)

    def get_action(self, observation: Transition):
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[observation])

    def train(self, action: Union[int, str], state: np.array):
        pass

    def generate_n_step(self, t: int, num_steps: int, exp_tuple: deque):
        tau = t - self.n_step + 1
        if tau >= 0:
            G = np.sum(
                [self.gamma ** (i - tau - 1) * exp_tuple[i].r for i
                 in range(tau + 1, min(tau + self.n_step, num_steps))])

            if tau + self.n_step < num_steps:
                idx = tau + self.n_step - 1
                idx_exp = exp_tuple[idx]
                tau_exp = exp_tuple[tau]
                G = G + self.gamma ** self.n_step * self.Q[idx_exp.s1]
                self.Q[tau_exp.s1] += self.alpha * (G - self.Q[tau_exp.s1])

    def interact(self, num_steps: int):
        done = False
        cr, t = 0, 0
        observation = self.env.reset()
        action = self.get_action(observation=observation)
        exp_tuple = deque(maxlen=self.n_step)
        while not done and t < num_steps:
            next_observation, reward, done, info = self.env.step(action)
            cr += reward
            self.Q[observation] += self.alpha * (
                    reward + self.gamma * self.Q[next_observation] - self.Q[
                observation])
            exp_tuple.append(Transition(s0=observation, a=action, r=reward,
                                        s1=next_observation, done=done))
            if self.n_step > 0:
                self.generate_n_step(t=t, exp_tuple=exp_tuple)
            if not done:
                self.get_action(observation=next_observation)
                observation = next_observation
        yield {
            'cum_reward': cr,
            'time_to_solve': t
        }


@Agent.register('QLearningAgent')
class QLearningAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.eps = agent_cfg['eps']
        self.gamma = agent_cfg['gamma']
        self.alpha = agent_cfg['alpha']
        self.Q = np.zeros(
            (len(self.env.observation_space), self.env.action_space.n))

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, **params)

    def get_action(self, observation: Transition):
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[observation])

    def train(self, action: Union[int, str], state: np.array):
        pass

    def generate_n_step(self, t: int, num_steps: int, exp_tuple: deque):
        tau = t - self.n_step + 1
        if tau >= 0:
            G = np.sum(
                [self.gamma ** (i - tau - 1) * exp_tuple[i].r for i
                 in range(tau + 1, min(tau + self.n_step, num_steps))])

            if tau + self.n_step < num_steps:
                idx = tau + self.n_step - 1
                idx_exp = exp_tuple[idx]
                tau_exp = exp_tuple[tau]
                G = G + self.gamma ** self.n_step * np.max(
                    self.Q[idx_exp.s1])

                self.Q[tau_exp.s1][tau_exp.action] += self.alpha * (
                        G - self.Q[tau_exp.s1][tau_exp.action])

    def interact(self, num_steps: int):
        done = False
        cr, t = 0, 0
        observation = self.env.reset()
        action = self.get_action(observation=observation)
        exp_tuple = deque(maxlen=self.n_step)
        while not done and t < num_steps:
            next_observation, reward, done, info = self.env.step(action)
            cr += reward
            self.Q[observation][action] += self.alpha * (
                    reward + self.gamma * np.max(self.Q[next_observation]) -
                    self.Q[observation][action])
            exp_tuple.append(Transition(s0=observation, a=action, r=reward,
                                        s1=next_observation, done=done))
            if self.n_step > 0:
                self.generate_n_step(t=t, exp_tuple=exp_tuple)
            if not done:
                self.get_action(observation=next_observation)
                observation = next_observation

        yield {
            'cum_reward': cr,
            'time_to_solve': t
        }


@Agent.register('SarsaAgent')
class SarsaAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.eps = agent_cfg['eps']
        self.gamma = agent_cfg['gamma']
        self.alpha = agent_cfg['alpha']
        self.Q = np.zeros(
            (len(self.env.observation_space), self.env.action_space.n))

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, **params)

    def get_action(self, observation: Transition):
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[observation])

    def train(self, action: Union[int, str], state: np.array):
        pass

    def generate_n_step(self, t: int, num_steps: int, exp_tuple: deque):
        tau = t - self.n_step + 1
        if tau >= 0:
            G = np.sum(
                [self.gamma ** (i - tau - 1) * exp_tuple[i].r for i
                 in range(tau + 1, min(tau + self.n_step, num_steps))])

            if tau + self.n_step < num_steps:
                idx = tau + self.n_step - 1
                idx_exp = exp_tuple[idx]
                tau_exp = exp_tuple[tau]
                G = G + self.gamma ** self.n_step * \
                    self.Q[idx_exp.s1][idx_exp.action]

                self.Q[tau_exp.s1][tau_exp.action] += self.alpha * (
                        G - self.Q[tau_exp.s1][tau_exp.action])

    def interact(self, num_steps: int):
        done = False
        cr, t = 0, 0
        observation = self.env.reset()
        action = self.get_action(observation=observation)
        exp_tuple = deque(maxlen=self.n_step)
        while not done and t < num_steps:
            next_observation, reward, done, info = self.env.step(action)
            cr += reward
            if not done:
                next_action = self.get_action(observation=next_observation)
                self.Q[observation][action] += self.alpha * (
                        reward + self.gamma * self.Q[next_observation][
                    next_action] - self.Q[observation][action])
                exp_tuple.append(Transition(s0=observation, a=action, r=reward,
                                            s1=next_observation, done=done))
                if self.n_step > 0:
                    self.generate_n_step(t=t, exp_tuple=exp_tuple)
                observation = next_observation
                action = next_action

        yield {
            'cum_reward': cr,
            'time_to_solve': t
        }


@Agent.register('ExpectedSarsaAgent')
class ExpectedSarsaAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.eps = agent_cfg['eps']
        self.gamma = agent_cfg['gamma']
        self.alpha = agent_cfg['alpha']
        self.Q = np.zeros(
            (len(self.env.observation_space), self.env.action_space.n))

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, **params)

    def get_action(self, observation: Transition):
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            return np.random.randint(self.env.action_space.n)
        else:
            return np.argmax(self.Q[observation])

    def generate_probs(self, action: Union[int, str]):
        probs = [self.eps / self.env.action_space.n] * self.env.action_space.n
        probs[action] = 1 - self.eps + self.eps / self.env.action_space.n
        return probs

    def train(self, action: Union[int, str], state: np.array):
        pass

    def generate_n_step(self, t: int, num_steps: int, exp_tuple: deque):
        tau = t - self.n_step + 1
        if tau >= 0:
            G = np.sum(
                [self.gamma ** (i - tau - 1) * exp_tuple[i].r for i
                 in range(tau + 1, min(tau + self.n_step, num_steps))])

            if tau + self.n_step < num_steps:
                idx = tau + self.n_step - 1
                idx_exp = exp_tuple[idx]
                tau_exp = exp_tuple[tau]

                G = G + self.gamma ** self.n_step * np.sum(
                    [self.Q[idx_exp.s1][a] for a in
                     self.env.action_space])
                self.Q[tau_exp.s1][tau_exp.action] += self.alpha * (
                        G - self.Q[tau_exp.s1][tau_exp.action])

    def interact(self, num_steps: int):
        done = False
        cr, t = 0, 0
        observation = self.env.reset()
        action = self.get_action(observation=observation)
        exp_tuple = deque(maxlen=self.n_step)
        while not done and t < num_steps:
            next_observation, reward, done, info = self.env.step(action)
            cr += reward
            # TODO: don't we have this calculated in the gym environments?...
            probs = self.generate_probs(action=action)
            self.Q[observation][action] += self.alpha * (
                    reward + self.gamma * np.sum(
                [self.Q[next_observation][a] * probs[a] for a in
                 self.env.action_space]) - self.Q[observation][action])
            exp_tuple.append(Transition(s0=observation, a=action, r=reward,
                                        s1=next_observation, done=done))
            if self.n_step > 0:
                self.generate_n_step(t=t, exp_tuple=exp_tuple)
            if not done:
                self.get_action(observation=next_observation)
                observation = next_observation

        yield {
            'cum_reward': cr,
            'time_to_solve': t
        }
