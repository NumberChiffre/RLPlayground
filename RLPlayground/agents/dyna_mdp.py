import numpy as np
from typing import Dict, List, Tuple
from gym import Env

from RLPlayground.agents.agent import Agent
from RLPlayground.utils.data_structures import RLAlgorithm


# @Agent.register('dyna_mdp')
class MDPDynaAgent(Agent):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super(MDPDynaAgent, self).__init__(env=env, agent_cfg=agent_cfg)
        self.theta = agent_cfg['theta']
        self.gamma = agent_cfg['gamma']

    def random_policy(self) -> np.array:
        # generate random policy to evaluate different experiments by seeds
        return np.random.randint(0, self.env.nA, size=self.env.nS)

    def train(self, algo: str, num_steps: int, value_policy: List) -> \
            Tuple[np.array, np.array]:
        # value policy = [policy, value_function]
        if algo == RLAlgorithm.VI.value:
            policy, value_func = self.value_iteration(num_steps=num_steps,
                                                      value_func=value_policy[
                                                          1])
        elif algo == RLAlgorithm.PI.value:
            policy, value_func = self.policy_iteration(num_steps=num_steps,
                                                       policy=value_policy[
                                                           0])
        return policy, value_func

    def interact(self, num_steps: int, opt_policy: np.array) -> Tuple[
        float, int]:
        # use agent to interact with environment by making actions based on
        # optimal policy to obtain cumulative rewards
        observation = self.env.reset()
        cr = 0
        for t in range(num_steps):
            action = opt_policy[observation]
            observation, reward, done, info = self.env.step(action)
            cr += reward
            if done:
                return cr, t

    def get_action_values(self, s: str, value_func: np.array) -> np.array:
        # given state of the environment, compute action values
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            value = 0
            for prob, next_s, reward, done in self.env.P[s][a]:
                value += prob * (
                        reward + self.gamma * value_func[next_s])
            action_values[a] = value
        return action_values

    def policy_evaluation(self, policy: np.array) -> np.array:
        # compute V[s] given state and policy[state] by generating action values
        old_value_func = np.zeros(self.env.nS)
        while True:
            new_value_func = np.zeros(self.env.nS)
            for s in range(self.env.nS):
                action_values = self.get_action_values(s, old_value_func)
                new_value_func[s] = action_values[policy[s]]
            if np.max(np.abs(new_value_func - old_value_func)) < self.theta:
                break
            old_value_func = new_value_func
        return new_value_func

    def policy_improvement(self, value_func: np.array) -> np.array:
        # set the integer type for policy in order to use numpy indices
        policy = np.zeros(self.env.nS, dtype=int)
        for s in range(self.env.nS):
            action_values = self.get_action_values(s, value_func)
            policy[s] = np.argmax(action_values)
        return policy

    def policy_iteration(self, num_steps: int = 1,
                         policy: np.array = None) -> Tuple[np.array, np.array]:
        value_func = None
        if policy is None:
            policy = self.random_policy()
        for i in range(num_steps):
            value_func = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(value_func)
            if np.array_equal(new_policy, policy):
                break
            policy = new_policy
        return policy, value_func

    def value_iteration(self, num_steps: int = 1,
                        value_func: np.array = None) -> Tuple[
        np.array, np.array]:
        if value_func is None:
            value_func = np.zeros(self.env.nS)
        for i in range(num_steps):
            delta = 0
            for s in range(self.env.nS):
                v = value_func[s]
                value_func[s] = max(self.get_action_values(s, value_func))
                delta = max(delta, abs(value_func[s] - v))
            if delta < self.theta:
                break
        policy = self.policy_improvement(value_func)
        return policy, value_func
