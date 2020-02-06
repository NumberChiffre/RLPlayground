import numpy as np
from typing import Dict


class MDPDynaAgent:
    def __init__(self,
                 env,
                 agent_cfg: Dict):
        self.env = env
        self.theta = agent_cfg['theta']
        self.discount_rate = agent_cfg['discount_rate']

    def random_policy(self):
        # generate random policy to evaluate different experiments by seeds
        return np.random.randint(0, self.env.nA, size=self.env.nS)

    def get_action_values(self, s: str, value_func: np.array):
        # given state of the environment, compute action values
        action_values = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            value = 0
            for prob, next_s, reward, done in self.env.P[s][a]:
                value += prob * (reward + self.discount_rate * value_func[next_s])
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

    def policy_iteration(self, num_steps: int = 1, old_policy=None):
        value_func = None
        if old_policy is None:
            old_policy = self.random_policy()
        for i in range(num_steps):
            value_func = self.policy_evaluation(old_policy)
            new_policy = self.policy_improvement(value_func)
            if np.array_equal(new_policy, old_policy):
                break
            old_policy = new_policy
        return old_policy, value_func

    def value_iteration(self, num_steps: int = 1, value_func=None):
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
