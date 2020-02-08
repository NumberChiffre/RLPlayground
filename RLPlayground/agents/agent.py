from typing import Dict


class Agent:
    def __init__(self,
                 env,
                 agent_cfg: Dict):
        self.env = env
        self.agent_cfg = agent_cfg

    def random_policy(self):
        raise NotImplementedError("Agent's random policy not implemented!")

    def train(self):
        raise NotImplementedError("Agent's training method not implemented!")

