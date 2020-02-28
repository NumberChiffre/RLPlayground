from gym import Env
from typing import Dict


class Agent:
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict,
                 *args,
                 **kwargs):
        """

        :param env: gym environment
        :param agent_cfg: dictionary containing params/hyperparams for agent
        :param args:
        :param kwargs:
        """
        self.env = env
        self.agent_cfg = agent_cfg

    def train(self):
        raise NotImplementedError("Agent's training method not implemented!")

    def interact(self):
        raise NotImplementedError("Agent's interact method not implemented!")
