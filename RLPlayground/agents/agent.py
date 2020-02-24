from typing import Dict
from RLPlayground.utils.registration import Registrable


class Agent:
    def __init__(self,
                 env,
                 agent_cfg: Dict,
                 *args,
                 **kwargs):
        # super().__init__(*args, **kwargs)
        self.env = env
        self.agent_cfg = agent_cfg

    def train(self):
        raise NotImplementedError("Agent's training method not implemented!")

    def interact(self):
        raise NotImplementedError("Agent's interact method not implemented!")
