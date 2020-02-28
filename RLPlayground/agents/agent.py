from gym import Env
from typing import Dict
from RLPlayground.utils.registration import Registrable


class Agent(Registrable):
    @classmethod
    def build(cls, type: str, env: Env, params: Dict):
        agent = cls.by_name(type)
        return agent.from_params(env, params)

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        raise NotImplementedError(
            f'from_params not implemented in {cls.__class__.name}')
