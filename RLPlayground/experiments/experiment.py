from typing import Dict, List
from collections import defaultdict
from datetime import datetime

from RLPlayground.utils.logger import ProjectLogger
from RLPlayground.utils.registration import Registrable


class Experiment(Registrable):
    def __init__(self,
                 logger: ProjectLogger,
                 env_names: List,
                 agents: List,
                 seeds: List,
                 experiment_cfg: dict,
                 agent_cfg: dict,
                 *args, **kwargs):
        self.logger = logger
        self.env_names = env_names
        self.agents = agents
        self.seeds = seeds
        self.experiment_cfg = experiment_cfg
        self.agent_cfg = agent_cfg
        self.experiment_cfg['date'] = datetime.today().strftime('%Y-%m-%d')

    @classmethod
    def build(cls, type: str, logger: ProjectLogger, params: Dict):
        experiment = cls.by_name(type)
        return experiment.from_params(logger, params)

    @classmethod
    def from_params(cls, logger: ProjectLogger, params: Dict):
        return cls(logger, **params)

    def generate_metrics(self, results: List) -> defaultdict:
        """generate whatever metrics needed for the experiment"""
        raise NotImplementedError('Experiment must generate metrics!')
