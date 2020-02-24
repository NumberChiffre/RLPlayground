from typing import Dict, List
from RLPlayground.utils.logger import ProjectLogger


class Experiment(object):
    def __init__(self,
                 logger: ProjectLogger,
                 experiment_cfg: Dict,
                 agent_cfg: Dict,
                 env_names: List,
                 algos: List,
                 params_vals: List,
                 seeds: List):
        self.logger = logger
        self.experiment_cfg = experiment_cfg
        self.agent_cfg = agent_cfg
        self.env_names = env_names
        self.algos = algos
        self.params_vals = params_vals
        self.seeds = seeds

    def run(self):
        raise NotImplementedError('Experiment require run method')

    def _inner_run(self):
        raise NotImplementedError('Experiment require inner run method')

    def tune_hyperparams(self):
        raise NotImplementedError('Experiment require tune hyperparams method')

    def _inner_tune_hyperparams(self):
        raise NotImplementedError('Experiment require tune hyperparams method')