import numpy as np
from collections import defaultdict
from typing import Dict, List
from RLPlayground.experiments.experiment import Experiment
from RLPlayground.utils.logger import ProjectLogger


class DPExperiment(Experiment):
    def __init__(self,
                 logger: ProjectLogger,
                 *args, **kwargs):
        super().__init__(logger=logger, *args, **kwargs)

    def generate_metrics(self, env_name: str, algo: str, results: List,
                         output: defaultdict) -> defaultdict:
        """generate whatever metrics needed for the experiment over multiple
        seeds"""
        output[env_name][algo]['train'][
            'mean_cum_rewards'] = np.mean(
            np.vstack([results[i][0] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['train'][
            'var_cum_rewards'] = np.var(
            np.vstack([results[i][0] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['train']['upper_var_cum_rewards'] = \
            output[env_name][algo]['train']['mean_cum_rewards'] + \
            output[env_name][algo]['train']['var_cum_rewards']
        output[env_name][algo]['train']['lower_var_cum_rewards'] = \
            output[env_name][algo]['train']['mean_cum_rewards'] - \
            output[env_name][algo]['train']['var_cum_rewards']
        output[env_name][algo]['train'][
            'max_cum_rewards'] = np.max(
            np.vstack([results[i][0] for i in range(len(results))]),
            axis=0)

        output[env_name][algo]['test'][
            'mean_cum_rewards'] = np.mean(
            np.vstack([results[i][1] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['test']['var_cum_rewards'] = np.var(
            np.vstack([results[i][1] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['test']['upper_var_cum_rewards'] = \
            output[env_name][algo]['test']['mean_cum_rewards'] + \
            output[env_name][algo]['test']['var_cum_rewards']

        output[env_name][algo]['test']['lower_var_cum_rewards'] = \
            output[env_name][algo]['test']['mean_cum_rewards'] - \
            output[env_name][algo]['test']['var_cum_rewards']

        output[env_name][algo]['test']['max_cum_rewards'] = np.max(
            np.vstack([results[i][1] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['train'][
            'mean_timesteps'] = np.mean(
            np.vstack([results[i][2] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['train']['min_timesteps'] = np.min(
            np.vstack([results[i][2] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['test']['mean_timesteps'] = np.mean(
            np.vstack([results[i][3] for i in range(len(results))]),
            axis=0)
        output[env_name][algo]['test']['min_timesteps'] = np.min(
            np.vstack([results[i][3] for i in range(len(results))]),
            axis=0)
        return output
