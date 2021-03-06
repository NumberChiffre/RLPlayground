import ray
import gym
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List

from RLPlayground.agents.agent import Agent
from RLPlayground.experiments.experiment import Experiment
from RLPlayground.utils.logger import ProjectLogger
from RLPlayground.utils.utils import nested_d


@Experiment.register('DPExperiment')
class DPExperiment(Experiment):
    def __init__(self,
                 logger: ProjectLogger,
                 params_vals: List,
                 *args, **kwargs):
        super().__init__(logger=logger, *args, **kwargs)
        self.params_vals = params_vals

    def run(self) -> defaultdict:
        output = defaultdict(nested_d)
        for env_name in self.env_names:
            for agent in self.agents:
                results = [
                    Experiment._inner_run.remote(agent_cfg=self.agent_cfg,
                                                 experiment_cfg=self.experiment_cfg,
                                                 env_name=env_name,
                                                 seed=seed,
                                                 agent=agent)
                    for seed in self.seeds]
                results = ray.get(results)
                output = self.generate_metrics(results=results)
                with open(self.experiment_cfg['experiment_path'], 'wb') as file:
                    pickle.dump(output, file)
                self.logger.info(
                    f'Finished running experiments for {env_name} | {agent}')
        return output

    @staticmethod
    @ray.remote
    def _inner_run(agent_cfg: dict, experiment_cfg: dict, env_name: str,
                   seed: int = 1, agent: str = 'sarsa'):

        # random seed init
        np.random.seed(seed)

        # create environment and agent
        env = gym.make(env_name)
        agent = Agent.build(type=agent, env=env,
                            agent_cfg=agent_cfg[env_name][agent])

        # result initialization
        cum_reward = np.zeros(
            (experiment_cfg['runs'], experiment_cfg['episodes']))
        time_to_solve = np.ones(
            (experiment_cfg['runs'], experiment_cfg['episodes'])) * \
                        experiment_cfg['steps']
        test_cum_reward = np.zeros((experiment_cfg['runs'],
                                    experiment_cfg['episodes'] // 10))
        test_time_to_solve = np.ones((experiment_cfg['runs'],
                                      experiment_cfg['episodes'] // 10)) * \
                             experiment_cfg['steps']

        # O(runs * episodes * max(test_rng * steps, steps))
        for r in range(experiment_cfg['runs']):
            for i_episode in range(experiment_cfg['episodes']):

                # for every 10th episode, lock optimal policy update for testing
                if i_episode % experiment_cfg['train_rng'] == 0:
                    avg_cr = list()
                    # get reward for next 5 episodes
                    for test_episode in range(experiment_cfg['test_rng']):
                        cr, t = agent.learn(
                            num_steps=experiment_cfg['steps'],
                            episode=i_episode)
                        test_time_to_solve[r, i_episode // 10 - 1] = t
                        avg_cr.append(cr)
                    test_cum_reward[r, i_episode // 10 - 1] = np.mean(avg_cr)

                # interact with environment to get reward based on optimal policy
                cr, t = agent.learn(num_steps=experiment_cfg['steps'],
                                       episode=i_episode)
                time_to_solve[r, i_episode] = t
                cum_reward[r, i_episode] = cr
                print(f'episode {i_episode + 1} | cum_reward {cr}')
        env.close()
        cum_reward = np.mean(cum_reward, axis=0)
        test_cum_reward = np.mean(test_cum_reward, axis=0)
        return cum_reward, test_cum_reward, time_to_solve, test_time_to_solve

    def tune_hyperparams(self):
        output, best_params = defaultdict(nested_d), defaultdict(nested_d)
        best_per_method = defaultdict(nested_d)
        results = [self._inner_tune_hyperparams.remote(self=self,
                                                       params=params)
                   for params in self.params_vals]
        results = ray.get(results)

        # list of parallelized results --> reorganized for plotting
        i = 0
        for params in self.params_vals:
            for env_name in self.env_names:
                for algo in self.algos:
                    # convert from ray list into readable output
                    output[env_name][algo][f'{params[0]}_{params[1]}'] = \
                        results[i][f'{params[0]}_{params[1]}'][env_name][
                            algo]

                    # save the mean reward..
                    best_params[env_name][algo][
                        f'{params[0]}_{params[1]}'] = \
                        np.mean(
                            output[env_name][algo][
                                f'{params[0]}_{params[1]}'])

                    # store onto disk
                    with open(self.experiment_cfg['hyperparams_path'],
                              'wb') as file:
                        pickle.dump(output, file)

                # get the best params per environment and agent
                best_per_method[env_name][algo] = max(
                    best_params[env_name][algo].items(),
                    key=lambda x: x[1])
                max_params = best_per_method[env_name][algo][0].split(
                    '_')
                idx = 0
                for param in self.agent_cfg[env_name][algo]:
                    self.agent_cfg[env_name][algo][param] = float(
                        max_params[idx])
                    idx += 1
            i += 1
        return self.agent_cfg, output

    @staticmethod
    @ray.remote
    def _inner_tune_hyperparams(agent_cfg: dict, experiment_cfg: dict,
                                env_names: List, algos: List, params: List,
                                seeds: List) -> defaultdict:
        output = defaultdict(nested_d)
        # slow-mo grid search over theta-discount rates
        agent_cfg['theta'] = params[0]
        agent_cfg['gamma'] = params[1]

        # go for each environment
        for env_name in env_names:
            # generate results for policy iteration
            for algo in algos:
                results = [Experiment._inner_run.remote(agent_cfg=agent_cfg,
                                                        experiment_cfg=experiment_cfg,
                                                        env_name=env_name,
                                                        seed=seed,
                                                        algo=algo)
                           for seed in seeds]
                results = ray.get(results)

                print(
                    f'Start running experiments for {env_name} | {algo} '
                    f'| {params}')

                # results over the seeds
                output[f'{params[0]}_{params[1]}'][env_name][
                    algo] = np.mean(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)
        print(f'Finished experiments for params: {params}')
        return output


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
