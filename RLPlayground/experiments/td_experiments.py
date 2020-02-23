import ray
import gym
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List

from RLPlayground import RESULT_DIR
from RLPlayground.agents.td import TDAgent
from RLPlayground.experiments.experiment import Experiment
from RLPlayground.utils.logger import ProjectLogger
from RLPlayground.utils.data_structures import foo, foo1, foo2


class TDExperiment:
    def __init__(self,
                 logger: ProjectLogger,
                 experiment_cfg: Dict,
                 agent_cfg: Dict,
                 env_names: List,
                 algos: List,
                 params_vals: List,
                 seeds: List):
        # super().__init__(logger=logger,
        #                  experiment_cfg=experiment_cfg,
        #                  agent_cfg=agent_cfg,
        #                  env_names=env_names, algos=algos,
        #                  params_vals=params_vals, seeds=seeds)
        self.logger = logger
        self.experiment_cfg = experiment_cfg
        self.agent_cfg = agent_cfg
        self.env_names = env_names
        self.algos = algos
        self.params_vals = params_vals
        self.seeds = seeds

    def run(self):
        output = defaultdict(foo)

        # go for each environment
        for env_name in self.env_names:
            # generate results for policy iteration
            for algo in self.algos:
                results = [
                    self._inner_run.remote(self=self,
                                           env_name=env_name,
                                           seed=seed,
                                           algo=algo)
                    for seed in self.seeds]
                results = ray.get(results)

                # results over the seeds
                output[env_name][algo]['train'][
                    'mean_cum_rewards'] = np.mean(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)

                # save onto disk
                with open(f'{RESULT_DIR}/dyna_mdp_experiments.pickle',
                          'wb') as file:
                    pickle.dump(output, file)
                print(
                    f'Finished running experiments for {env_name} | {algo}')
                self.logger.info(
                    f'Finished running experiments for {env_name} | {algo}')
        return output

    @ray.remote
    def _inner_run(self, env_name: str, seed: int = 1, algo: str = 'sarsa'):

        # random seed init
        np.random.seed(seed)

        # create environment and agent
        env = gym.make(env_name)
        agent = TDAgent(env=env, agent_cfg=self.agent_cfg[env_name][algo])

        # result initialization
        cum_reward = np.zeros(
            (self.experiment_cfg['runs'], self.experiment_cfg['episodes']))
        time_to_solve = np.ones(
            (self.experiment_cfg['runs'], self.experiment_cfg['episodes'])) * \
                        self.experiment_cfg['steps']
        test_cum_reward = np.zeros((self.experiment_cfg['runs'],
                                    self.experiment_cfg['episodes'] // 10))
        test_time_to_solve = np.ones((self.experiment_cfg['runs'],
                                      self.experiment_cfg['episodes'] // 10)) * \
                             self.experiment_cfg['steps']

        # O(runs * episodes * max(test_rng * steps, steps))
        for r in range(self.experiment_cfg['runs']):
            for i_episode in range(self.experiment_cfg['episodes']):

                # for every 10th episode, lock optimal policy update for testing
                if i_episode % self.experiment_cfg['train_rng'] == 0:
                    avg_cr = list()
                    # get reward for next 5 episodes
                    for test_episode in range(self.experiment_cfg['test_rng']):
                        cr, t = agent.interact(
                            num_steps=self.experiment_cfg['steps'])
                        test_time_to_solve[r, i_episode // 10 - 1] = t
                        avg_cr.append(cr)
                    test_cum_reward[r, i_episode // 10 - 1] = np.mean(avg_cr)

                # interact with environment to get reward based on optimal policy
                cr, t = agent.interact(num_steps=self.experiment_cfg['steps'])
                time_to_solve[r, i_episode] = t
                cum_reward[r, i_episode] = cr
        env.close()
        cum_reward = np.mean(cum_reward, axis=0)
        test_cum_reward = np.mean(test_cum_reward, axis=0)
        return cum_reward, test_cum_reward, time_to_solve, test_time_to_solve
