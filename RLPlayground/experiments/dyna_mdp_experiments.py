import ray
import gym
import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List

from RLPlayground import RESULT_DIR
from RLPlayground.agents.dyna_mdp import MDPDynaAgent
from RLPlayground.utils.logger import ProjectLogger


# gotta put more keys into defaultdict + able to pickle..
def foo1():
    return defaultdict(dict)


def foo2():
    return defaultdict(int)


def foo():
    return defaultdict(foo1)


class Experimenting:
    def __init__(self,
                 logger: ProjectLogger,
                 experiment_cfg: Dict,
                 agent_cfg: Dict,
                 env_names: List,
                 dp_methods: List,
                 params_vals: List,
                 seeds: List):
        self.logger = logger
        self.experiment_cfg = experiment_cfg
        self.agent_cfg = agent_cfg
        self.env_names = env_names
        self.dp_methods = dp_methods
        self.params_vals = params_vals
        self.seeds = seeds

    def run_dp(self):
        output = defaultdict(foo)

        # go for each environment
        for env_name in self.env_names:
            # generate results for policy iteration
            for dp_method in self.dp_methods:
                results = [
                    self._inner_dp_experiment.remote(self=self,
                                                     env_name=env_name,
                                                     seed=seed,
                                                     dp_method=dp_method)
                    for seed in self.seeds]
                results = ray.get(results)

                # results over the seeds
                output[env_name][dp_method]['train'][
                    'mean_cum_rewards'] = np.mean(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['train'][
                    'var_cum_rewards'] = np.var(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['train']['upper_var_cum_rewards'] = \
                    output[env_name][dp_method]['train']['mean_cum_rewards'] + \
                    output[env_name][dp_method]['train']['var_cum_rewards']
                output[env_name][dp_method]['train']['lower_var_cum_rewards'] = \
                    output[env_name][dp_method]['train']['mean_cum_rewards'] - \
                    output[env_name][dp_method]['train']['var_cum_rewards']
                output[env_name][dp_method]['train'][
                    'max_cum_rewards'] = np.max(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)

                output[env_name][dp_method]['test'][
                    'mean_cum_rewards'] = np.mean(
                    np.vstack([results[i][1] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['test']['var_cum_rewards'] = np.var(
                    np.vstack([results[i][1] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['test']['upper_var_cum_rewards'] = \
                    output[env_name][dp_method]['test']['mean_cum_rewards'] + \
                    output[env_name][dp_method]['test']['var_cum_rewards']

                output[env_name][dp_method]['test']['lower_var_cum_rewards'] = \
                    output[env_name][dp_method]['test']['mean_cum_rewards'] - \
                    output[env_name][dp_method]['test']['var_cum_rewards']

                output[env_name][dp_method]['test']['max_cum_rewards'] = np.max(
                    np.vstack([results[i][1] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['train'][
                    'mean_timesteps'] = np.mean(
                    np.vstack([results[i][2] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['train']['min_timesteps'] = np.min(
                    np.vstack([results[i][2] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['test']['mean_timesteps'] = np.mean(
                    np.vstack([results[i][3] for i in range(len(results))]),
                    axis=0)
                output[env_name][dp_method]['test']['min_timesteps'] = np.min(
                    np.vstack([results[i][3] for i in range(len(results))]),
                    axis=0)

                # save onto disk
                with open(f'{RESULT_DIR}/dyna_mdp_experiments.pickle',
                          'wb') as file:
                    pickle.dump(output, file)
                print(
                    f'Finished running experiments for {env_name} | {dp_method}')
                self.logger.info(
                    f'Finished running experiments for {env_name} | {dp_method}')
        return output

    @ray.remote
    def _inner_dp_experiment(self, env_name: str, seed: int = 1,
                             dp_method: str = 'policy_iteration'):
        # random seed init
        np.random.seed(seed)

        # create environment and agent
        env = gym.make(env_name)
        dp_agent = MDPDynaAgent(env=env,
                                agent_cfg=self.agent_cfg[env_name][dp_method])

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

        # init policy/value functions
        opt_policy, opt_value_func = None, None

        # O(runs * episodes * max(test_rng * steps, steps))
        for r in range(self.experiment_cfg['runs']):
            for i_episode in range(self.experiment_cfg['episodes']):
                # train policy/value function with 1 step
                opt_policy, opt_value_func = dp_agent.train(dp_method=dp_method,
                                                            num_steps=1,
                                                            value_policy=[
                                                                opt_policy,
                                                                opt_value_func])

                # for every 10th episode, lock optimal policy update for testing
                if i_episode % self.experiment_cfg['train_rng'] == 0:
                    avg_cr = list()
                    # get reward for next 5 episodes
                    for test_episode in range(self.experiment_cfg['test_rng']):
                        cr, t = dp_agent.interact(
                            num_steps=self.experiment_cfg['steps'],
                            opt_policy=opt_policy)
                        test_time_to_solve[r, i_episode // 10 - 1] = t
                        avg_cr.append(cr)
                    test_cum_reward[r, i_episode // 10 - 1] = np.mean(avg_cr)

                # interact with environment to get reward based on optimal policy
                cr, t = dp_agent.interact(
                    num_steps=self.experiment_cfg['steps'],
                    opt_policy=opt_policy)
                time_to_solve[r, i_episode] = t
                cum_reward[r, i_episode] = cr
        env.close()
        cum_reward = np.mean(cum_reward, axis=0)
        test_cum_reward = np.mean(test_cum_reward, axis=0)
        return cum_reward, test_cum_reward, time_to_solve, test_time_to_solve

    def evaluate_hyperparameters(self):
        output, best_params = defaultdict(foo), defaultdict(foo)
        best_per_method = defaultdict(foo1)
        results = [self._inner_evaluate_hyperparameters.remote(self=self,
                                                               params=params)
                   for params in self.params_vals]
        results = ray.get(results)

        # list of parallelized results --> reorganized for plotting
        i = 0
        for params in self.params_vals:
            for env_name in self.env_names:
                for dp_method in self.dp_methods:
                    # convert from ray list into readable output
                    output[env_name][dp_method][f'{params[0]}_{params[1]}'] = \
                        results[i][f'{params[0]}_{params[1]}'][env_name][
                            dp_method]

                    # save the mean reward..
                    best_params[env_name][dp_method][
                        f'{params[0]}_{params[1]}'] = \
                        np.mean(
                            output[env_name][dp_method][
                                f'{params[0]}_{params[1]}'])

                    # store onto disk
                    with open(f'{RESULT_DIR}/dyna_mdp_experiments_'
                              f'hyperparameters.pickle', 'wb') as file:
                        pickle.dump(output, file)

                # get the best params per environment and agent
                best_per_method[env_name][dp_method] = max(
                    best_params[env_name][dp_method].items(),
                    key=lambda x: x[1])
                max_params = best_per_method[env_name][dp_method][0].split(
                    '_')
                idx = 0
                for param in self.agent_cfg[env_name][dp_method]:
                    self.agent_cfg[env_name][dp_method][param] = float(
                        max_params[idx])
                    idx += 1
            i += 1
        return self.agent_cfg, output

    @ray.remote
    def _inner_evaluate_hyperparameters(self, params: List):
        output = defaultdict(foo)
        # slow-mo grid search over theta-discount rates
        self.agent_cfg['theta'] = params[0]
        self.agent_cfg['discount_rate'] = params[1]

        # go for each environment
        for env_name in self.env_names:
            # generate results for policy iteration
            for dp_method in self.dp_methods:
                results = [self._inner_dp_experiment.remote(self=self,
                                                            env_name=env_name,
                                                            seed=seed,
                                                            dp_method=dp_method)
                           for seed in self.seeds]
                results = ray.get(results)

                print(
                    f'Start running experiments for {env_name} | {dp_method} '
                    f'| {params}')
                self.logger.info(
                    f'Start running experiments for {env_name} | {dp_method} '
                    f'| {params}')

                # results over the seeds
                output[f'{params[0]}_{params[1]}'][env_name][
                    dp_method] = np.mean(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)
        print(f'Finished experiments for params: {params}')
        self.logger.info(f'Finished experiments for params: {params}')
        return output
