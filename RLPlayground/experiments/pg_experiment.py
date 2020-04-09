import ray
import gym
import torch
import numpy as np
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from RLPlayground.agents.agent import Agent
from RLPlayground.experiments.experiment import Experiment
from RLPlayground.utils.logger import ProjectLogger
from RLPlayground.utils.utils import nested_d
from RLPlayground.utils.plotter import plot_episodic_results
from RLPlayground.utils.data_structures import Transition

from torch.utils.tensorboard import SummaryWriter


@Experiment.register('PGExperiment')
class PGExperiment(Experiment):
    def __init__(self,
                 logger: ProjectLogger,
                 *args, **kwargs):
        super().__init__(logger=logger, *args, **kwargs)

    @classmethod
    def from_params(cls, logger: ProjectLogger, params: Dict):
        return cls(logger, **params)

    def run(self) -> defaultdict:
        """for each gym environment and RL algorithms, test different replay
        buffer capacity over multiple seeds"""
        output = defaultdict(nested_d)
        for env_name in self.env_names:
            for agent in self.agents:
                results = [PGExperiment._inner_run.remote(
                    agent_cfg=self.agent_cfg,
                    experiment_cfg=self.experiment_cfg,
                    env_name=env_name, seed=seed, agent_name=agent) for seed
                    in self.seeds]
                results = ray.get(results)
                output = self.generate_metrics(env_name=env_name,
                                               agent=agent,
                                               results=results,
                                               output=output)
                with open(self.experiment_cfg['experiment_path'],
                          'wb') as file:
                    pickle.dump(output, file)
            self.logger.info(
                f'Finished running experiments for {env_name} | {agent}')
        return output

    @staticmethod
    @ray.remote
    def _inner_run(agent_cfg: dict, experiment_cfg: dict,
                   env_name: str, seed: int = 1, agent_name: str = 'sarsa') -> \
            Tuple[np.array, np.array]:

        # seed and result initialization
        np.random.seed(seed)
        episode_time = np.zeros(
            (experiment_cfg['runs'], experiment_cfg['episodes']))
        cum_reward = np.zeros(
            (experiment_cfg['runs'], experiment_cfg['episodes']))
        time_to_solve = np.ones(
            (experiment_cfg['runs'], experiment_cfg['episodes'])) * \
                        experiment_cfg['steps']
        env = gym.make(env_name)
        env.seed(seed)

        # create agent, set the learning rate, tensorboard path..
        agent_config = agent_cfg[env_name][agent_name]
        agent_config['seed'] = seed
        arg_path = '/'.join([f'{k}/{v}' if not isinstance(v, List) else
                             f"{k}/{'_'.join(map(str, v))}"
                             for k, v in agent_config.items()])
        writer = SummaryWriter(
            log_dir=f"{experiment_cfg['tensorboard_path']}/"
            f"{experiment_cfg['date']}/{arg_path}")
        agent = Agent.build(type=agent_name, env=env, params=agent_config)

        # O(runs * episodes * max(test_rng * steps, steps))
        # go through runs, in order to further average, and episodes
        for r in range(experiment_cfg['runs']):
            for i_episode in range(experiment_cfg['episodes']):
                generator_obj = agent.learn(
                    num_steps=experiment_cfg['steps'])
                episode_result = next(generator_obj)
                cum_reward[r, i_episode] = episode_result[
                    'cum_reward']
                time_to_solve[r, i_episode] = episode_result[
                    'time_to_solve']
                episode_time[r, i_episode] = episode_result[
                    'episode_time']

                msg = f"run {r} | episode {i_episode}"
                for k, v in episode_result.items():
                    msg += f"| {k} {v} "
                print(msg)

            # add episodic results to step by step results for plotting
            agent.episodic_result['cum_reward'] = cum_reward[r]
            agent.episodic_result['time_to_solve'] = time_to_solve[r]
            agent.episodic_result['episode_time'] = episode_time[r]
            plot_episodic_results(idx=0, seed=seed, writer=writer,
                                  episode_result=agent.episodic_result)
        env.close()
        writer.close()

        # episodic rewards/time to solve
        cum_reward = np.mean(cum_reward, axis=0)
        time_to_solve = np.mean(time_to_solve, axis=0)
        return cum_reward, time_to_solve

    def generate_metrics(self, env_name: str, agent: str, results: List,
                         output: defaultdict) -> defaultdict:
        """generate whatever metrics needed for the experiment"""
        # results over the seeds
        output[env_name][agent]['mean_cum_rewards'] = np.mean(
            np.vstack([results[i][0] for i in range(len(results))]), axis=0)
        output[env_name][agent]['std_cum_rewards'] = np.std(
            np.vstack([results[i][0] for i in range(len(results))]), axis=0)
        output[env_name][agent]['upper_std_cum_rewards'] = \
            output[env_name][agent]['mean_cum_rewards'] + \
            output[env_name][agent]['std_cum_rewards']
        output[env_name][agent]['lower_std_cum_rewards'] = \
            output[env_name][agent]['mean_cum_rewards'] - \
            output[env_name][agent]['std_cum_rewards']
        output[env_name][agent]['max_cum_rewards'] = np.max(
            np.vstack([results[i][0] for i in range(len(results))]), axis=0)
        output[env_name][agent]['mean_timesteps'] = np.mean(
            np.vstack([results[i][1] for i in range(len(results))]), axis=0)
        output[env_name][agent]['min_timesteps'] = np.min(
            np.vstack([results[i][1] for i in range(len(results))]), axis=0)
        output[env_name][agent]['max_timesteps'] = np.max(
            np.vstack([results[i][1] for i in range(len(results))]), axis=0)
        return output
