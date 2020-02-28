import ray
import gym
import numpy as np
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from RLPlayground.agents.agent import Agent
from RLPlayground.experiments.experiment import Experiment
from RLPlayground.utils.logger import ProjectLogger
from RLPlayground.utils.utils import nested_d
from RLPlayground.utils.plotter import plot_episodic_results

from torch.utils.tensorboard import SummaryWriter


@Experiment.register('DeepTDExperiment')
class DeepTDExperiment(Experiment):
    def __init__(self,
                 logger: ProjectLogger,
                 *args, **kwargs):
        super().__init__(logger=logger, *args, **kwargs)
        self.replay_buffer_capacities = self.experiment_cfg[
            'replay_buffer_capacities']
        self.lrs = self.experiment_cfg['lrs']

    @classmethod
    def from_params(cls, logger: ProjectLogger, params: Dict):
        return cls(logger, **params)

    def run(self) -> defaultdict:
        """for each gym environment and RL agentrithms, test different replay
        buffer capaciity over multiple seeds"""
        output = defaultdict(nested_d)
        for env_name in self.env_names:
            for agent in self.agents:
                for capacity in self.replay_buffer_capacities:
                    self.agent_cfg[agent]['experience_replay']['params'][
                        'capacity'] = capacity
                    # TODO: get rid of this for hardcoding..
                    if capacity >= 10000:
                        self.agent_cfg[agent]['update_freq'] = 60000
                        self.agent_cfg[agent]['warm_up_freq'] = 500
                    results = [DeepTDExperiment._inner_run.remote(
                        agent_cfg=self.agent_cfg,
                        experiment_cfg=self.experiment_cfg,
                        env_name=env_name, seed=seed, agent_name=agent) for seed
                        in self.seeds]
                    results = ray.get(results)
                    output = self.generate_metrics(env_name=env_name,
                                                   agent=agent,
                                                   capacity=capacity,
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
        episode_time = np.zeros((len(experiment_cfg['lrs']),
                                 experiment_cfg['runs'],
                                 experiment_cfg['episodes']))
        cum_reward = np.zeros((len(experiment_cfg['lrs']),
                               experiment_cfg['runs'],
                               experiment_cfg['episodes']))
        time_to_solve = np.ones((len(experiment_cfg['lrs']),
                                 experiment_cfg['runs'],
                                 experiment_cfg['episodes'])) * experiment_cfg[
                            'steps']
        env = gym.make(env_name)

        # O(lrs * runs * episodes * max(test_rng * steps, steps))
        for i_lr in range(len(experiment_cfg['lrs'])):
            # create agent, set the learning rate, tensorboard path..
            agent_config = agent_cfg[agent_name]
            agent_config['lr'] = experiment_cfg['lrs'][i_lr]
            agent_config['seed'] = seed
            arg_path = '/'.join([f'{k}/{v}' if not isinstance(v, List) else
                                 f"{k}/{'_'.join(map(str, v))}"
                                 for k, v in agent_config.items()])
            writer = SummaryWriter(
                log_dir=f"{experiment_cfg['tensorboard_path']}/"
                f"{experiment_cfg['date']}/{arg_path}")
            agent = Agent.build(type=agent_name, env=env, params=agent_config)

            # go through runs, in order to further average, and episodes
            for r in range(experiment_cfg['runs']):
                for i_episode in range(experiment_cfg['episodes']):
                    generator_obj = agent.interact(
                        num_steps=experiment_cfg['steps'])
                    episode_result = next(generator_obj)
                    cum_reward[i_lr, r, i_episode] = episode_result[
                        'cum_reward']
                    time_to_solve[i_lr, r, i_episode] = episode_result[
                        'time_to_solve']
                    episode_time[i_lr, r, i_episode] = episode_result[
                        'episode_time']

                    # if cum_reward[i_lr, r, i_episode] >= agent_cfg[
                    #     'average_score_to_solve']:
                    msg = f"lr {agent_config['lr']} | run {r} | " \
                        f"episode {i_episode} | eps {agent.eps} "
                    for k, v in episode_result.items():
                        msg += f"| {k} {v} "
                    print(msg)

                # add episodic results to step by step results for plotting
                agent.episodic_result['cum_reward'] = cum_reward[i_lr, r]
                agent.episodic_result['time_to_solve'] = time_to_solve[i_lr, r]
                agent.episodic_result['episode_time'] = episode_time[i_lr, r]
                plot_episodic_results(idx=0, seed=seed, writer=writer,
                                      episode_result=agent.episodic_result)
        env.close()
        writer.close()

        # generates learning rates * episodes
        cum_reward = np.mean(cum_reward, axis=1)
        time_to_solve = np.mean(time_to_solve, axis=1)
        return cum_reward, time_to_solve

    def generate_metrics(self, env_name: str, agent: str, capacity: int,
                         results: List, output: defaultdict) -> defaultdict:
        """generate whatever metrics needed for the experiment"""
        # results over the seeds
        for idx, lr in enumerate(self.experiment_cfg['lrs']):
            output[env_name][agent][capacity]['mean_cum_rewards'][lr] = np.mean(
                [results[i][0] for i in range(len(results))], axis=0)[idx]
            output[env_name][agent][capacity]['std_cum_rewards'][lr] = np.std(
                [results[i][0] for i in range(len(results))], axis=0)[idx]
            output[env_name][agent][capacity]['upper_std_cum_rewards'][lr] = \
                output[env_name][agent][capacity]['mean_cum_rewards'][lr] + \
                output[env_name][agent][capacity]['std_cum_rewards'][lr]
            output[env_name][agent][capacity]['lower_std_cum_rewards'][lr] = \
                output[env_name][agent][capacity]['mean_cum_rewards'][lr] - \
                output[env_name][agent][capacity]['std_cum_rewards'][lr]
            output[env_name][agent][capacity]['max_cum_rewards'][lr] = np.max(
                [results[i][0] for i in range(len(results))], axis=0)[idx]
            output[env_name][agent][capacity]['mean_timesteps'][lr] = np.mean(
                [results[i][1] for i in range(len(results))], axis=0)[idx]
            output[env_name][agent][capacity]['min_timesteps'][lr] = np.min(
                [results[i][1] for i in range(len(results))], axis=0)[idx]
            output[env_name][agent][capacity]['max_timesteps'][lr] = np.max(
                [results[i][1] for i in range(len(results))], axis=0)[idx]
        return output
