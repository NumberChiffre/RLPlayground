import ray
import gym
import numpy as np
import pickle
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from RLPlayground.agents.td import TDAgent
from RLPlayground.experiments.experiment import Experiment
from RLPlayground.utils.logger import ProjectLogger
from RLPlayground.utils.utils import nested_d

from torch.utils.tensorboard import SummaryWriter


class TDExperiment(Experiment):
    def __init__(self,
                 logger: ProjectLogger,
                 *args, **kwargs):
        super().__init__(logger=logger, *args, **kwargs)
        self.replay_buffer_capacities = self.experiment_cfg[
            'replay_buffer_capacities']
        self.lrs = np.array(
            [float((i) / 1000) if i > 0 else float(1 / 1000) for i in
             range(0, 301, 10)])
        self.lrs = np.array([0.001])
        self.experiment_cfg['lrs'] = self.lrs

    def generate_metrics(self, env_name: str, algo: str, capacity: int,
                         results: List, output: defaultdict) -> defaultdict:
        """generate whatever metrics needed for the experiment"""
        # results over the seeds
        for idx, lr in enumerate(self.experiment_cfg['lrs']):
            output[env_name][algo][capacity]['mean_cum_rewards'][lr] = np.mean(
                [results[i][0] for i in range(len(results))], axis=0)[idx]
            output[env_name][algo][capacity]['std_cum_rewards'][lr] = np.std(
                [results[i][0] for i in range(len(results))], axis=0)[idx]
            output[env_name][algo][capacity]['upper_std_cum_rewards'][lr] = \
                output[env_name][algo][capacity]['mean_cum_rewards'][lr] + \
                output[env_name][algo][capacity]['std_cum_rewards'][lr]
            output[env_name][algo][capacity]['lower_std_cum_rewards'][lr] = \
                output[env_name][algo][capacity]['mean_cum_rewards'][lr] - \
                output[env_name][algo][capacity]['std_cum_rewards'][lr]
            output[env_name][algo][capacity]['max_cum_rewards'][lr] = np.max(
                [results[i][0] for i in range(len(results))], axis=0)[idx]
            output[env_name][algo][capacity]['mean_timesteps'][lr] = np.mean(
                [results[i][1] for i in range(len(results))], axis=0)[idx]
            output[env_name][algo][capacity]['min_timesteps'][lr] = np.min(
                [results[i][1] for i in range(len(results))], axis=0)[idx]
            output[env_name][algo][capacity]['max_timesteps'][lr] = np.max(
                [results[i][1] for i in range(len(results))], axis=0)[idx]
        return output

    def run(self) -> defaultdict:
        """for each gym environment and RL algorithms, test different replay
        buffer capaciity over multiple seeds"""
        output = defaultdict(nested_d)
        for env_name in self.env_names:
            for algo in self.algos:
                for capacity in self.replay_buffer_capacities:
                    self.agent_cfg[env_name][algo]['replay_capacity'] = capacity
                    results = [TDExperiment._inner_run.remote(
                        agent_cfg=self.agent_cfg,
                        experiment_cfg=self.experiment_cfg,
                        env_name=env_name, seed=seed, algo=algo) for seed in
                        self.seeds]
                    results = ray.get(results)
                    output = self.generate_metrics(env_name=env_name, algo=algo,
                                                   capacity=capacity,
                                                   results=results,
                                                   output=output)
                    with open(self.experiment_cfg['experiment_path'],
                              'wb') as file:
                        pickle.dump(output, file)
                self.logger.info(
                    f'Finished running experiments for {env_name} | {algo}')
        return output

    @staticmethod
    @ray.remote
    def _inner_run(agent_cfg: dict, experiment_cfg: dict, env_name: str,
                   seed: int = 1, algo: str = 'sarsa') -> Tuple[
        np.array, np.array]:
        # seed and result initialization
        np.random.seed(seed)
        cum_reward = np.zeros((len(experiment_cfg['lrs']),
                               experiment_cfg['runs'],
                               experiment_cfg['episodes']))
        time_to_solve = np.ones((len(experiment_cfg['lrs']),
                                 experiment_cfg['runs'],
                                 experiment_cfg['episodes'])) * experiment_cfg[
                            'steps']

        # O(lrs * runs * episodes * max(test_rng * steps, steps))
        for i_lr in range(len(experiment_cfg['lrs'])):
            # create environment and agent
            # set the learning rate
            # set the tensorboard path..
            env = gym.make(env_name)
            agent_config = agent_cfg[env_name][algo]
            agent_config['lr'] = experiment_cfg['lrs'][i_lr]
            agent_config['seed'] = seed
            arg_path = '/'.join([f'{k}/{v}' for k, v in agent_config.items()])
            writer = SummaryWriter(
                log_dir=f"{experiment_cfg['tensorboard_path']}/{arg_path}/seed/"
                f"{seed}")
            agent = TDAgent(env=env, writer=writer, agent_cfg=agent_config)

            # go through runs, in order to further average, and episodes
            for r in range(experiment_cfg['runs']):
                if 'CartPole' in env_name:
                    scores = deque(maxlen=agent_cfg[env_name][
                        'consecutive_steps_to_solve'])
                for i_episode in range(experiment_cfg['episodes']):
                    cr, t = agent.interact(num_steps=experiment_cfg['steps'],
                                           episode=i_episode)
                    writer.add_scalar(tag='Episode-Duration', scalar_value=t,
                                      global_step=i_episode)
                    time_to_solve[i_lr, r, i_episode] = t
                    cum_reward[i_lr, r, i_episode] = cr
                    print(f"lr {experiment_cfg['lrs'][i_lr]} | "
                          f"episode {i_episode + 1} | cum_reward {cr} | "
                          f"time_to_solve {t}")

                    if 'CartPole' in env_name:
                        scores.append(t)
                        if np.mean(scores) >= agent_cfg[env_name][
                            'average_score_to_solve'] and i_episode >= \
                                agent_cfg[env_name]['consecutive_steps_to_solve']:
                            print(
                                f"Ran {i_episode} episodes, solved after "
                                f"{i_episode - agent_cfg[env_name]['consecutive_steps_to_solve']}")
                            break
            env.close()
        writer.close()
        # generates learning rates * episodes
        cum_reward = np.mean(cum_reward, axis=1)
        time_to_solve = np.mean(time_to_solve, axis=1)
        return cum_reward, time_to_solve
