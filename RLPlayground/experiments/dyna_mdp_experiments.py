import gym
import numpy as np
import pickle
import ray
from collections import defaultdict
from typing import Dict, List
import plotly
import plotly.graph_objects as go

from RLPlayground import ROOT_DIR
from RLPlayground.agents.dyna_mdp import MDPDynaAgent
from RLPlayground.utils.utilities import ProjectLogger

# logger
logger = ProjectLogger(level=10)


# gotta put more keys into defaultdict + able to pickle..
def foo1():
    return defaultdict(dict)


def foo():
    return defaultdict(foo1)


@ray.remote
def run_dp(env_name: str, seed: int = 1, dp_method: str = 'policy_iteration',
           experiment_cfg: Dict = {}, agent_cfg: Dict = {}):
    # random seed init
    np.random.seed(seed)

    # create agent
    env = gym.make(env_name)
    dp_agent = MDPDynaAgent(env=env, agent_cfg=agent_cfg)

    # result initialization
    cum_reward = np.zeros((experiment_cfg['runs'], experiment_cfg['episodes']))
    time_to_solve = np.ones(
        (experiment_cfg['runs'], experiment_cfg['episodes'])) * experiment_cfg[
                        'steps']
    test_cum_reward = np.zeros(
        (experiment_cfg['runs'], experiment_cfg['episodes'] // 10))
    test_time_to_solve = np.ones(
        (experiment_cfg['runs'], experiment_cfg['episodes'] // 10)) * \
                         experiment_cfg['steps']

    # init policy/value functions
    opt_policy, opt_value_func = None, None

    # O(runs * episodes * max(test_rng * steps, steps))
    for r in range(experiment_cfg['runs']):
        for i_episode in range(experiment_cfg['episodes']):

            if dp_method == 'policy_iteration':
                opt_policy, opt_value_func = dp_agent.policy_iteration(
                    num_steps=1, old_policy=opt_policy)
            elif dp_method == 'value_iteration':
                opt_policy, opt_value_func = dp_agent.value_iteration(
                    num_steps=1, value_func=opt_value_func)

            # for every 10th episode, lock current policy for testing
            if i_episode % experiment_cfg['train_rng'] == 0:
                avg_cr = list()

                # get reward for next 5 episodes
                for test_episode in range(experiment_cfg['test_rng']):
                    observation = dp_agent.env.reset()
                    cr = 0
                    for t in range(experiment_cfg['steps']):
                        action = opt_policy[observation]
                        observation, reward, done, info = dp_agent.env.step(
                            action)
                        cr += reward

                        if done:
                            test_time_to_solve[r, i_episode // 10 - 1] = t
                            break
                    avg_cr.append(cr)
                test_cum_reward[r, i_episode // 10 - 1] = np.mean(avg_cr)

            # keep updating the training return
            observation = dp_agent.env.reset()
            cr = 0
            for t in range(experiment_cfg['steps']):
                action = opt_policy[observation]
                observation, reward, done, info = dp_agent.env.step(action)
                cr += reward
                if done:
                    time_to_solve[r, i_episode] = t
                    break
            cum_reward[r, i_episode] = cr
        # print(f'cum reward {np.cumsum(cum_reward, axis=0)}')
    # cum_reward = np.cumsum(cum_reward, axis=0)
    env.close()
    cum_reward = np.mean(cum_reward, axis=0)
    # test_cum_reward = np.cumsum(test_cum_reward, axis=0)
    test_cum_reward = np.mean(test_cum_reward, axis=0)
    return cum_reward, test_cum_reward, time_to_solve, test_time_to_solve


def generate_plots(output: dict = None, episodes: int = 100,
                   train_rng: int = 10, plot_hyperparams: bool = True):
    """

    :param output: dictionary containing all environments, dp methods, metrics
    :param episodes: number of total number of episodes ran on experiments
    :param train_rng: training range used for testing
    :return:
    """
    if output is None:
        with open(f'{ROOT_DIR}/results/dyna_mdp_experiments.pickle',
                  'rb') as file:
            output = pickle.load(file)
    for env_name in output.keys():
        for dp_method in output[env_name].keys():

            if not plot_hyperparams:
                # training plots
                fig = generate_plot(env_name=env_name, train_or_test='train',
                                    dp_method=dp_method,
                                    output=output[env_name][dp_method]['train'],
                                    episodes=episodes)
                plotly.offline.plot(fig,
                                    filename=f'{ROOT_DIR}/results/{env_name}_{dp_method}_train.html')
                # testing plots
                fig = generate_plot(env_name=env_name, train_or_test='test',
                                    dp_method=dp_method,
                                    output=output[env_name][dp_method]['test'],
                                    episodes=episodes // train_rng)
                plotly.offline.plot(fig,
                                    filename=f'{ROOT_DIR}/results/{env_name}_{dp_method}_test.html')
            else:
                fig = generate_hyperparameters_plot(env_name, dp_method,
                                                    episodes,
                                                    output[env_name][dp_method])
                plotly.offline.plot(fig,
                                    filename=f'{ROOT_DIR}/results/{env_name}_{dp_method}_hyperparams.html')


def generate_hyperparameters_plot(env_name: str, dp_method: str,
                                  episodes: int, output: Dict):
    x_axis = list(range(episodes))
    subplots = [f'[{env_name.capitalize()}] | '
                f'[{dp_method.capitalize()}] | '
                f'Average Cumulative Rewards Per Episode Over 5 Seeds'
                ]

    # Main Layout
    n = len(subplots)
    fig = plotly.subplots.make_subplots(
        rows=n,
        cols=1,
        subplot_titles=subplots,
        vertical_spacing=0.1,
    )
    fig['layout'].update(
        # height=800,
        # width=1600,
        showlegend=True,
        title=f'HYPERPARAM EVALUATION for {env_name} - {dp_method}',
        titlefont={"size": 25},
        # margin={'l': 100, 't': 0, 'r': 100},
        # hovermode='closest',
    )
    # colors =

    for theta_dr, values in output.items():
        trace = go.Scatter(
            x=x_axis,
            y=values,
            mode='lines',
            name=f"theta | discount rate = [{theta_dr.split('_')[0]} | "
            f"{theta_dr.split('_')[1]}]",
            # marker=dict(
            #     color=colors[epsilon],
            # )
        )
        fig.append_trace(trace, 1, 1)
    return fig


def generate_plot(env_name: str, train_or_test: str, dp_method: str,
                  episodes: int, output: Dict):
    x_axis = list(range(episodes))
    subplots = [
        f'[{train_or_test.capitalize()}] | [{env_name.capitalize()}] | '
        f'[{dp_method.capitalize()}] | Average Cumulative Rewards Per Episode Over 5 Seeds',
        f'[{train_or_test.capitalize()}] | [{env_name.capitalize()}] | '
        f'[{dp_method.capitalize()}] | Average Number of Timesteps to Solve Per Episode Over 5 Seeds'
    ]

    # Main Layout
    n = len(subplots)
    fig = plotly.subplots.make_subplots(
        rows=n,
        cols=1,
        subplot_titles=subplots,
        vertical_spacing=0.1,
    )
    fig['layout'].update(
        # height=800,
        # width=1600,
        showlegend=True,
        title=f'Experiments for {env_name} - {dp_method}',
        titlefont={"size": 25},
        # margin={'l': 100, 't': 0, 'r': 100},
        # hovermode='closest',
    )

    colors = {
        'mean_cum_rewards': 'darkgreen',
        'upper_var_cum_rewards': 'midnightblue',
        'lower_var_cum_rewards': 'midnightblue',
        'max_cum_rewards': 'crimson',
    }

    time_colors = {
        'mean_timesteps': 'firebrick',
        'min_timesteps': 'royalblue',
    }

    for metric, values in output.items():
        if metric != 'var_cum_rewards':
            if metric in colors.keys():
                color = colors[metric]
                x = x_axis
                row = 1
                x_label = 'Episodes'
                y_label = 'Rewards'
            elif metric in time_colors.keys():
                color = time_colors[metric]
                x = x_axis
                row = 2
                x_label = 'Episodes [10th]'
                y_label = 'Timesteps'

            if 'min' in metric or 'max' in metric:
                line_style = 'dot'
            elif 'var_cum_rewards' in metric:
                line_style = 'dash'
            else:
                line_style = 'solid'

            trace = go.Scatter(
                x=x,
                y=values,
                mode='lines',
                line={
                    'dash': line_style
                },
                name=f'{metric}',
                # labels={
                #     'x': x_label,
                #     'y': y_label,
                # },
                marker=dict(
                    color=color,
                )
            )
            fig.append_trace(trace, row, 1)
    return fig


@ray.remote
def evaluate_hyperparameters(output, experiment_cfg: Dict, agent_cfg: Dict,
                             env_names: List, seeds: List, theta: float,
                             discount_rates: List):

    # slow-mo grid search over theta-discount rates
    agent_cfg['theta'] = theta
    for discount_rate in discount_rates:
        agent_cfg['discount_rate'] = discount_rate

        # go for each environment
        for env_name in env_names:

            # generate results for policy iteration
            for dp_method in dp_methods:
                print(f'{theta}_{discount_rate}_{env_name}_{dp_method}')
                results = [run_dp.remote(env_name=env_name, seed=seed,
                                         dp_method=dp_method,
                                         experiment_cfg=experiment_cfg,
                                         agent_cfg=agent_cfg[env_name])
                           for
                           seed in seeds]
                results = ray.get(results)
                logger.info(f'{dp_method} done with {theta}_{discount_rate}')

                # results over the seeds
                output[env_name][dp_method][
                    f'{theta}_{discount_rate}'] = np.mean(
                    np.vstack([results[i][0] for i in range(len(results))]),
                    axis=0)
                # with open(f'{ROOT_DIR}/results/dyna_mdp_experiments_hyperparameters.pickle',
                #           'wb') as file:
                #     pickle.dump(output, file)
    return output


if __name__ == "__main__":
    ray.init(
        # local_mode=True,
        ignore_reinit_error=True,
    )

    # specs for the experiment
    experiment_cfg = {
        'runs': 1,
        'steps': 5000,
        'episodes': 102,
        'train_rng': 10,
        'test_rng': 5,
    }

    # specs for the agent for tabular environments
    agent_cfg = {
        'FrozenLake-v0': {
            'theta': 1e-6,
            'discount_rate': 1.0,
        },
        'FrozenLake8x8-v0': {
            'theta': 1e-6,
            'discount_rate': 1.0,
        },
        'Taxi-v3': {
            'theta': 1e-8,
            'discount_rate': 1.0,
        }
    }

    # generate_plots(episodes=experiment_cfg['episodes'])
    # random seeds
    seeds = [34243]

    # dp methods
    dp_methods = ['value_iteration']

    # environments
    env_names = ['FrozenLake-v0']

    # eval hyperparams
    theta_vals = [1e-5]
    discount_rates = [1.0, 0.9]
    output = defaultdict(foo)
    results = [evaluate_hyperparameters.remote(output=output,
                                               experiment_cfg=experiment_cfg,
                                               agent_cfg=agent_cfg,
                                               env_names=env_names, seeds=seeds,
                                               theta=theta,
                                               discount_rates=discount_rates)
               for theta in theta_vals]
    results = ray.get(results)
    logger.info(results)
    # with open(f'{ROOT_DIR}/results/dyna_mdp_experiments_hyperparameters.pickle',
    #           'rb') as file:
    #     results = pickle.load(file)
    # generate_plots(episodes=experiment_cfg['episodes'], output=results[0],
    #                plot_hyperparams=True)

    # nested results, use module level for nested dict to save as pickle
    output = defaultdict(foo)
    #
    # # go for each environment
    # for env_name in env_names:
    #     # generate results for policy iteration
    #     for dp_method in dp_methods:
    #         results = [
    #             run_dp.remote(env_name=env_name, seed=seed,
    #                           dp_method=dp_method,
    #                           experiment_cfg=experiment_cfg,
    #                           agent_cfg=agent_cfg[env_name]) for seed in
    #             seeds]
    #         results = ray.get(results)
    #
    #         # results over the seeds
    #         output[env_name][dp_method]['train'][
    #             'mean_cum_rewards'] = np.mean(
    #             np.vstack([results[i][0] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['train'][
    #             'var_cum_rewards'] = np.var(
    #             np.vstack([results[i][0] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['train']['upper_var_cum_rewards'] = \
    #             output[env_name][dp_method]['train']['mean_cum_rewards'] + \
    #             output[env_name][dp_method]['train']['var_cum_rewards']
    #         output[env_name][dp_method]['train']['lower_var_cum_rewards'] = \
    #             output[env_name][dp_method]['train']['mean_cum_rewards'] - \
    #             output[env_name][dp_method]['train']['var_cum_rewards']
    #         output[env_name][dp_method]['train'][
    #             'max_cum_rewards'] = np.max(
    #             np.vstack([results[i][0] for i in range(len(results))]),
    #             axis=0)
    #
    #         output[env_name][dp_method]['test'][
    #             'mean_cum_rewards'] = np.mean(
    #             np.vstack([results[i][1] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['test']['var_cum_rewards'] = np.var(
    #             np.vstack([results[i][1] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['test']['upper_var_cum_rewards'] = \
    #             output[env_name][dp_method]['test']['mean_cum_rewards'] + \
    #             output[env_name][dp_method]['test']['var_cum_rewards']
    #
    #         output[env_name][dp_method]['test']['lower_var_cum_rewards'] = \
    #             output[env_name][dp_method]['test']['mean_cum_rewards'] - \
    #             output[env_name][dp_method]['test']['var_cum_rewards']
    #
    #         output[env_name][dp_method]['test']['max_cum_rewards'] = np.max(
    #             np.vstack([results[i][1] for i in range(len(results))]),
    #             axis=0)
    #
    #         output[env_name][dp_method]['train'][
    #             'mean_timesteps'] = np.mean(
    #             np.vstack([results[i][2] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['train']['min_timesteps'] = np.min(
    #             np.vstack([results[i][2] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['test']['mean_timesteps'] = np.mean(
    #             np.vstack([results[i][3] for i in range(len(results))]),
    #             axis=0)
    #         output[env_name][dp_method]['test']['min_timesteps'] = np.min(
    #             np.vstack([results[i][3] for i in range(len(results))]),
    #             axis=0)
    #
    #         # save onto disk
    #         with open(f'{ROOT_DIR}/results/dyna_mdp_experiments.pickle',
    #                   'wb') as file:
    #             pickle.dump(output, file)
    #         logger.info(output)
    #
    # generate_plots(episodes=experiment_cfg['episodes'])
