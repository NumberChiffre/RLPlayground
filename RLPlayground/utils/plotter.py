import pickle
import os
from typing import Dict
from collections import defaultdict
import plotly
import plotly.graph_objects as go
from torch.utils.tensorboard import SummaryWriter
from RLPlayground import RESULT_DIR, TENSORBOARD_DIR


def plot_episodic_results(idx: int, seed: int, writer: SummaryWriter,
                          episode_result: defaultdict):
    """writes episodic results into tensorboard, called at the end of each
    episode"""
    for k, v in episode_result.items():
        for i in range(idx, idx + len(v)):
            if 'net_params' not in k:
                writer.add_scalar(tag=k, scalar_value=v[i - idx], global_step=i)
            else:
                for tag, value in v[i - idx]:
                    tag_ = f"{tag.replace('.', '/')}/{str(seed)}"
                    writer.add_histogram(tag_, value.data.cpu().numpy(), i)
                    tag_ = f"{tag.replace('.', '/')}/grad/{str(seed)}"
                    writer.add_histogram(tag_, value.grad.data.cpu().numpy(),
                                         i)


def generate_plots(plot_hyperparams: bool, use_tensorboards: bool, output: dict,
                   experiment_cfg: dict):
    """

    :param output: dictionary containing all environments, dp methods, metrics
    :param plot_hyperparams: checks for what type of plot
    :param use_tensorboards: checks for using plotly or tensorboard
    :return:
    """
    episodes = experiment_cfg['episodes']
    train_rng = experiment_cfg['train_rng']
    experiment_name = experiment_cfg['experiment_name']
    if use_tensorboards:
        if plot_hyperparams:
            logdir = f'{TENSORBOARD_DIR}/{experiment_name}/hyperparams'
        else:
            logdir = f'{TENSORBOARD_DIR}/{experiment_name}/experiments'
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None
    for env_name in output.keys():
        for algo in output[env_name].keys():
            if not plot_hyperparams:
                if not use_tensorboards:
                    logdir = f'{RESULT_DIR}/{env_name}_{algo}'
                # training plots
                generate_plot(env_name=env_name, train_or_test='train',
                              algo=algo,
                              output=output[env_name][algo]['train'],
                              episodes=episodes, logdir=f'{logdir}_train.html',
                              writer=writer)
                # testing plots
                generate_plot(env_name=env_name, train_or_test='test',
                              algo=algo,
                              output=output[env_name][algo]['test'],
                              episodes=episodes // train_rng,
                              logdir=f'{logdir}_test.html',
                              writer=writer)
            else:
                if not use_tensorboards:
                    logdir = f'{RESULT_DIR}/{env_name}_{algo}_hyperparams.html'
                else:
                    logdir = f'{logdir}/{env_name}_{algo}_hyperparams'
                generate_hyperparameters_plot(env_name=env_name,
                                              algo=algo,
                                              episodes=episodes,
                                              output=output[env_name][
                                                  algo],
                                              logdir=logdir,
                                              writer=writer)
    if use_tensorboards:
        writer.close()


def generate_hyperparameters_plot(env_name: str, algo: str,
                                  episodes: int, output: Dict, logdir: str,
                                  writer: SummaryWriter = None):
    subplots = [
        f'{env_name.capitalize()}_{algo.capitalize()}_Hyperparameters']
    if writer is not None:
        # for theta_dr, values in output.items():
        #     for i in range(len(values)):
        #         writer.add_scalar(tag=f'{theta_dr}',
        #                           scalar_value=values[i], global_step=i)
        # writer.add_custom_scalars_multilinechart(list(output.keys()))
        for i in range(episodes):
            writer.add_scalars(
                main_tag=subplots[0],
                tag_scalar_dict={theta_dr: values[i] for theta_dr, values in
                                 output.items()}, global_step=i)
    else:
        # Main Layout
        x_axis = list(range(episodes))
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
            title=f'HYPERPARAM EVALUATION for {env_name} - {algo}',
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
        plotly.offline.plot(fig, filename=logdir)


def generate_plot(env_name: str, train_or_test: str, algo: str,
                  episodes: int, output: Dict, logdir: str,
                  writer: SummaryWriter = None):
    subplots = [
        f'{train_or_test.capitalize()}_{env_name.capitalize()}_'
        f'{algo.capitalize()}_'
        f'Average Cumulative Rewards Per Episode Over 10 Seeds',
        f'{train_or_test.capitalize()}_{env_name.capitalize()}_'
        f'{algo.capitalize()}_'
        f'Average Number of Timesteps to Solve Per Episode Over 10 Seeds'
    ]

    colors = {
        'upper_std_cum_rewards': 'midnightblue',
        'mean_cum_rewards': 'darkgreen',
        'lower_std_cum_rewards': 'midnightblue',
        'max_cum_rewards': 'crimson',
    }

    time_colors = {
        'mean_timesteps': 'firebrick',
        'min_timesteps': 'royalblue',
        'max_timesteps': 'midnightblue',
    }

    if writer is not None:
        timesteps = {k: v for k, v in output.items() if k in time_colors.keys()}
        bounded_rewards = {k: v for k, v in output.items() if
                           k in colors.keys() and 'max' not in k}
        #
        # for metric, values in output.items():
        #     for i in range(episodes):
        #         if metric in colors.keys():
        #             writer.add_scalar(f'{env_name}/{algo}/{metric}',
        #                               values[i], i)
        #
        # layout = {env_name:
        #               {algo:
        #                    ['Margin', [f'{env_name}/{algo}/{l}' for l in
        #                      list(bounded_rewards.keys())]]
        #               },
        #          }
        #
        # writer.add_custom_scalars(layout)
        # # writer.add_custom_scalars_marginchart(
        # #     title=f'{train_or_test}/{env_name}/{algo}/rewards',
        # #     tags=[f'{env_name}/{algo}/{l}' for l in
        # #           list(bounded_rewards.keys())])

        for i in range(episodes):
            writer.add_scalars(
                main_tag=subplots[0],
                tag_scalar_dict={
                    f'{metric}': values[i] for metric, values in
                    bounded_rewards.items()}, global_step=i)
            writer.add_scalars(
                main_tag=subplots[1],
                tag_scalar_dict={
                    f'{metric}': values[i] for metric, values in
                    timesteps.items()}, global_step=i)

    else:
        # Main Layout
        x_axis = list(range(episodes))
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
            title=f'Experiments for {env_name} - {algo}',
            titlefont={"size": 25},
            # margin={'l': 100, 't': 0, 'r': 100},
            # hovermode='closest',
        )

        for metric, values in output.items():
            if metric != 'std_cum_rewards':
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

        plotly.offline.plot(fig, filename=logdir)

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def plot_lr_reward(output: Dict):
#     for env_name in output.keys():
#         for agent, capacities in output[env_name].items():
#
#             plt.rcParams.update({'font.size': 18})
#             fig = plt.figure(figsize=(10, 16)).add_subplot(111)
#             fig.title.set_text(f'{agent} Plot #1')
#             fig.set_ylabel('Average Reward of last 10 episodes')
#             fig.set_xlabel(r'$\alpha$')
#
#             for capacity, metrics in capacities.items():
#                 for metric, lrs in metrics.items():
#                     if metric == 'mean_cum_rewards':
#                         fig.plot(list(lrs.keys()),
#                                  [np.mean(v[-10:]) for k, v in lrs.items()],
#                                  label=f'capacity={capacity}')
#
#             plt.grid(linestyle='--')
#             plt.legend(loc='upper left')
#             plt.show()
#             plt.savefig(f'{agent}_plot_1.png')
#             plt.clf()
#
#     # find max based on...?
#     max_cap, max_lr = 500, 0.05
#     # pick a capacity/lr with best episode rewards..
#     for env_name in output.keys():
#         for agent, capacities in output[env_name].items():
#             plt.rcParams.update({'font.size': 18})
#             fig = plt.figure(figsize=(16, 10)).add_subplot(111)
#             fig.title.set_text(f'{agent} Plot #2')
#             fig.set_ylabel('Reward')
#             fig.set_xlabel('Episode')
#             for capacity, metrics in capacities.items():
#                 if capacity == max_cap:
#                     for metric, lrs in metrics.items():
#                         if 'mean' in metric or 'upper' in metric or 'lower' in metric:
#                             fig.plot(np.arange(len(lrs[max_lr])),
#                                      lrs[max_lr],
#                                      label=metric)
#             plt.grid(linestyle='--')
#             plt.legend(loc='upper left')
#             plt.show()
#             plt.savefig(f'{agent}_plot_2.png')
#             plt.clf()

if __name__ == '__main__':
    from _jsonnet import evaluate_file
    from RLPlayground import CONFIG_DIR
    import json

    # with open(f'{RESULT_DIR}/dyna_mdp_experiments_hyperparameters.pickle',
    #           'rb') as file:
    #     hyperparams = pickle.load(file)
    with open(f'{RESULT_DIR}/experiments.pickle', 'rb') as file:
        output = pickle.load(file)

    cfg = evaluate_file(f'{CONFIG_DIR}/n_step_td_config.jsonnet')
    cfg = json.loads(cfg)

    # specs for the experiment
    experiment_cfg = cfg['experiment_cfg']
    plot_lr_reward(output)
    # generate_plots(output=hyperparams, plot_hyperparams=True,
    # use_tensorboards=False, experiment_cfg=experiment_cfg)
    #
    # generate_plots(output=output, plot_hyperparams=False,
    #                use_tensorboards=False, experiment_cfg=experiment_cfg)
