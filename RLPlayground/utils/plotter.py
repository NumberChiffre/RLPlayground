import pickle
import os
from typing import Dict
import plotly
import plotly.graph_objects as go
from torch.utils.tensorboard import SummaryWriter
from RLPlayground import RESULT_DIR, TENSORBOARD_DIR


def generate_plots(output: dict = None, episodes: int = 100,
                   train_rng: int = 10, plot_hyperparams: bool = True,
                   use_tensorboards: bool = True,
                   experiment_name: str = 'dyna_mdp'):
    """

    :param output: dictionary containing all environments, dp methods, metrics
    :param episodes: number of total number of episodes ran on experiments
    :param train_rng: training range used for testing
    :param plot_hyperparams: checks for what type of plot
    :param use_tensorboards: checks for using plotly or tensorboard
    :return:
    """
    if use_tensorboards:
        if plot_hyperparams:
            logdir = f'{TENSORBOARD_DIR}/{experiment_name}/hyperparams'
        else:
            logdir = f'{TENSORBOARD_DIR}/{experiment_name}/experiments'
        writer = SummaryWriter(log_dir=logdir)
    else:
        writer = None
    for env_name in output.keys():
        for dp_method in output[env_name].keys():
            if not plot_hyperparams:
                if not use_tensorboards:
                    logdir = f'{RESULT_DIR}/{env_name}_{dp_method}'
                # training plots
                generate_plot(env_name=env_name, train_or_test='train',
                              dp_method=dp_method,
                              output=output[env_name][dp_method]['train'],
                              episodes=episodes, logdir=f'{logdir}_train.html',
                              writer=writer)
                # testing plots
                generate_plot(env_name=env_name, train_or_test='test',
                              dp_method=dp_method,
                              output=output[env_name][dp_method]['test'],
                              episodes=episodes // train_rng,
                              logdir=f'{logdir}_test.html',
                              writer=writer)
            else:
                if not use_tensorboards:
                    logdir = f'{RESULT_DIR}/{env_name}_{dp_method}_hyperparams.html'
                else:
                    logdir = f'{logdir}/{env_name}_{dp_method}_hyperparams'
                generate_hyperparameters_plot(env_name=env_name,
                                              dp_method=dp_method,
                                              episodes=episodes,
                                              output=output[env_name][
                                                  dp_method],
                                              logdir=logdir,
                                              writer=writer)
    if use_tensorboards:
        writer.close()


def generate_hyperparameters_plot(env_name: str, dp_method: str,
                                  episodes: int, output: Dict, logdir: str,
                                  writer: SummaryWriter = None):
    subplots = [
        f'{env_name.capitalize()}_{dp_method.capitalize()}_Hyperparameters']
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
        plotly.offline.plot(fig, filename=logdir)


def generate_plot(env_name: str, train_or_test: str, dp_method: str,
                  episodes: int, output: Dict, logdir: str,
                  writer: SummaryWriter = None):
    subplots = [
        f'{train_or_test.capitalize()}_{env_name.capitalize()}_'
        f'{dp_method.capitalize()}_'
        f'Average Cumulative Rewards Per Episode Over 10 Seeds',
        f'{train_or_test.capitalize()}_{env_name.capitalize()}_'
        f'{dp_method.capitalize()}_'
        f'Average Number of Timesteps to Solve Per Episode Over 10 Seeds'
    ]

    colors = {
        'upper_var_cum_rewards': 'midnightblue',
        'mean_cum_rewards': 'darkgreen',
        'lower_var_cum_rewards': 'midnightblue',
        'max_cum_rewards': 'crimson',
    }

    time_colors = {
        'mean_timesteps': 'firebrick',
        'min_timesteps': 'royalblue',
    }

    if writer is not None:
        timesteps = {k: v for k, v in output.items() if k in time_colors.keys()}
        bounded_rewards = {k: v for k, v in output.items() if
                           k in colors.keys() and 'max' not in k}
        #
        # for metric, values in output.items():
        #     for i in range(episodes):
        #         if metric in colors.keys():
        #             writer.add_scalar(f'{env_name}/{dp_method}/{metric}',
        #                               values[i], i)
        #
        # layout = {env_name:
        #               {dp_method:
        #                    ['Margin', [f'{env_name}/{dp_method}/{l}' for l in
        #                      list(bounded_rewards.keys())]]
        #               },
        #          }
        #
        # writer.add_custom_scalars(layout)
        # # writer.add_custom_scalars_marginchart(
        # #     title=f'{train_or_test}/{env_name}/{dp_method}/rewards',
        # #     tags=[f'{env_name}/{dp_method}/{l}' for l in
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
            title=f'Experiments for {env_name} - {dp_method}',
            titlefont={"size": 25},
            # margin={'l': 100, 't': 0, 'r': 100},
            # hovermode='closest',
        )

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

        plotly.offline.plot(fig, filename=logdir)


if __name__ == '__main__':
    from _jsonnet import evaluate_file
    from RLPlayground import CONFIG_DIR
    import json

    with open(f'{RESULT_DIR}/dyna_mdp_experiments_hyperparameters.pickle',
              'rb') as file:
        hyperparams = pickle.load(file)
    with open(f'{RESULT_DIR}/dyna_mdp_experiments.pickle', 'rb') as file:
        output = pickle.load(file)

    cfg = evaluate_file(f'{CONFIG_DIR}/dyna_mdp_config.jsonnet')
    cfg = json.loads(cfg)

    # specs for the experiment
    experiment_cfg = cfg['experiment_cfg']
    generate_plots(output=hyperparams, episodes=experiment_cfg['episodes'],
                   plot_hyperparams=True, use_tensorboards=False,
                   experiment_name=cfg['experiment_name'])

    generate_plots(output=output, episodes=experiment_cfg['episodes'],
                   train_rng=experiment_cfg['train_rng'],
                   plot_hyperparams=False, use_tensorboards=False,
                   experiment_name=cfg['experiment_name'])
