import pickle
from typing import Dict
import plotly
import plotly.graph_objects as go
from RLPlayground import RESULT_DIR


def generate_plots(output: dict = None, episodes: int = 100,
                   train_rng: int = 10, plot_hyperparams: bool = True):
    """

    :param output: dictionary containing all environments, dp methods, metrics
    :param episodes: number of total number of episodes ran on experiments
    :param train_rng: training range used for testing
    :return:
    """
    if output is None:
        with open(f'{RESULT_DIR}/dyna_mdp_experiments.pickle', 'rb') as file:
            output = pickle.load(file)
    for env_name in output.keys():
        for dp_method in output[env_name].keys():
            if not plot_hyperparams:
                # training plots
                generate_plot(env_name=env_name, train_or_test='train',
                                   dp_method=dp_method,
                                   output=output[env_name][dp_method]['train'],
                                   episodes=episodes)
                # testing plots
                generate_plot(env_name=env_name, train_or_test='test',
                                   dp_method=dp_method,
                                   output=output[env_name][dp_method]['test'],
                                   episodes=episodes // train_rng)
            else:
                generate_hyperparameters_plot(env_name=env_name,
                                                   dp_method=dp_method,
                                                   episodes=episodes,
                                                   output=output[env_name][
                                                       dp_method])

def generate_hyperparameters_plot(env_name: str, dp_method: str,
                                  episodes: int, output: Dict):
    x_axis = list(range(episodes))
    subplots = [f'[{env_name.capitalize()}] | '
                f'[{dp_method.capitalize()}] | '
                f'Average Cumulative Rewards Per Episode Over 10 Seeds'
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
    plotly.offline.plot(fig, filename=f'{RESULT_DIR}/{env_name}_{dp_method}_'
    f'hyperparams.html')


def generate_plot(env_name: str, train_or_test: str, dp_method: str,
                  episodes: int, output: Dict):
    x_axis = list(range(episodes))
    subplots = [
        f'[{train_or_test.capitalize()}] | [{env_name.capitalize()}] | '
        f'[{dp_method.capitalize()}] '
        f'| Average Cumulative Rewards Per Episode Over 10 Seeds',
        f'[{train_or_test.capitalize()}] | [{env_name.capitalize()}] | '
        f'[{dp_method.capitalize()}] | '
        f'Average Number of Timesteps to Solve Per Episode Over 10 Seeds'
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

    plotly.offline.plot(fig, filename=f'{RESULT_DIR}/{env_name}_'
    f'{dp_method}_{train_or_test}.html')
