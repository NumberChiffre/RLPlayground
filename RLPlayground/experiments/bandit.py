import numpy as np
from typing import List, Dict
from random import random, sample
from tqdm import tqdm
from plotly import tools
import plotly
import plotly.graph_objs as go
from RLPlayground import ROOT_DIR, RESULT_DIR


def multi_armed_testbed(k: int, steps: int, epsilon: List, seed: int = 1):
    """
    :param k: the number of armed bandits in the experiment
    :param steps: the number of plays in a given run
    :param epsilon: probability of selecting a random action
    :param theseed: random seed used to ensure reproducibility of simulation results
    """
    np.random.seed(seed)
    sample_actions = list(range(1, k + 1))
    action_values = {a: 0 for a in sample_actions}
    samples = [0] * k
    expected_rewards = np.random.standard_normal(k)
    observed_rewards = np.zeros(steps)
    actions = np.zeros(steps)

    # loop through each simulation
    for t in range(steps):
        if random() < epsilon:
            a = sample(sample_actions, 1)[0]
        else:
            a = max(action_values, key=action_values.get)
        actions[t] = a
        samples[a - 1] += 1
        r = np.random.normal(expected_rewards[a - 1], 1)
        observed_rewards[t] = r

        # use equal weighting to update the value estimates
        action_values[a] = action_values[a] + (r - action_values[a]) / samples[
            a - 1]
    return expected_rewards, observed_rewards, actions


def generate_plot(steps: int, observed_rewards: Dict, optimality: Dict):
    x_axis = list(range(steps))
    subplots = ['Average reward', '% Optimal action']
    n = len(subplots)

    fig = tools.make_subplots(
        rows=n,
        cols=1,
        subplot_titles=subplots,
        vertical_spacing=0.1,
    )
    fig['layout'].update(
        height=800,
        showlegend=True,
        title='k-armed bandit',
    )
    colors = {
        0.0: 'darkgreen',
        0.1: 'midnightblue',
        0.01: 'crimson',
    }
    for epsilon, average in observed_rewards.items():
        trace = go.Scatter(
            x=x_axis,
            y=average,
            mode='lines',
            name=f'epsilon={epsilon}',
            marker=dict(
                color=colors[epsilon],
            )
        )
        fig.append_trace(trace, 1, 1)

    for epsilon, percent_optimal in optimality.items():
        trace = go.Scatter(
            x=x_axis,
            y=percent_optimal,
            mode='lines',
            name=f'epsilon={epsilon}',
            marker=dict(
                color=colors[epsilon],
            )
        )
        fig.append_trace(trace, 2, 1)
    return fig


if __name__ == '__main__':
    k = 10
    runs = 2000
    steps = 1000
    epsilons = [0, 0.01, 0.1]
    avg_reward = {e: list() for e in epsilons}
    optimality_ratio = {e: list() for e in epsilons}

    for e in tqdm(epsilons):
        expected_rewards, observed_rewards, sample_actions = np.zeros(
            (runs, k)), np.zeros((runs, steps)), np.zeros((runs, steps))

        for i in tqdm(range(runs)):
            expected_rewards[i], observed_rewards[i], sample_actions[
                i] = multi_armed_testbed(k, steps, e)

        # compute average observed_rewards
        avg_reward[e] = np.average(observed_rewards, axis=0)

        # take argmax over all states
        opt = np.argmax(expected_rewards, axis=1).reshape(runs, 1) + np.ones(
            (runs, 1))

        # filter the optimal sample actions
        act = np.ma.masked_values(sample_actions, opt).mask
        optimality_ratio[e] = np.average(act, axis=0)

    # save into local plot
    fig = generate_plot(steps, avg_reward, optimality_ratio)
    plotly.offline.plot(fig,
                        filename=f'{RESULT_DIR}/bandits/multi_armed_testbed.html')
