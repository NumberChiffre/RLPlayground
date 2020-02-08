import numpy as np
from random import random, sample
from tqdm import tqdm
from plotly import tools
import plotly
import plotly.graph_objs as go


def kArmedTestbed(k, steps, epsilon, theseed=1):
    """
    :param k: the number of armed bandits in the experiment
    :param steps: the number of plays in a given run
    :param epsilon: probability of selecting a random action
    :param theseed: random seed used to ensure reproducibility of simulation results
    """

    # (v = current value-estimate, n = # of times action was sampled)
    actions = list(range(1, k + 1))
    action_values = {a: 0 for a in actions}
    samples = [0] * k

    # list of expected rewards
    muvec = np.random.standard_normal(k)
    actionvec, rewardvec = np.zeros(steps), np.zeros(steps)
    rewardvec_test = np.zeros(steps // 10)

    # loop through each simulation
    for t in range(steps):
        if random() < epsilon:
            a = sample(actions, 1)[0]
        else:
            a = max(action_values, key=action_values.get)

        if t % 10 == 0:
            reward_per_state_actions = []
            for i in range(5):
                actionvec[t + i] = a
                samples[a - 1] += 1
                reward_per_state_actions.append(np.random.normal(muvec[a - 1], 1))
            rewardvec_test[t // 10 - 1] = np.mean(reward_per_state_actions)
        # else:
        actionvec[t] = a
        samples[a - 1] += 1
        r = np.random.normal(muvec[a - 1], 1)
        rewardvec[t] = r

        # use equal weighting to update the value estimates
        action_values[a] = action_values[a] + (r - action_values[a]) / \
                           samples[
                               a - 1]
    return muvec, rewardvec, actionvec, rewardvec_test


def generate_plot(steps, rewards, optimality, test_rewards):
    x_axis = list(range(steps))
    sub_x = list(range(steps // 10))

    # these have their own yaxis
    subplots = ['Average reward', '% Optimal action', 'Test reward']

    # Main Layout
    n = len(subplots)
    fig = tools.make_subplots(
        rows=n,
        cols=1,
        subplot_titles=subplots,
        vertical_spacing=0.1,
    )
    fig['layout'].update(
        height=800,
        # width=1600,
        showlegend=True,
        title='k-armed bandit',
        # titlefont={"size": 25},
        # margin={'l': 100, 't': 0, 'r': 100},
        # hovermode='closest',
    )
    colors = {
        0.0: 'darkgreen',
        0.1: 'midnightblue',
        0.01: 'crimson',
    }
    for epsilon, average in rewards.items():
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

    for epsilon, average in test_rewards.items():
        trace = go.Scatter(
            x=sub_x,
            y=average,
            mode='lines',
            name=f'epsilon={epsilon}',
            marker=dict(
                color=colors[epsilon],
            )
        )
        fig.append_trace(trace, 3, 1)
    return fig


if __name__ == '__main__':
    epsilons = [0, 0.01, 0.1]
    runs = 10
    k = 10
    steps = 1000

    reward_per_state_action = {e: list() for e in epsilons}
    optimality_ratio = {e: list() for e in epsilons}
    t_rewards = {e: list() for e in epsilons}

    for e in tqdm(epsilons):
        # Thread(target=compute_metrics, args=(e,)).start()
        expected_rewards, observed_rewards, actions, test_rewards = np.zeros(
            (runs, k)), np.zeros((runs, steps)), np.zeros(
            (runs, steps)), np.zeros((runs, steps // 10))
        for i in tqdm(range(runs)):
            expected_rewards[i], observed_rewards[i], actions[
                i], test_rewards[i] = kArmedTestbed(k, steps, e)

        reward_per_state_action[e] = np.average(observed_rewards,
                                   axis=0)  # compute average rewards
        t_rewards[e] = np.average(test_rewards, axis=0)
        opt = np.argmax(expected_rewards, axis=1).reshape(runs, 1) + np.ones(
            (runs, 1))  # take argmax over all states
        act = np.ma.masked_values(actions,
                                  opt).mask  # filter the optimal actions
        optimality_ratio[e] = np.average(act, axis=0)

    fig = generate_plot(steps, reward_per_state_action, optimality_ratio, t_rewards)
    plotly.offline.plot(fig, filename='bandit_ex.html')
