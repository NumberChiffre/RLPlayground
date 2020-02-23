import numpy as np
from typing import Dict, List, Tuple
from gym import Env
import torch.optim as optim
import torch.nn as nn
import torch

from RLPlayground.agents.agent import Agent
from RLPlayground.models.feed_forward import LinearFeedForward, \
    TwoLayerFCBodyWithAction
from RLPlayground.models.replay import ReplayBuffer
from RLPlayground.utils.data_structures import RLAlgorithm, Experience


# TODO: implement non-nstep algos for Temporal-Difference Learning
class TDAgent(Agent):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        super(TDAgent, self).__init__(env=env, agent_cfg=agent_cfg)

        # specs for RL agent
        self.eps = agent_cfg['eps']
        self.gamma = agent_cfg['gamma']
        self.n_step = agent_cfg['n_step']
        self.alpha = agent_cfg['alpha']
        self.algo = agent_cfg['algo']

        # details for the NN model + experience replay
        self.replay_buffer = ReplayBuffer(
            capacity=agent_cfg['replay_capacity'])
        self.model = TwoLayerFCBodyWithAction(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_units=agent_cfg['nn_hidden_units']
        )
        self.target_model = TwoLayerFCBodyWithAction(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_units=agent_cfg['nn_hidden_units']
        )
        self.lr = agent_cfg['lr']
        self.batch_size = agent_cfg['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.loss_func = nn.MSELoss()

        self.Q = np.zeros(
            self.env.observation_space.shape[0] + (self.env.action_space.n,))

    def get_action(self, observation):
        # either take greedy action or explore with epsilon rate
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(observation, dtype=torch.float).unsqueeze(0)
            q_value = self.model(state)
            action = q_value.max(1)[1].data[0].item()
            return action

    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        observation, action, next_observation, reward, done = zip(*batch)
        observation = np.concatenate(observation, 0)
        next_observation = np.concatenate(next_observation, 0)

        observation = torch.FloatTensor(observation)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done)

        q_values = self.model.forward(observation)
        next_q_values = self.target_model.forward(next_observation)
        argmax_actions = self.model.forward(next_observation).max(1)[
            1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(
            1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # incorporate n-step for expected q-value
        if self.n_step > 0:
            expected_q_value = reward + self.gamma * (1 - done) * next_q_value

        # loss = loss_fn(q_value, expected_q_value.detach())
        loss = (expected_q_value.detach() - q_value).pow(2)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # if count % soft_update_freq == 0:
        self.target_model.load_state_dict(self.model.state_dict())

    def generate_n_step_q(self, t: int, num_steps: int,
                          experience_tuple: List[Experience]):
        tau = t - self.n_step + 1
        if tau >= 0:
            G = np.sum(
                [self.gamma ** (i - tau - 1) * experience_tuple[i].r for i
                 in range(tau + 1, min(tau + self.n_step, num_steps))])

            if tau + self.n_step < num_steps:
                idx = tau + self.n_step - 1
                idx_exp = experience_tuple[idx]
                tau_exp = experience_tuple[tau]

    # TODO: n-step with experienced replay + NN model for q-values
    def interact(self, num_steps: int) -> Tuple[float, int]:
        # use agent to interact with environment by making actions based on
        # optimal policy to obtain cumulative rewards
        observation = self.env.reset()
        experience_tuple = np.zeros(self.n_step + 1, )
        cr = 0
        for t in range(num_steps):
            next_observation, reward, done, info = self.env.step(action)
            exp = Experience(s0=observation, a=action, s1=next_observation,
                             reward=reward, done=done)
            self.replay_buffer.push(exp)
            experience_tuple[t % self.n_step] = exp

            # last step reached in episode
            # update the target network
            if done:
                num_steps = t + 1
                self.train()
            else:
                action = self.get_action(observation=next_observation)

            cr += reward
            # tau = t - self.n_step + 1
            # if tau >= 0:
            #     G = np.sum(
            #         [self.gamma ** (i - tau - 1) * experience_tuple[i].r for i
            #          in range(tau + 1, min(tau + self.n_step, num_steps))])
            #
            #     if tau + self.n_step < num_steps:
            #         idx = tau + self.n_step - 1
            #         idx_exp = experience_tuple[idx]
            #         tau_exp = experience_tuple[tau]
            #
            #         # sum the actions for given state
            #         if self.algo == RLAlgorithm.EXPECTED_SARSA.value:
            #             G = G + self.gamma ** self.n_step * np.sum(
            #                 [self.Q[idx_exp.s1][a] for a in
            #                  self.env.action_space])
            #
            #         # follow the action given by given policy/greedy
            #         elif self.algo == RLAlgorithm.SARSA.value:
            #             G = G + self.gamma ** self.n_step * \
            #                 self.Q[idx_exp.s1][idx_exp.action]
            #
            #         # use max operator for q-learning
            #         elif self.algo == RLAlgorithm.QLearning.value:
            #             G = G + self.gamma ** self.n_step * np.max(
            #                 self.Q[idx_exp.s1])
            #
            #         self.Q[tau_exp.s1][tau_exp.action] = \
            #             self.Q[tau_exp.s1][tau_exp.action] + self.alpha * (
            #                     G - self.Q[tau_exp.s1][tau_exp.action])
        return cr, num_steps
