import numpy as np
from typing import Dict, List, Tuple
from gym import Env
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from RLPlayground.agents.agent import Agent
from RLPlayground.models.feed_forward import LinearFCBody
from RLPlayground.models.replay import ReplayBuffer
from RLPlayground.utils.data_structures import RLAlgorithm, Transition


class TDAgent(Agent):
    def __init__(self,
                 env: Env,
                 writer: SummaryWriter,
                 agent_cfg: Dict):
        """

        :param env: gym environment used for Experiment
        :param agent_cfg: config file for given agent
        """
        super(TDAgent, self).__init__(env=env, agent_cfg=agent_cfg)
        self.train_count = 0

        # specs for RL agent
        self.eps = agent_cfg['eps']
        self.gamma = agent_cfg['gamma']
        self.n_step = agent_cfg['n_step']
        self.alpha = agent_cfg['alpha']
        self.algo = agent_cfg['algo']
        self.update_freq = agent_cfg['update_freq']

        # details for the NN model + experience replay
        self.replay_buffer = ReplayBuffer(
            capacity=agent_cfg['replay_capacity'], n_step=self.n_step)
        self.model = LinearFCBody(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_units=tuple(agent_cfg['nn_hidden_units'])
        )
        self.target_model = LinearFCBody(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_units=tuple(agent_cfg['nn_hidden_units'])
        )
        self.lr = agent_cfg['lr']
        self.batch_size = agent_cfg['batch_size']
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

        # Huber loss acts like the mean squared error when the error is small,
        # but like the mean absolute error when the error is large
        # this makes it more robust to outliers when the estimates of Q
        # are very noisy. It is calculated over a batch of transitions
        # sampled from the replay memory:
        self.loss_func = nn.SmoothL1Loss()
        self.writer = writer

    def get_action(self, observation):
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(observation)
            q_value = self.model(state)
            action = q_value.max(1)[1].data[0].item()
            return action

    def train(self) -> Tuple[float, float]:
        batch = self.replay_buffer.sample(self.batch_size)
        if self.algo == RLAlgorithm.QLearning.value:
            next_q = self.target_model(batch.s1).max(1)[0]
        elif self.algo == RLAlgorithm.SARSA.value:
            next_q = self.target_model(batch.s1). \
                gather(1, batch.a.unsqueeze(1)).squeeze(1)
        elif self.algo == RLAlgorithm.EXPECTED_SARSA.value:
            next_q = torch.sum(self.target_model(batch.s1), axis=1)
        expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        q = self.model(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)

        # loss = 0.5 * (expected_q.detach() - q).pow(2).mean()
        loss = self.loss_func(expected_q.detach(), q)
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        # for param in self.model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # soft update rule
        if self.train_count != 0 and self.update_freq % self.train_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return float(loss.detach().cpu().numpy()), float(
            np.mean(q.detach().cpu().numpy()))

    def generate_n_step_q(self):
        """transition tuple: (s0, a, s1, r, done)"""
        transitions = self.replay_buffer.n_step_memory
        reward, next_observation, done = transitions[-1][-3:]

        for _, _, rew, next_obs, do in reversed(list(transitions)[: -1]):
            reward = self.gamma * reward * (1 - do) + rew
            next_observation, done = (next_obs, do) if do else (
                next_observation, done)
        return reward, next_observation, done

    def interact(self, num_steps: int, episode: int) -> Tuple[float, int]:
        """use agent to interact with environment by making actions based on
        optimal policy to obtain cumulative rewards"""
        observation = self.env.reset()
        observation = np.expand_dims(observation, 0)
        cr = 0
        action = self.get_action(observation=observation)
        for t in range(num_steps):
            next_observation, reward, done, info = self.env.step(action)
            next_observation = np.expand_dims(next_observation, 0)
            cr += reward

            # if we use n step
            if self.n_step > 0:
                if t % self.n_step == 0 and t != 0:
                    reward, next_observation, done = self.generate_n_step_q()

            # store into experience replay buffer and sample batch of
            # transitions to estimate the q-values and train on losses
            transition = Transition(s0=observation, a=action,
                                    s1=next_observation, r=reward, done=done)
            self.replay_buffer.push(transition)
            if len(self.replay_buffer.memory) > self.batch_size:
                loss, q = self.train()
                self.writer.add_scalar(tag='Training/Q-loss', scalar_value=loss,
                                       global_step=self.train_count)
                self.writer.add_scalar(tag='Training/Q-Value', scalar_value=q,
                                       global_step=self.train_count)
                self.train_count += 1
                print(
                    f'episode {episode} | step {t} | loss {loss} | q-value {q}')
            observation = next_observation

            # last step reached in episode
            # update the target network
            if done:
                num_steps = t + 1
                print(f'episode {episode} finished at {num_steps} steps')
                return cr, num_steps
            else:
                action = self.get_action(observation=observation)