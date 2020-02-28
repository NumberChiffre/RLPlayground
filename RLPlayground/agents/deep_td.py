import time
import numpy as np
from typing import Dict, Generator, Tuple
from gym import Env
import torch.optim as optim
import torch.nn as nn
import torch
import torchviz

from RLPlayground.utils.utils import torch_argmax_mask, get_epsilon_dist, \
    get_epsilon, soft_update, hard_update
from RLPlayground.agents.agent import Agent
from RLPlayground.models.feed_forward import LinearFCBody
from RLPlayground.models.replay import ReplayBuffer
from RLPlayground.utils.data_structures import Transition, TargetUpdate
from RLPlayground.utils.registration import Registrable


class DeepTDAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        """

        :param env: gym environment used for Experiment
        :param agent_cfg: config file for given agent
        """
        super(DeepTDAgent, self).__init__(env=env, agent_cfg=agent_cfg)
        self.epochs, self.total_steps = 0, 0

        # specs for RL agent
        self.eps = agent_cfg['eps']
        if agent_cfg['use_eps_decay']:
            self.use_eps_decay = agent_cfg['use_eps_decay']
            self.eps_decay = agent_cfg['eps_decay']
            self.eps_min = agent_cfg['eps_min']
        self.gamma = agent_cfg['gamma']
        self.n_step = agent_cfg['n_step']
        self.update_type = agent_cfg['update_type']
        if self.update_type == TargetUpdate.SOFT.value:
            self.tau = agent_cfg['tau']
        self.update_freq = agent_cfg['update_freq']
        self.warm_up_freq = agent_cfg['warm_up_freq']
        self.use_grad_clipping = agent_cfg['use_grad_clipping']
        self.lr = agent_cfg['lr']
        self.batch_size = agent_cfg['batch_size']
        self.seed = agent_cfg['seed']
        self.params = vars(self).copy()

        # details for the NN model + experience replay
        self.replay_buffer = ReplayBuffer(
            capacity=agent_cfg['replay_capacity'], n_step=self.n_step)
        self.value_net = LinearFCBody(seed=self.seed,
                                      state_dim=
                                      self.env.observation_space.shape[0],
                                      action_dim=self.env.action_space.n,
                                      hidden_units=tuple(
                                          agent_cfg['nn_hidden_units'])
                                      )
        self.target_net = LinearFCBody(seed=self.seed,
                                       state_dim=
                                       self.env.observation_space.shape[0],
                                       action_dim=self.env.action_space.n,
                                       hidden_units=tuple(
                                           agent_cfg['nn_hidden_units'])
                                       )
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.value_net.parameters(), self.lr)

        # Huber loss acts like the mean squared error when the error is small,
        # but like the mean absolute error when the error is large
        # this makes it more robust to outliers when the estimates of Q
        # are very noisy. It is calculated over a batch of transitions
        # sampled from the replay memory:
        self.loss_func = nn.SmoothL1Loss()

    @classmethod
    def build(cls, type: str, env: Env, params: Dict):
        agent = cls.by_name(type)
        return agent.from_params(env, params)

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        raise NotImplementedError('from_params not implemented in DeepTDAgent')

    @torch.no_grad()
    def get_action(self, observation) -> int:
        """either take greedy action or explore with epsilon rate"""
        # if np.random.random() < self.eps:
        #     # return self.env.action_space.sample()
        #     return np.random.randint(self.env.action_space.n)
        # else:
        #     state = torch.FloatTensor(observation)
        #     return self.value_net(state).max(1)[1].data[0].item()
        observation = torch.FloatTensor(observation)
        dist = get_epsilon_dist(eps=self.eps, env=self.env,
                                model=self.value_net, observation=observation)
        return dist.sample().item()

    def train(self):
        raise NotImplementedError('DeepTD Agent requires a train() method')

    def training_update(self):
        """backprop loss, target network, epsilon, and result updates"""
        self.epochs += 1
        self.train()

        # hard update
        if self.update_type == TargetUpdate.HARD.value:
            if self.epochs != 0 and self.update_freq % self.epochs == 0:
                self.target_net.load_state_dict(
                    self.value_net.state_dict())
        elif self.update_type == TargetUpdate.SOFT.value:
            soft_update(value_net=self.value_net, target_net=self.target_net,
                        tau=self.tau)

        # save episodic results
        self.episodic_result['Training/Q-Loss'].append(
            float(self.loss.detach().cpu().numpy()))
        self.episodic_result['Training/Mean-Q-Value-Action'].append(
            float(np.mean(self.q.detach().cpu().numpy())))
        self.episodic_result[
            'Training/Mean-Q-Value-Opposite-Action'].append(
            float(np.mean(self.q_.detach().cpu().numpy())))
        # self.episodic_result['value_net_params'].append(
        #     self.value_net.named_parameters())

    def generate_n_step_q(self) -> Transition:
        """transition tuple: (s0, a, s1, r, done)"""
        transitions = self.replay_buffer.n_step_memory
        reward, next_observation, done = transitions[-1][-3:]

        for i in range(len(transitions) - 1):
            reward = self.gamma * reward * (1 - transitions[i].done) + \
                     transitions[i].r
            next_observation, done = (transitions[i].s1, transitions[i].done) \
                if transitions[i].done else (next_observation, done)
        observation, action = transitions[0][:2]
        return Transition(s0=observation, a=action, r=reward,
                          s1=next_observation, done=done)

    def interact(self, num_steps: int, episode: int) -> Generator:
        """use agent to interact with environment by making actions based on
        optimal policy to obtain cumulative rewards"""
        cr, t = 0, 0
        done = False
        self.episodic_result = dict()
        self.episodic_result['Training/Q-Loss'] = []
        self.episodic_result['Training/Mean-Q-Value-Action'] = []
        self.episodic_result['Training/Mean-Q-Value-Opposite-Action'] = []
        self.episodic_result['value_net_params'] = []

        start = time.time()
        observation = self.env.reset()
        observation = np.expand_dims(observation, 0)
        action = self.get_action(observation=observation)

        while not done and t < num_steps:
            self.total_steps += 1
            next_observation, reward, done, info = self.env.step(action)
            next_observation = np.expand_dims(next_observation, 0)
            cr += reward

            # store into experience replay buffer and sample batch of
            # transitions to estimate the q-values and train on losses
            transition = Transition(s0=observation, a=action, r=reward,
                                    s1=next_observation, done=done)
            if self.n_step > 0:
                if t % self.n_step == 0 and t != 0:
                    transition = self.generate_n_step_q()
            self.replay_buffer.push(transition)

            # train policy network and update target network
            # update epsilon decay, more starting exploration
            if self.total_steps >= self.warm_up_freq:
                self.training_update()

            observation = next_observation
            action = self.get_action(observation=observation)
            if self.use_eps_decay:
                # self.eps = get_epsilon(eps_start=self.eps, eps_final=self.eps_min,
                #                        eps_decay=self.eps_decay, t=1)
                if self.eps >= self.eps_min:
                    self.eps *= self.eps_decay
            t += 1

        # TODO: pause training and use eval with generator, need to add eval!
        yield {
            'cum_reward': cr,
            'time_to_solve': t,
            'episode_time': time.time() - start,
        }


@DeepTDAgent.register('DQNAgent')
class DQNAgent(DeepTDAgent):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)
        self.use_double = self.agent_cfg['use_double']

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        if self.use_double:
            next_q = self.target_net(batch.s1).max(1)[0].detach()
        else:
            next_q_actions = torch.max(self.value_net(batch.s1), dim=1)[1]
            next_q = self.target_net(batch.s1).gather(1,
                                                      next_q_actions.unsqueeze(
                                                          1)).squeeze(1)

        # expected Q and Q using value net
        expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        self.q = self.value_net(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)
        with torch.no_grad():
            self.q_ = self.value_net(batch.s0).gather(1,
                                                      1 - batch.a.unsqueeze(
                                                          1)).squeeze(1)

        self.loss = self.loss_func(expected_q.detach(), self.q)
        # torchviz.make_dot(loss).render('loss')
        self.optimizer.zero_grad()
        self.loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        if self.use_grad_clipping:
            for param in self.value_net.parameters():
                param.grad.data.clamp_(-1, 1)
            # nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()


@DeepTDAgent.register('DeepSarsaAgent')
class DeepSarsaAgent(DeepTDAgent):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        next_q = self.target_net(batch.s1).gather(1,
                                                  batch.a.unsqueeze(
                                                      1)).squeeze(
            1).detach()

        # expected Q and Q using value net
        expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        self.q = self.value_net(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)
        with torch.no_grad():
            self.q_ = self.value_net(batch.s0).gather(1,
                                                      1 - batch.a.unsqueeze(
                                                          1)).squeeze(1)

        self.loss = self.loss_func(expected_q.detach(), self.q)
        # torchviz.make_dot(loss).render('loss')
        self.optimizer.zero_grad()
        self.loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        if self.use_grad_clipping:
            for param in self.value_net.parameters():
                param.grad.data.clamp_(-1, 1)
            # nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()


@DeepTDAgent.register('DeepExpectedSarsaAgent')
class DeepExpectedSarsaAgent(DeepTDAgent):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        prob_dist = get_epsilon_dist(eps=self.eps, env=self.env,
                                     model=self.value_net, observation=batch.s1)
        next_q = torch.sum(self.target_net(batch.s1) * prob_dist.probs,
                           axis=1).detach()

        # expected Q and Q using value net
        expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        self.q = self.value_net(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)
        with torch.no_grad():
            self.q_ = self.value_net(batch.s0).gather(1,
                                                      1 - batch.a.unsqueeze(
                                                          1)).squeeze(1)

        self.loss = self.loss_func(expected_q.detach(), self.q)
        # torchviz.make_dot(loss).render('loss')
        self.optimizer.zero_grad()
        self.loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        if self.use_grad_clipping:
            for param in self.value_net.parameters():
                param.grad.data.clamp_(-1, 1)
            # nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()
