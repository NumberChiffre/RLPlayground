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
from RLPlayground.models.model import TorchModel
from RLPlayground.models.replay import Replay
from RLPlayground.utils.data_structures import Transition, TargetUpdate, \
    ReplayType
from RLPlayground.utils.registration import Registrable


@Agent.register('DeepTDAgent')
class DeepTDAgent(Agent, Registrable):
    def __init__(self,
                 env: Env,
                 agent_cfg: Dict):
        """

        :param env: gym environment used for Experiment
        :param agent_cfg: config file for given agent
        """
        super().__init__()
        self.env = env
        self.epochs, self.total_steps = 0, 0
        self.episodic_result = dict()
        self.episodic_result['Training/Q-Loss'] = []
        self.episodic_result['Training/Mean-Q-Value-Action'] = []
        self.episodic_result['Training/Mean-Q-Value-Opposite-Action'] = []
        self.episodic_result['value_net_params'] = []

        # specs for RL agent
        self.eps = agent_cfg['eps']
        if agent_cfg['use_eps_decay']:
            self.use_eps_decay = agent_cfg['use_eps_decay']
            self.eps_decay = agent_cfg['eps_decay']
            self.eps_min = agent_cfg['eps_min']
        self.gamma = agent_cfg['gamma']
        self.update_type = agent_cfg['update_type']
        if self.update_type == TargetUpdate.SOFT.value:
            self.tau = agent_cfg['tau']
        self.update_freq = agent_cfg['update_freq']
        self.warm_up_freq = agent_cfg['warm_up_freq']
        self.use_grad_clipping = agent_cfg['use_grad_clipping']
        self.grad_clipping = agent_cfg['grad_clipping']
        self.lr = agent_cfg['lr']
        self.batch_size = agent_cfg['batch_size']
        self.seed = agent_cfg['seed']
        self.params = vars(self).copy()

        # details experience replay
        self.replay_buffer = Replay.build(
            type=agent_cfg['experience_replay']['type'],
            params=agent_cfg['experience_replay']['params'])

        # details for the NN model
        agent_cfg['model']['seed'] = self.seed
        use_cuda = agent_cfg['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.value_net = TorchModel.build(type=agent_cfg['model']['type'],
                                          params=agent_cfg['model']['params'])
        self.target_net = TorchModel.build(type=agent_cfg['model']['type'],
                                           params=agent_cfg['model']['params'])
        self.value_net.to_device(self.device)
        self.target_net.to_device(self.device)
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
    def from_params(cls, env: Env, params: Dict):
        return cls(env, **params)

    @torch.no_grad()
    def get_action(self, observation) -> int:
        """either take greedy action or explore with epsilon rate"""
        if np.random.random() < self.eps:
            # return self.env.action_space.sample()
            return np.random.randint(self.env.action_space.n)
        else:
            state = torch.FloatTensor(observation).to(self.device)
            return self.value_net(state).max(1)[1].data[0].item()
        # observation = torch.FloatTensor(observation).to(self.device)
        # dist = get_epsilon_dist(eps=self.eps, env=self.env,
        #                         model=self.value_net, observation=observation)
        # return dist.sample().item()

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

        if self.use_eps_decay:
            # self.eps = get_epsilon(eps_start=self.eps, eps_final=self.eps_min,
            #                        eps_decay=self.eps_decay, t=1)
            if self.eps >= self.eps_min:
                self.eps *= self.eps_decay

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

    def interact(self, num_steps: int) -> Generator:
        """use agent to interact with environment by making actions based on
        optimal policy to obtain cumulative rewards"""
        cr, t = 0, 0
        done = False
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
            self.replay_buffer.push(transition)

            # train policy network and update target network
            # update epsilon decay, more starting exploration
            if self.total_steps >= self.warm_up_freq:
                self.training_update()

            observation = next_observation
            action = self.get_action(observation=observation)
            t += 1

        # TODO: pause training and use eval with generator, need to add eval!
        yield {
            'cum_reward': cr,
            'time_to_solve': t,
            'episode_time': time.time() - start,
        }


@Agent.register('DQNAgent')
class DQNAgent(DeepTDAgent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)
        self.use_double = agent_cfg['use_double']

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self):
        # handle different replay types
        if self.replay_buffer.replay_type == ReplayType.EXPERIENCE_REPLAY.value:
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
        elif self.replay_buffer.replay_type == \
                ReplayType.PRIORITIZED_EXPERIENCE_REPLAY.value:
            batch, weights = self.replay_buffer.sample(
                batch_size=self.batch_size)

        # handle different DQN
        if self.use_double:
            next_q = self.target_net(batch.s1).max(1)[0].detach()
        else:
            next_q_actions = torch.max(self.value_net(batch.s1), dim=1)[1]
            next_q = self.target_net(batch.s1).gather(1,
                                                      next_q_actions.unsqueeze(
                                                          1)).squeeze(1)

        # expected Q and Q using value net
        self.q = self.value_net(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)
        with torch.no_grad():
            self.q_ = self.value_net(batch.s0).gather(1,
                                                      1 - batch.a.unsqueeze(
                                                          1)).squeeze(1)
            expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        if self.replay_buffer.replay_type == ReplayType.EXPERIENCE_REPLAY.value:
            self.loss = self.loss_func(expected_q, self.q)
        elif self.replay_buffer.replay_type == ReplayType. \
                PRIORITIZED_EXPERIENCE_REPLAY.value:
            self.loss = self.loss_func(expected_q, self.q) * torch.FloatTensor(
                weights)
        # torchviz.make_dot(self.loss).render('loss')
        self.loss = self.loss.mean()
        self.optimizer.zero_grad()
        self.loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        if self.use_grad_clipping:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clipping)

        # update replay buffer..
        if self.replay_buffer.replay_type == ReplayType. \
                PRIORITIZED_EXPERIENCE_REPLAY.value:
            # less memory used
            with torch.no_grad():
                abs_td_error = torch.abs(expected_q - self.q).cpu().numpy() + \
                               self.replay_buffer.non_zero_variant
            self.replay_buffer.update_priorities(losses=abs_td_error)
        self.optimizer.step()


@Agent.register('DeepSarsaAgent')
class DeepSarsaAgent(DeepTDAgent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self):
        # handle different replay types
        if self.replay_buffer.replay_type == ReplayType.EXPERIENCE_REPLAY.value:
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
        elif self.replay_buffer.replay_type == \
                ReplayType.PRIORITIZED_EXPERIENCE_REPLAY.value:
            batch, weights = self.replay_buffer.sample(
                batch_size=self.batch_size)

        next_q = self.target_net(batch.s1).gather(1,
                                                  batch.a.unsqueeze(
                                                      1)).squeeze(
            1).detach()

        # expected Q and Q using value net
        self.q = self.value_net(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)
        with torch.no_grad():
            self.q_ = self.value_net(batch.s0).gather(1,
                                                      1 - batch.a.unsqueeze(
                                                          1)).squeeze(1)
            expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        if self.replay_buffer.replay_type == ReplayType.EXPERIENCE_REPLAY.value:
            self.loss = self.loss_func(expected_q, self.q)
        elif self.replay_buffer.replay_type == ReplayType. \
                PRIORITIZED_EXPERIENCE_REPLAY.value:
            self.loss = self.loss_func(expected_q, self.q) * torch.FloatTensor(
                weights)
        # torchviz.make_dot(self.loss).render('loss')
        self.loss = self.loss.mean()
        self.optimizer.zero_grad()
        self.loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        if self.use_grad_clipping:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clipping)

        # update replay buffer..
        if self.replay_buffer.replay_type == ReplayType. \
                PRIORITIZED_EXPERIENCE_REPLAY.value:
            # less memory used
            with torch.no_grad():
                abs_td_error = torch.abs(expected_q - self.q).cpu().numpy() + \
                               self.replay_buffer.non_zero_variant
            self.replay_buffer.update_priorities(losses=abs_td_error)
        self.optimizer.step()


@Agent.register('DeepExpectedSarsaAgent')
class DeepExpectedSarsaAgent(DeepTDAgent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self):
        # handle different replay types
        if self.replay_buffer.replay_type == ReplayType.EXPERIENCE_REPLAY.value:
            batch = self.replay_buffer.sample(batch_size=self.batch_size)
        elif self.replay_buffer.replay_type == \
                ReplayType.PRIORITIZED_EXPERIENCE_REPLAY.value:
            batch, weights = self.replay_buffer.sample(
                batch_size=self.batch_size)
        prob_dist = get_epsilon_dist(eps=self.eps, env=self.env,
                                     model=self.value_net, observation=batch.s1)
        next_q = torch.sum(self.target_net(batch.s1) * prob_dist.probs,
                           axis=1).detach()

        # expected Q and Q using value net
        self.q = self.value_net(batch.s0).gather(1, batch.a.unsqueeze(
            1)).squeeze(1)
        with torch.no_grad():
            self.q_ = self.value_net(batch.s0).gather(1,
                                                      1 - batch.a.unsqueeze(
                                                          1)).squeeze(1)
            expected_q = batch.r + self.gamma * (1 - batch.done) * next_q
        if self.replay_buffer.replay_type == ReplayType.EXPERIENCE_REPLAY.value:
            self.loss = self.loss_func(expected_q, self.q)
        elif self.replay_buffer.replay_type == ReplayType. \
                PRIORITIZED_EXPERIENCE_REPLAY.value:
            self.loss = self.loss_func(expected_q, self.q) * torch.FloatTensor(
                weights)
        # torchviz.make_dot(self.loss).render('loss')
        self.loss = self.loss.mean()
        self.optimizer.zero_grad()
        self.loss.backward()

        # gradient clipping to avoid loss divergence based on DeepMind's DQN in
        # 2015, where the author clipped the gradient within [-1, 1]
        if self.use_grad_clipping:
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clipping)

        # update replay buffer..
        if self.replay_buffer.replay_type == ReplayType. \
                PRIORITIZED_EXPERIENCE_REPLAY.value:
            # less memory used
            with torch.no_grad():
                abs_td_error = torch.abs(expected_q - self.q).cpu().numpy() + \
                               self.replay_buffer.non_zero_variant
            self.replay_buffer.update_priorities(losses=abs_td_error)
        self.optimizer.step()
