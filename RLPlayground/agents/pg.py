import time
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from typing import Dict, Generator, List, Tuple, Union
from gym import Env

from RLPlayground.agents.agent import Agent
from RLPlayground.models.model import TorchModel
from RLPlayground.utils.data_structures import Transition
from RLPlayground.utils.utils import nested_d
from RLPlayground.utils.registration import Registrable


@Agent.register('REINFORCEAgent')
class REINFORCEAgent(Agent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__()
        self.env = env
        self.γ = agent_cfg['gamma']
        self.seed = agent_cfg['seed']
        agent_cfg['model']['seed'] = self.seed
        use_cuda = agent_cfg['use_gpu'] and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.value_net = TorchModel.build(
            type=agent_cfg['model']['type'],
            params=agent_cfg['model']['params'])
        self.lr = agent_cfg['lr']
        self.optimizer = optim.Adam(self.value_net.parameters(), self.lr)
        self.loss = nn.MSELoss()
        self.episodic_result = dict()

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    @torch.no_grad()
    def get_action(self, s0) -> int:
        state = torch.FloatTensor(s0).to(self.device)
        _, π = self.value_net.forward(state)
        dist = torch.distributions.Categorical(probs=π)
        return dist.sample().cpu().item()

    def train(self, trajectory: List[Transition]):
        s0s = torch.FloatTensor(
            [transition.s0 for transition in trajectory]).to(self.device)
        actions = torch.LongTensor(
            [transition.a for transition in trajectory]).view(-1, 1).to(
            self.device)
        rewards = torch.FloatTensor(
            [transition.r for transition in trajectory]).to(self.device)

        # compute discounted rewards
        G = [torch.sum(torch.FloatTensor(
            [self.γ ** i for i in range(rewards[j:].size(0))]) * rewards[j:])
             for j in range(rewards.size(0))]

        _, πs = self.value_net.forward(s0s)
        probs = torch.distributions.Categorical(probs=πs)

        actor_loss = (-probs.log_prob(actions.view(actions.size(0))).view(-1,
                                                                          1) *
                      torch.FloatTensor(G).view(-1, 1).to(
                          self.device)).sum()

        # decouple backpropagation for actor/critic
        total_loss = actor_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return np.mean(total_loss.detach().cpu().numpy())

    def learn(self, num_steps: int) -> Generator:
        done = False
        cr, t = 0, 0
        start = time.time()
        s0 = self.env.reset()
        trajectory = []
        while not done and t < num_steps:
            action = self.get_action(s0=s0)
            s1, r, done, info = self.env.step(action)
            transition = Transition(s0=s0, a=action, r=r, s1=s1, done=done)
            trajectory.append(transition)
            s0 = s1
            cr += r
            t += 1
        # train at the end of each episode using backpropagation for
        # policy/critic losses
        loss = self.train(trajectory=trajectory)
        yield {
            'cum_reward': cr,
            'time_to_solve': t,
            'loss': loss,
            'episode_time': time.time() - start,
        }


@Agent.register('A2CAgent')
class A2CAgent(REINFORCEAgent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)
        self.entropy_weight = agent_cfg['entropy_weight']

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self, trajectory: List[Transition]):
        s0s = torch.FloatTensor(
            [transition.s0 for transition in trajectory]).to(self.device)
        actions = torch.LongTensor(
            [transition.a for transition in trajectory]).view(-1, 1).to(
            self.device)
        rewards = torch.FloatTensor(
            [transition.r for transition in trajectory]).to(self.device)

        # compute discounted rewards
        G = [torch.sum(torch.FloatTensor(
            [self.γ ** i for i in range(rewards[j:].size(0))]) * rewards[j:])
             for j in range(rewards.size(0))]
        Q_targets = rewards.view(-1, 1) + torch.FloatTensor(G).view(-1, 1).to(
            self.device)

        Qs, πs = self.value_net.forward(s0s)
        probs = torch.distributions.Categorical(probs=πs)

        # compute entropy bonus
        entropy = torch.stack(
            [-torch.sum(π.mean() * torch.log(π)) for π in πs]).sum()

        # compute critic/policy loss
        critic_loss = self.loss(Qs, Q_targets.detach())
        advantage = Q_targets - Qs
        actor_loss = (-probs.log_prob(actions.view(actions.size(0))).view(-1,
                                                                          1) *
                      advantage.detach()).mean() - self.entropy_weight * entropy

        # decouple backpropagation for actor/critic
        total_loss = critic_loss + actor_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return np.mean(total_loss.detach().cpu().numpy())


# TODO: refactor this to have an actor-critic base class
@Agent.register('ACKTRAgent')
class ACKTRAgent(REINFORCEAgent, Registrable):
    def __init__(self, env: Env, agent_cfg: Dict):
        super().__init__(env, agent_cfg)
        self.value_loss_weight = agent_cfg['value_loss_weight']

    @classmethod
    def from_params(cls, env: Env, params: Dict):
        return cls(env, params)

    def train(self, trajectory: List[Transition]):
        s0s = torch.FloatTensor(
            [transition.s0 for transition in trajectory]).to(self.device)
        actions = torch.LongTensor(
            [transition.a for transition in trajectory]).view(-1, 1).to(
            self.device)
        rewards = torch.FloatTensor(
            [transition.r for transition in trajectory]).to(self.device)

        # compute discounted rewards
        G = [torch.sum(torch.FloatTensor(
            [self.γ ** i for i in range(rewards[j:].size(0))]) * rewards[j:])
             for j in range(rewards.size(0))]
        Q_targets = rewards.view(-1, 1) + torch.FloatTensor(G).view(-1, 1).to(
            self.device)

        Qs, πs = self.value_net.forward(s0s)
        probs = torch.distributions.Categorical(probs=πs)

        # compute entropy bonus
        entropy = torch.stack(
            [-torch.sum(π.mean() * torch.log(π)) for π in πs]).sum()

        # compute critic/policy loss
        critic_loss = self.loss(Qs, Q_targets.detach())
        advantage = Q_targets - Qs
        actor_loss = (-probs.log_prob(actions.view(actions.size(0))).view(-1,
                                                                          1) *
                      advantage.detach()).mean() - self.entropy_weight * entropy

        if self.optimizer.steps % self.optimizer.Ts == 0:
            self.value_net.zero_grad()
            pg_fisher_loss = probs.mean()
            Qs_noise = torch.randn(Qs.size()).to(self.device)
            Qs_sample = Qs + Qs_noise
            vf_fisher_loss = -(Qs - Qs_sample.detach()).pow(2).mean()
            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (critic_loss * self.value_loss_weight + actor_loss - entropy *
         self.entropy_weight).backward()
        self.optimizer.step()
        return critic_loss + actor_loss
