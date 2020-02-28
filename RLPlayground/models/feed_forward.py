import torch.nn as nn
import torch


def layer_init(layer, w_scale: int = 1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_units: tuple = (64, 64),
                 gate: nn.functional = nn.functional.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(
            nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi


class LinearFCBody(nn.Module):
    def __init__(self, seed: int, state_dim: int, action_dim: int,
                 hidden_units: tuple = (64, 64),
                 gate: nn.PReLU = nn.PReLU):
        super(LinearFCBody, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size1),
            # nn.BatchNorm1d(hidden_size1),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size1, hidden_size2),
            # nn.BatchNorm1d(hidden_size2),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(hidden_size2, self.action_dim)
        # self.gate = gate

    def forward(self, observation):
        x = self.fc1(observation)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TwoLayerFCBody(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_units: tuple = (64, 64),
                 gate: nn.functional = nn.functional.relu):
        super(TwoLayerFCBody, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(
            nn.Linear(hidden_size2, action_dim))
        self.gate = gate

    def forward(self, x):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(x))
        return phi


class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim: int, action_dim: int,
                 hidden_units: tuple = (64, 64),
                 gate: nn.functional = nn.functional.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi
