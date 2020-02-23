import torch.nn as nn
import torch
from typing import Dict, List, Tuple


# # define agent
# def agent(states, actions):
#     """Simple Deep Neural Network."""
#     model = Sequential()
#     model.add(Flatten(input_shape=(1, states)))
#     model.add(Dense(16))
#     model.add(Activation('relu'))
#     model.add(Dense(16))
#     model.add(Activation('relu'))
#     model.add(Dense(16))
#     model.add(Activation('relu'))
#     model.add(Dense(actions))
#     model.add(Activation('linear'))
#     return model
#
#
# model = agent(states, actions)
#
# vgg_in_channels = 3
#
# # define layers for VGG, 'M' for max pooling
# vgg_config = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
#               512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
#               'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
#               512, 512, 'M', 512, 512, 512, 512, 'M'],
#     'VGGAttention': [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M',
#                      512, 512, 512, 'M', 512, 'M', 512, 'M'],
# }
#

class LinearFeedForward(nn.Module):
    def __init__(self, nn_config: List):
        super(LinearFeedForward, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(nn_config['num_states'], nn_config['units']),
            nn.ReLU(),
            nn.Linear(nn_config['units'], nn_config['num_actions'])
        )

    def forward(self, x):
        l1 = self.l1(x)
        return l1


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
