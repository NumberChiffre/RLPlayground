from typing import Dict, Generator, List, Tuple
import torch
import torch.nn as nn

from RLPlayground.utils.registration import Registrable


class TorchModel(nn.Module, Registrable):
    @classmethod
    def build(cls, type: str, params: Dict):
        model = cls.by_name(type)
        return model.from_params(params)

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def forward(self, *input):
        raise NotImplementedError()

    def to_device(self, device):
        self.to(device)
        self.device = device


@TorchModel.register('LinearFCBody')
class LinearFCBody(TorchModel):
    def __init__(self,
                 seed: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_units: List = [64, 64],
                 gate: nn.ReLU = nn.ReLU):
        super(LinearFCBody, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_unit_1, hidden_unit_2 = hidden_units[0], hidden_units[1]
        self.fc1 = nn.Sequential(
            nn.Linear(self.state_dim, hidden_unit_1),
            # nn.BatchNorm1d(hidden_size1),
            gate()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_unit_1, hidden_unit_2),
            # nn.BatchNorm1d(hidden_size2),
            gate()
        )
        self.fc3 = nn.Linear(hidden_unit_2, self.action_dim)

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


@TorchModel.register('BasicRNN')
class BasicRNN(TorchModel):
    def __init__(self,
                 seed: int,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int,
                 dropout: float = 0,
                 bidirectional: bool = False):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.num_directions,
                         x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions,
                         x.size(0), self.hidden_dim).to(self.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


@TorchModel.register('Inception')
class Inception(TorchModel):
    def __init__(self,
                 in_channels: int,
                 gate: nn.ReLU):
        super().__init__()
        out_channels = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            # gate(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1)), )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)),
            # gate(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1)), )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1)),
            # gate(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1)), )

    @classmethod
    def from_params(cls, params: Dict):
        return cls(**params)

    def forward(self, x):
        # (batch_size, out_channels, timesteps, features),
        # timesteps=timesteps-kernel+1
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        outputs = [conv1, conv2, conv3]
        return torch.cat(outputs, 2)  # concatenate by timesteps

