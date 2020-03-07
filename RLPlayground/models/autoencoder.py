import torch
import torch.nn as nn

from RLPlayground.models.model import TorchModel


@TorchModel.register('AutoEncoder')
class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 z_dim: int):
        super(AutoEncoder, self).__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = self.enc(x)
        dec = self.dec(hidden)
        return dec


@TorchModel.register('VAE')
class VAE(nn.Module):
    def __init__(self,
                 enc: nn.Module,
                 dec: nn.Module):
        super(VAE, self).__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)
        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        if self.training:
            std = torch.exp(z_var * 0.5)
            eps = torch.randn_like(std)
            x_sample = z_mu + eps * std
        else:
            x_sample = z_mu
        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var


class VAEEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 z_dim: int):
        super(VAEEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim]
        hidden = nn.Relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]
        return z_mu, z_var


class VAEDecoder(nn.Module):
    def __init__(self,
                 z_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super(VAEDecoder, self).__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim]
        hidden = nn.Relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]
        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]
        return predicted
