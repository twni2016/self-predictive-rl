import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import numpy as np
import utils
from utils import logger


class DetEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, latent_dims),
        )

        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, x):
        loc = self.encoder(x)
        return utils.Dirac(loc)


class StoEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dims, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 2 * latent_dims),
        )

        self.std_min = 0.1
        self.std_max = 10.0
        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, x):
        x = self.encoder(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = 30 * torch.tanh(mean / 30)  # [-30, 30]
        std = self.std_max - F.softplus(self.std_max - std)
        std = self.std_min + F.softplus(std - self.std_min)  # (-std_min, std_max)
        return td.independent.Independent(td.Normal(mean, std), 1)


class DetModel(nn.Module):
    def __init__(
        self, latent_dims, action_dims, hidden_dims, num_layers=2, obs_dims=None
    ):
        super().__init__()
        self.latent_dims = latent_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.obs_dims = obs_dims

        self.model = self._build_model()
        self.apply(utils.weight_init)
        logger.log(self)

    def _build_model(self):
        model = [nn.Linear(self.action_dims + self.latent_dims, self.hidden_dims)]
        model += [nn.ELU()]
        for i in range(self.num_layers - 1):
            model += [nn.Linear(self.hidden_dims, self.hidden_dims)]
            model += [nn.ELU()]
        model += [
            nn.Linear(
                self.hidden_dims,
                self.latent_dims if self.obs_dims is None else self.obs_dims,
            )
        ]
        return nn.Sequential(*model)

    def forward(self, z, action):
        x = torch.cat([z, action], axis=-1)
        loc = self.model(x)
        return utils.Dirac(loc)


class StoModel(nn.Module):
    def __init__(
        self, latent_dims, action_dims, hidden_dims, num_layers=2, obs_dims=None
    ):
        super().__init__()
        self.latent_dims = latent_dims
        self.action_dims = action_dims
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.obs_dims = obs_dims

        self.std_min = 0.1
        self.std_max = 10.0
        self.model = self._build_model()
        self.apply(utils.weight_init)
        logger.log(self)

    def _build_model(self):
        model = [nn.Linear(self.action_dims + self.latent_dims, self.hidden_dims)]
        model += [nn.ELU()]
        for i in range(self.num_layers - 1):
            model += [nn.Linear(self.hidden_dims, self.hidden_dims)]
            model += [nn.ELU()]
        model += [
            nn.Linear(
                self.hidden_dims,
                2 * self.latent_dims if self.obs_dims is None else 2 * self.obs_dims,
            )
        ]
        return nn.Sequential(*model)

    def forward(self, z, action):
        x = torch.cat([z, action], axis=-1)
        x = self.model(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = 30 * torch.tanh(mean / 30)
        std = self.std_max - F.softplus(self.std_max - std)
        std = self.std_min + F.softplus(std - self.std_min)
        return td.independent.Independent(td.Normal(mean, std), 1)


class RewardPrior(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_dims):
        super().__init__()
        self.reward = nn.Sequential(
            nn.Linear(latent_dims + action_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 1),
        )
        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, z, a):
        z_a = torch.cat([z, a], -1)
        reward = self.reward(z_a)
        return reward


class Discriminator(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_dims):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2 * latent_dims + action_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 2),
        )
        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, z, a, z_next):
        x = torch.cat([z, a, z_next], -1)
        logits = self.classifier(x)
        return logits

    def get_reward(self, z, a, z_next):
        x = torch.cat([z, a, z_next], -1)
        logits = self.classifier(x)
        reward = torch.sub(logits[..., 1], logits[..., 0])
        return reward.unsqueeze(-1)


class Critic(nn.Module):
    def __init__(self, latent_dims, hidden_dims, action_shape):
        super().__init__()
        self.Q1 = nn.Sequential(
            nn.Linear(latent_dims + action_shape, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(latent_dims + action_shape, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ELU(),
            nn.Linear(hidden_dims, 1),
        )

        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, x, a):
        x_a = torch.cat([x, a], -1)
        q1 = self.Q1(x_a)
        q2 = self.Q2(x_a)
        return q1, q2


class Actor(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_shape, low, high):
        super().__init__()
        self.low = low
        self.high = high
        self.fc1 = nn.Linear(input_shape, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.mean = nn.Linear(hidden_dims, output_shape)
        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, x, std):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        std = torch.ones_like(mean) * std
        dist = utils.TruncatedNormal(mean, std, self.low, self.high)
        return dist


class StoActor(nn.Module):
    def __init__(self, input_shape, hidden_dims, output_shape, low, high):
        super().__init__()
        self.low = low
        self.high = high
        self.fc1 = nn.Linear(input_shape, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 2 * output_shape)
        self.std_min = np.exp(-5)
        self.std_max = np.exp(2)
        self.apply(utils.weight_init)
        logger.log(self)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc3(x)
        mean, std = torch.chunk(x, 2, -1)
        mean = torch.tanh(mean)
        std = self.std_max - F.softplus(self.std_max - std)
        std = self.std_min + F.softplus(std - self.std_min)
        dist = utils.TruncatedNormal(mean, std, self.low, self.high)
        return dist
