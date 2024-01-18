import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class SeqEncoder(nn.Module):
    """
    rho in AIS, phi in RL literature.
    Deterministic model z = phi(h)
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(SeqEncoder, self).__init__()
        input_ndims = (
            num_obs + num_actions + 1
        )  # including reward, but it is uninformative
        self.AIS_state_size = AIS_state_size
        self.fc1 = nn.Linear(input_ndims, AIS_state_size)
        self.fc2 = nn.Linear(AIS_state_size, AIS_state_size)
        self.lstm = nn.LSTM(AIS_state_size, AIS_state_size, batch_first=True)

        self.apply(weights_init_)

    def get_initial_hidden(self, batch_size, device):  # TODO:
        return (
            torch.zeros(1, batch_size, self.AIS_state_size).to(device),
            torch.zeros(1, batch_size, self.AIS_state_size).to(device),
        )

    def forward(
        self,
        x,
        batch_size,
        hidden,
        device,
        batch_lengths,
        pack_sequence=True,
    ):
        if hidden == None:
            hidden = self.get_initial_hidden(batch_size, device)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        if pack_sequence is True:
            x = pack_padded_sequence(
                x, batch_lengths, batch_first=True, enforce_sorted=False
            )
            # print('packed',x.data.shape)
        x, hidden = self.lstm(x, hidden)
        return x, hidden


class LatentModel(nn.Module):
    """
    psi in AIS, P_theta in RL.
    Deterministic latent transition models.
    E[o' | z, a] or E[z' | z, a], depends on num_obs
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(LatentModel, self).__init__()
        input_ndims = AIS_state_size + num_actions
        self.fc1_d = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_d = nn.Linear(AIS_state_size // 2, num_obs)

        self.apply(weights_init_)

    def forward(self, x):
        x_d = F.elu(self.fc1_d(x))
        obs = self.fc2_d(x_d)
        return obs


class AISModel(nn.Module):
    """
    psi in AIS, P_theta in RL.
    Deterministic transition and reward models.
    E[o' | z, a] or E[z' | z, a] AND E[r | z, a]
    """

    def __init__(self, num_obs, num_actions, AIS_state_size):
        super(AISModel, self).__init__()
        input_ndims = AIS_state_size + num_actions
        self.fc1_d = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_d = nn.Linear(AIS_state_size // 2, num_obs)
        self.fc1_r = nn.Linear(input_ndims, AIS_state_size // 2)
        self.fc2_r = nn.Linear(AIS_state_size // 2, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x_d = F.elu(self.fc1_d(x))
        obs = self.fc2_d(x_d)
        x_r = F.elu(self.fc1_r(x))
        rew = self.fc2_r(x_r)
        return obs, rew


class QNetwork_discrete(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork_discrete, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.elu(self.linear1(state))
        x1 = F.elu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


def convert_int_to_onehot(value, num_values):
    onehot = torch.zeros(num_values)
    if value >= 0:  # ignore negative index
        onehot[int(value)] = 1.0
    return onehot


def weights_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        torch.nn.init.constant_(m.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
