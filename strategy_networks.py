import torch
import numpy as np
import copy
from transformer_for_ts import Transformer

from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi


env = rl_env.HanabiEnv(
        config={
            "colors":
                1,
            "ranks":
                2,
            "players":
                2,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })

d_model = env.vectorized_observation_shape()[0]
q = 2
v = 2
h = 2
N = 4
d_ff = 64
dropout = 0

d_input = env.vectorized_observation_shape()[0]
d_output = env.num_moves()


class StrategyNetworks:

    def __init__(self, n_players=2, net_arr=None, obs_max_length=30, obs_size=113, action_size=9):

        self.obs_max_length = obs_max_length
        self.obs_size = obs_size
        self.action_size = action_size

        if net_arr is None:

            self.net_arr = [None] * n_players

            for i in range(n_players):
                self.net_arr[i] = Transformer(self.obs_size, self.obs_size, self.action_size, q, v, h, N, d_ff, dropout=dropout, compute_value=1).cuda()
        else:
            self.net_arr = net_arr

    def get_values(self, obs_h, p):

        self.net_arr[p].eval()

        with torch.no_grad():
            x = torch.zeros(1, self.obs_max_length, self.obs_size).cuda().float()
            obs_h = torch.tensor(obs_h).view(1, -1, self.obs_size).cuda().float()
            length = torch.tensor([obs_h.shape[1]])
            # print(length)
            x[0, :length[0]] = obs_h
            output = self.net_arr[p](x, length)[0]
            softmax_values = torch.softmax(output, dim=0).cpu().numpy()
            values = output.cpu().numpy()
        clip_values = np.clip(values, a_min=0, a_max=None)
        return softmax_values, clip_values


PolicyNet = Transformer(d_input, d_model, d_output, q, v, h, N, d_ff, dropout=dropout, compute_value=0).cuda()


def loss_fn(y_pred, t, y):
    loss = t.clip(min = 0, max = 100).view(-1, 1) * (y_pred - y) ** 2

    return torch.sum(loss) / len(t)
