from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class ForwardDynamicNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: Union[int, List[int]],
        hidden_size: Tuple[int] = (512, 512),
        activiation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        layers = []
        for i, h in enumerate(hidden_size):
            if i == 0:
                if len(act_dim) > 1:
                    layers.append(nn.Linear(obs_dim + sum(act_dim), h))
                else:
                    layers.append(nn.Linear(obs_dim + act_dim, h))
            else:
                layers.append(nn.Linear(hidden_size[i - 1], h))
            layers.append(activiation())
        layers.append(nn.Linear(hidden_size[-1], obs_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if len(self.act_dim) > 1:   # MultiDiscrete
            act_onehot = torch.cat([F.one_hot(a, d) for a, d in zip(act.T, self.act_dim)], dim=-1)
        else:
            act_onehot = F.one_hot(act, self.act_dim)
        return obs + self.net(torch.cat([obs, act_onehot], dim=-1))
