import importlib
from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..common.scheduler import ConstantScheduler


class MultiCategorical(object):
    def __init__(
        self,
        logits,
        num_categories: List[int],
    ):
        split_logits = torch.split(logits, num_categories, dim=-1)
        self.multi_categoricals = [
            Categorical(logits=logits) for logits in split_logits
        ]
        
    def sample(self):
        return torch.stack([categorical.sample() for categorical in self.multi_categoricals], dim=-1)

    def log_prob(self, action):
        split_actions = torch.split(action, [1] * len(self.multi_categoricals), dim=-1)
        return sum(
            categorical.log_prob(action.squeeze(-1))
            for categorical, action in zip(self.multi_categoricals, split_actions)
        )
        
    def entropy(self):
        return sum(categorical.entropy() for categorical in self.multi_categoricals)
    
    def mode(self):
        return torch.stack(
            [categorical.probs.argmax(dim=-1) for categorical in self.multi_categoricals], dim=-1)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        if len(x) == 0:
            return
        batch_mean = np.mean(x, axis=0, dtype=np.float32)
        batch_var = np.var(x, axis=0, dtype=np.float32)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean, self.var, self.count = new_mean, new_var, new_count


def get_callable(name: str):
    module_name, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def clamp(input, min, max):
    # Currently torch.clamp() does not support tensor arguments for min/max
    # while torch.min() / max() does not support float arguments
    if isinstance(min, torch.Tensor) and isinstance(max, torch.Tensor):
        clipped = torch.max(torch.min(input, max), min)
    else:
        clipped = torch.clamp(input, min=min, max=max)
    return clipped


def calculate_gae(
    rewards, values, firsts, last_value, last_first, discount_gamma, gae_lambda, 
    use_proper_time_limit=False, bad=None, last_bad=None,
    ):
    # borrow implementation from OpenAI's baselines
    n_steps = len(rewards)
    advs = np.zeros_like(rewards)
    lastadv = 0
    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            nextnonterminal = 1.0 - last_first
            nextvalues = last_value
        else:
            nextnonterminal = 1.0 - firsts[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + discount_gamma * nextvalues * nextnonterminal - values[t]
        advs[t] = delta + discount_gamma * gae_lambda * nextnonterminal * lastadv
        if use_proper_time_limit:
            if t == n_steps - 1:
                advs[t] *= 1.0 - last_bad
            else:
                advs[t] *= 1.0 - bad[t + 1]
        lastadv = advs[t]
    return advs


def get_scheduler(value: Union[Dict, float]):
    if isinstance(value, float):
        scheduler = ConstantScheduler(value=value)
    elif isinstance(value, Dict):
        assert "scheduler_fn" in value and "scheduler_kwargs" in value
        scheduler = get_callable(value["scheduler_fn"])(**value["scheduler_kwargs"])
    else:
        raise TypeError("value should be either Dict or float")
    return scheduler


def explained_variance(y_pred, y_true):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    :param y_pred: (np.ndarray) the prediction
    :param y_true: (np.ndarray) the expected value
    :return: (float) explained variance of ypred and y
    """
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def stack_obs(obs: Union[List[np.ndarray], List[dict]]) -> np.ndarray:
    if isinstance(obs[0], np.ndarray):
        return np.stack(obs, axis=0)
    elif isinstance(obs[0], dict):
        stacked_obs = {}
        for key in obs[0].keys():
            stacked_obs[key] = np.stack([x[key] for x in obs], axis=0)
        return stacked_obs
    else:
        raise ValueError("Obs must be either np.ndarray or dict.")


def get_state_keys(obs: dict, curiosity_key: str, scale: bool = False, int_: bool = False) -> List[tuple]:
    if curiosity_key == "gps":
        spec_obs = deepcopy(obs["gps"])
        if scale:
            # rescale gps to its original range
            spec_obs[..., 0] = spec_obs[..., 0] * 1000
            spec_obs[..., 1] = spec_obs[..., 1] * 100
            spec_obs[..., 2] = spec_obs[..., 2] * 1000
            if int_:
                # convert gps to int
                spec_obs = np.round(spec_obs).astype(int)
        state_keys = [tuple(x) for x in spec_obs.reshape(spec_obs.shape[0], -1).tolist()]
    else:
        raise NotImplementedError
    
    return state_keys


def orthogonal_init(policy: nn.Module):
    for m in policy.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    final_layer = list(policy.actor_vcritic.actor_head.pi.modules())[-2]
    assert isinstance(final_layer, torch.nn.Linear), "Not compatible with current implementation!"
    torch.nn.init.zeros_(final_layer.bias)
    final_layer.weight.data.copy_(0.01 * final_layer.weight.data)
