from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ..common.actor_critic import ActorVCritic, ParamsMixin
from ..utils.utils import multimap
from .utils import MultiCategorical, clamp, get_callable


class PPOBasePolicy(ParamsMixin, nn.Module):
    def __init__(
        self,
        n_outputs: int,
        extractor_config: Union[str, dict],
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation_fn: str = "torch.nn.ReLU",
        init_weight_fn: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(device)
        self.actor_vcritic = ActorVCritic(
            n_outputs=n_outputs,
            extractor_config=extractor_config,
            use_rnn=use_rnn,
            rnn_hidden=rnn_hidden,
            actor_hiddens=actor_hiddens,
            critic_hiddens=critic_hiddens,
            activation=get_callable(activation_fn),
        )
        self.rnn_states = None
        self.is_recurrent = use_rnn
        if init_weight_fn is not None:
            get_callable(init_weight_fn)(self)

    @torch.no_grad()
    def step(self, obs, first, deterministic=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = multimap(lambda x: torch.as_tensor(x).to(self.device).float(), obs)
        first = torch.as_tensor(first).to(self.device)
        pi, value, self.rnn_states = self.actor_vcritic(obs, first, self.rnn_states)
        dist = self.distribution_cls(pi)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        logpacs = dist.log_prob(action)
        if isinstance(dist, Normal):
            logpacs = logpacs.sum(dim=-1)
        return (
            action.cpu().numpy(),
            value.squeeze(-1).cpu().numpy(),
            logpacs.cpu().numpy(),
        )
    
    @torch.no_grad()
    def act(self, obs, first, deterministic=False) -> Tuple[np.ndarray, np.ndarray]:
        obs = multimap(lambda x: torch.as_tensor(x).to(self.device).float(), obs)
        first = torch.as_tensor(first).to(self.device)
        pi, self.rnn_states = self.actor_vcritic.forward_actor(obs, first, self.rnn_states)
        dist = self.distribution_cls(pi)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        logpacs = dist.log_prob(action)
        if isinstance(dist, Normal):
            logpacs = logpacs.sum(dim=-1)
        return action.cpu().numpy(), logpacs.cpu().numpy()

    @torch.no_grad()
    def value(self, obs, first):
        obs = multimap(lambda x: torch.as_tensor(x).to(self.device).float(), obs)
        first = torch.as_tensor(first).to(self.device)
        value, _ = self.actor_vcritic.forward_critic(obs, first)
        return value.squeeze(-1).cpu().numpy()

    def loss(
        self,
        obs,
        advs,
        firsts,
        actions,
        old_values,
        old_logpacs,
        rnn_states,
        clip_range: float,
        vf_clip_range: float,
        normalize_adv: bool = False,
        normalize_adv_mt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Convert from numpy array to torch tensor
        obs = multimap(lambda x: torch.as_tensor(x).to(self.device).float(), obs)
        advs = torch.as_tensor(advs).to(self.device).float()
        firsts = torch.as_tensor(firsts).to(self.device)
        actions = torch.as_tensor(actions).to(self.device)
        old_values = torch.as_tensor(old_values).to(self.device).float()
        old_logpacs = torch.as_tensor(old_logpacs).to(self.device).float()
        if rnn_states is not None:
            rnn_states = torch.as_tensor(rnn_states).to(self.device).float().contiguous()
            # if rnn_states.shape[-1] > 1:
            #     rnn_states = rnn_states.unbind(-1)
            # else:
            #     rnn_states = rnn_states.squeeze(-1)

        # Calculate returns
        returns = advs + old_values
        # Advantage normalization
        if normalize_adv:
            if normalize_adv_mt:
                # normalize advantages in each task separately
                assert "target_index" in obs
                num_tasks = obs["target_index"].shape[-1]
                for i in range(num_tasks):
                    indices = obs["target_index"][..., i] > 0
                    if indices.sum() == 0:
                        continue
                    mu = advs[indices].mean()
                    std = advs[indices].std()
                    advs[indices] = (advs[indices] - mu) / (std + 1e-8)
            else:
                # normalize advantages of all tasks together
                advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Forward
        pi, values, _ = self.actor_vcritic(obs, firsts, rnn_states)
        values = values.squeeze(-1)

        # Compute policy loss
        dist = self.distribution_cls(pi)
        logpacs = dist.log_prob(actions)
        if isinstance(dist, Normal):
            logpacs = logpacs.sum(dim=-1)
        ratio = torch.exp(logpacs - old_logpacs)
        pg_losses1 = -advs * ratio
        pg_losses2 = -advs * clamp(ratio, min=1.0 - clip_range, max=1.0 + clip_range)
        pg_loss = torch.mean(torch.max(pg_losses1, pg_losses2))

        # Compute value loss
        vf_losses = torch.square(values - returns)
        values_clipped = clamp(
            values, min=old_values - vf_clip_range, max=old_values + vf_clip_range
        )
        vf_losses = torch.max(vf_losses, torch.square(values_clipped - returns))
        vf_loss = 0.5 * torch.mean(vf_losses)

        # Compute entropy
        entropy = dist.entropy()
        if isinstance(dist, Normal):
            entropy = entropy.sum(dim=-1)
        entropy = torch.mean(entropy)

        # Calculate additional quantities
        extra_out = {}
        with torch.no_grad():
            extra_out["approx_kl"] = 0.5 * torch.mean(torch.square(logpacs - old_logpacs))
            extra_out["clip_frac"] = torch.mean(((ratio - 1.0).abs() > clip_range).float())

        return pg_loss, vf_loss, entropy, extra_out
        

class PPOMultiDiscretePolicy(PPOBasePolicy):
    def __init__(
        self,
        n_actions: Tuple[int],
        extractor_config: Union[str, dict],
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation_fn: str = "torch.nn.ReLU",
        init_weight_fn: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            n_outputs=sum(n_actions),
            extractor_config=extractor_config,
            use_rnn=use_rnn,
            rnn_hidden=rnn_hidden,
            actor_hiddens=actor_hiddens,
            critic_hiddens=critic_hiddens,
            activation_fn=activation_fn,
            init_weight_fn=init_weight_fn,
            device=device,
        )
        self.n_actions = n_actions
        self.distribution_cls = lambda pi: MultiCategorical(logits=pi, num_categories=n_actions)


class PPOContinuousPolicy(PPOBasePolicy):
    def __init__(
        self,
        n_actions: int,
        extractor_config: Union[str, dict],
        log_std_init: float = 0.0,
        use_rnn: bool = False,
        rnn_hidden: int = 256,
        actor_hiddens: Tuple[int, ...] = (),
        critic_hiddens: Tuple[int, ...] = (),
        activation_fn: str = "torch.nn.ReLU",
        init_weight_fn: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(
            n_outputs=n_actions,
            extractor_config=extractor_config,
            use_rnn=use_rnn,
            rnn_hidden=rnn_hidden,
            actor_hiddens=actor_hiddens,
            critic_hiddens=critic_hiddens,
            activation_fn=activation_fn,
            init_weight_fn=init_weight_fn,
            device=device,
        )
        self.action_dim = n_actions
        self.log_std_init = log_std_init
        self.log_std = nn.Parameter(torch.ones(n_actions, device=device) * log_std_init)
        self.distribution_cls = lambda pi: Normal(
            loc=pi, scale=torch.ones_like(pi) * self.log_std.exp()
        )
