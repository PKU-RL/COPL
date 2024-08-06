from abc import ABC, abstractmethod
from typing import Callable, Optional, OrderedDict, Union

import gym
import torch

from .buffer import Buffer
from .utils import get_callable, get_n_actions


class Learner(ABC):
    def __init__(
        self,
        policy_fn: str,
        policy_kwargs: dict,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        Optimizer might be algorithm-specific
        """
        super().__init__()
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.device = device

    @abstractmethod
    def learn(self, scheduler_step: int, buffer: Buffer) -> dict:
        pass

    def broadcast_params(self) -> OrderedDict:
        return self.policy.get_params()

    def clip_gradient(
        self,
        max_norm: Optional[float] = None,
        clip_value: Optional[float] = None,
        norm_type: float = 2.0,
    ):
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm, norm_type)
        elif clip_value is not None:
            torch.nn.utils.clip_grad_value_(self.policy.parameters(), clip_value)

    def pre_optim_step_hook(self) -> None:
        pass


class Actor(ABC):
    def __init__(
        self,
        env_fn: Callable[..., gym.Env],
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.env = env_fn(**env_kwargs)
        self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
        self.policy.to(device)
        self.device = device

    @abstractmethod
    def collect(
        self,
        scheduler_step: int,
        buffer: Buffer,
        learner: Optional[Learner] = None,
    ) -> None:
        """
        Collect transitions from interaction with the environment, and
        add collected data to the buffer.
        """
        pass

    def sync_params(self, learner: Learner) -> None:
        params = learner.broadcast_params()
        self.policy.set_params(params)

    def add_batch_to_buffer(
        self, scheduler_step: int, batch: dict, size: int, buffer: Buffer
    ) -> None:
        next_idx = buffer.update_next_idx(size=size)
        buffer.add(
            scheduler_step=scheduler_step, data=batch, idx=next_idx, size=size
        )


def worker_class(Actor, Learner):
    class Worker(Actor, Learner):
        def __init__(
            self,
            env_fn: Callable[..., gym.Env],
            env_kwargs: dict,
            policy_fn: str,
            policy_kwargs: dict,
            device: Union[str, torch.device] = "cpu",
            worker_weight: float = 1.0,
        ) -> None:
            self.env = env_fn(**env_kwargs)
            policy_kwargs["n_actions"] = get_n_actions(self.env.ac_space)
            self.policy = get_callable(policy_fn)(device=device, **policy_kwargs)
            self.policy.to(device)
            self.device = device
            self.worker_weight = worker_weight

        def pre_optim_step_hook(self) -> None:
            super().pre_optim_step_hook()
        
        def close(self):
            # Ensure MineDojo closed properly
            self.env.callmethod("close")
        
    return Worker
