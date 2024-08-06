from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..common.buffer import Buffer
from ..envs.clip import CLIPReward
from ..utils import logger
from ..utils.utils import assign, multimap
from .base import PPOWorker
from .utils import RunningMeanStd, calculate_gae, stack_obs


class ModifiedPPOWorker(PPOWorker):
    """PPO Worker with MineCLIP Reward in MineCraft.
    """
    def __init__(
        self, 
        *args,
        extrinsic_reward_coef: float = 1.0,
        intrinsic_reward_coef: float = 0.0, # not use
        intrinsic_reward_norm: bool = False,
        intrinsic_reward_fuse: str = "add", # not use
        # mineclip
        mineclip_reward: Optional[CLIPReward] = None,
        mineclip_reward_coef: float = 0.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.extrinsic_reward_coef = extrinsic_reward_coef
        self.intrinsic_reward_norm = intrinsic_reward_norm
        self.intrinsic_reward_rms = RunningMeanStd()
        # for mineclip
        self.mineclip_reward = mineclip_reward
        self.mineclip_reward_coef = mineclip_reward_coef
        
        self._init_flag = False
        
    @staticmethod
    def exp_name(args) -> str:
        return "ppo"

    def collect_batch(
        self, epoch: int, deterministic: bool = False
    ) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
        if not self._init_flag:
            _, obs, _, _ = self.env.observe()
            self._init_flag = True

        # Rollout
        batch = defaultdict(list)
        if self.stdout:
            tbar = tqdm(range(self.n_steps), desc=f"Epoch {epoch}")
        else:
            tbar = range(self.n_steps)
        for t in tbar:
            reward, obs, first, bad = self.env.observe()
            if self.policy.is_recurrent:
                batch["rnn_states"].append(self.policy.rnn_states)
            # Multi-task setting
            if "MT" in self.env.__class__.__name__:
                batch["target"].append(self.env.env.cur_target.copy())
            action, value, logpacs = self.policy.step(
                multimap(lambda val: np.expand_dims(np.array(val), axis=0), obs), first[None, ...],
                deterministic=deterministic)
            batch["obs"].append(obs)
            batch["first"].append(first)
            batch["bad"].append(bad)
            batch["action"].append(action.squeeze(0))
            batch["value"].append(value.squeeze(0))
            batch["logpac"].append(logpacs.squeeze(0))
            self.env.act(action.squeeze(0))
            reward, next_obs, next_first, next_bad = self.env.observe()
            batch["reward"].append(reward)

            next_obs_ = deepcopy(next_obs)  # do not modify env.buf_obs
            if next_first.any():
                reset_indices = np.where(next_first)[0]
                env_infos = self.env.get_info()
                for index in reset_indices:
                    assign(next_obs_, env_infos[index]["reset_ob"], index)
            batch["next_obs"].append(next_obs_)
            
            if self.stdout:
                tbar.set_postfix_str(f"Ret: {self.env.callmethod('get_ep_stat_mean', 'r'):.2f}")
        
        # Concatenate
        batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
        batch["obs"] = stack_obs(batch["obs"])
        batch["next_obs"] = stack_obs(batch["next_obs"])
        batch["first"] = np.asarray(batch["first"], dtype=bool)
        batch["bad"] = np.asarray(batch["bad"], dtype=bool)
        batch["action"] = np.asarray(batch["action"])
        batch["value"] = np.asarray(batch["value"], dtype=np.float32)
        batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)

        if "MT" in self.env.__class__.__name__:
            batch["target"] = np.asarray(batch["target"], dtype=np.int64)
        
        if self.policy.is_recurrent:
            # deprecated code for LSTM
            # if batch["rnn_states"][0] is None:
            #     rnn_states = tuple(self.policy.rnn_states)
            #     batch["rnn_states"][0] = tuple(torch.zeros_like(s) for s in rnn_states)
            # for i in range(self.n_steps):
            #     batch["rnn_states"][i] = torch.stack(batch["rnn_states"][i], dim=-1).cpu().numpy()
            # TODO: compatible with both LSTM and GRU
            if batch["rnn_states"][0] is None:
                batch["rnn_states"][0] = torch.zeros_like(self.policy.rnn_states)
            batch["rnn_states"] = torch.stack(batch["rnn_states"], axis=0).swapaxes(1, 2).cpu().numpy()
        
        return batch, next_obs, next_first, next_bad

    def process_batch(
        self, batch: dict, last_obs: dict, last_first: np.ndarray, last_bad: np.ndarray,
    ) -> Tuple[dict, dict]:
        batch["reward"] *= self.extrinsic_reward_coef
        
        if self.mineclip_reward is not None:
            if "MT" in self.env.__class__.__name__:
                batch["mineclip_reward"] = self.mineclip_reward.compute_reward(batch["obs"]["obs_emb"],
                                                                               batch["next_obs"]["obs_emb"],
                                                                               batch["first"],
                                                                               batch["target"])
            else:
                batch["mineclip_reward"] = self.mineclip_reward.compute_reward(batch["obs"]["obs_emb"],
                                                                               batch["next_obs"]["obs_emb"],
                                                                               batch["first"])
            
            mineclip_reward = deepcopy(batch["mineclip_reward"])
            if self.intrinsic_reward_norm:
                self.intrinsic_reward_rms.update(mineclip_reward.reshape(-1, 1))
                mineclip_reward /= np.sqrt(self.intrinsic_reward_rms.var + 1e-8)
            
            batch["reward"] += mineclip_reward * self.mineclip_reward_coef    
        
        last_value = self.policy.value(
            multimap(lambda val: np.expand_dims(val, axis=0), last_obs), last_first[None, ...])
        advs = calculate_gae(
            rewards=batch["reward"],
            values=batch["value"],
            firsts=batch["first"],
            last_value=last_value,
            last_first=last_first,
            discount_gamma=self.discount_gamma,
            gae_lambda=self.gae_lambda,
            use_proper_time_limit=self.use_proper_time_limit,
            bad=batch["bad"],
            last_bad=last_bad,
        )
        batch["adv"] = advs
        return batch, {}

    def log_spec_reward(self, buffer: Buffer):
        pass
