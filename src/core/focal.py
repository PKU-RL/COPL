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


class FocalPPOWorker(PPOWorker):
    """PPO Worker with Focal Reward in MineCraft."""
    def __init__(
        self,
        *args,
        extrinsic_reward_coef: float = 1.0,
        intrinsic_reward_coef: float = 0.0,
        intrinsic_reward_norm: bool = False,
        intrinsic_reward_fuse: str = "add",     # not use
        # reward setting
        ir_logit: bool = False,      # logit / prob map to compute intrinsic reward
        ir_argmax: bool = False,     # only use patch where target is most possible
        ir_binary: float = 0.0,      # binarize the patch value to 0 or 1
        ir_kernel: str = "gaussian", # gaussian | linear
        ir_delta: bool = False,      # r_t <- r_t - r_{t-1}
        # mineclip (not use)
        mineclip_reward: Optional[CLIPReward] = None,
        mineclip_reward_coef: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.extrinsic_reward_coef = extrinsic_reward_coef
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.intrinsic_reward_norm = intrinsic_reward_norm
        self.num_target = len(self.env.env.target_name)
        self.intrinsic_reward_rms = [RunningMeanStd() for _ in range(self.num_target)]
        
        self.intrinsic_reward_key = "patch_logit" if ir_logit else "patch_prob"
        self.intrinsic_reward_argmax = ir_argmax
        self.intrinsic_reward_binary = ir_binary
        self.intrinsic_reward_kernel = ir_kernel
        self.kernel = None
        
        self.intrinsic_reward_delta = ir_delta
        self._prev_intrinsic_reward = None
        
    def _init_kernel(self, size: Tuple[int]):
        H, W = size # 10, 16
        x = np.linspace(0, W - 1, W)
        y = np.linspace(0, H - 1, H)
        X, Y = np.meshgrid(x, y)
        X, Y = X + 0.5, Y + 0.5
        
        if self.intrinsic_reward_kernel == "gaussian":
            mu_x, mu_y = W / 2, H / 2
            sigma_x, sigma_y = W / 3, H / 3
            kernel = np.exp(-((X - mu_x) ** 2 / (2 * sigma_x ** 2) + (Y - mu_y) ** 2 / (2 * sigma_y ** 2)))
            kernel /= kernel.max()
        elif self.intrinsic_reward_kernel == "gaussian_thin":
            mu_x, mu_y = W / 2, H / 2
            sigma_x, sigma_y = W / 5, H / 5
            kernel = np.exp(-((X - mu_x) ** 2 / (2 * sigma_x ** 2) + (Y - mu_y) ** 2 / (2 * sigma_y ** 2)))
            kernel /= kernel.max()
        elif self.intrinsic_reward_kernel == "linear":
            dist_to_center = np.linalg.norm(np.stack([X/W, Y/H], axis=-1) - np.array([1/2, 1/2]), axis=-1)
            kernel = 1 - dist_to_center
        else:   # none
            kernel = np.ones((H, W))
        self.kernel = kernel
        
    def compute_intrinsic_reward(self, map: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.kernel is None:
            self._init_kernel(map.shape[-2:])
        if self.intrinsic_reward_argmax:
            map *= mask
        if self.intrinsic_reward_binary > 0.:
            map[map > self.intrinsic_reward_binary] = 1.
            map[map < self.intrinsic_reward_binary] = 0.
        res = map * self.kernel
        res = res.reshape(*res.shape[:-2], -1).mean(axis=-1)
        
        if self.intrinsic_reward_delta:
            if self._prev_intrinsic_reward is None:
                self._prev_intrinsic_reward = np.zeros_like(res)
            rew = res - self._prev_intrinsic_reward
            self._prev_intrinsic_reward = res
        else:
            rew = res
        return rew
    
    @staticmethod
    def spec_parser(parser):
        parser.add_argument("--ir_logit", action="store_true")
        parser.add_argument("--ir_argmax", action="store_true")
        parser.add_argument("--ir_binary", type=float, default=0.0)
        parser.add_argument("--ir_kernel", type=str, default="gaussian", 
                            choices=["gaussian", "gaussian_thin", "linear", "none"])
        parser.add_argument("--ir_delta", action="store_true")

    @staticmethod
    def spec_config(config, args):
        config["worker_kwargs"].update(
            ir_logit=args.ir_logit,
            ir_argmax=args.ir_argmax,
            ir_binary=args.ir_binary,
            ir_kernel=args.ir_kernel,
            ir_delta=args.ir_delta,
        )
        
        # assert args.target_name is not None
        
    @staticmethod
    def exp_name(args) -> str:
        return f"focal_{args.ir_kernel}"
    
    def collect_batch(
        self, epoch: int, deterministic: bool = False
    ) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
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
            
            batch["intrinsic_reward"].append(
                self.compute_intrinsic_reward(next_obs_[self.intrinsic_reward_key],
                                              next_obs_["patch_mask"]))
            if self._prev_intrinsic_reward is not None and next_first.any():
                reset_indices = np.where(next_first)[0]
                for index in reset_indices:
                    self._prev_intrinsic_reward[index] = 0.
            
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
        batch["intrinsic_reward"] = np.asarray(batch["intrinsic_reward"], dtype=np.float32)
        
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
        if self.intrinsic_reward_norm:
            for i in range(self.num_target):
                intrinsic_reward_i = batch["intrinsic_reward"][batch["obs"]["target_index"][..., i]]
                self.intrinsic_reward_rms[i].update(intrinsic_reward_i[intrinsic_reward_i > 0.])
                batch["intrinsic_reward"][batch["obs"]["target_index"][..., i]] /= self.intrinsic_reward_rms[i].mean
                
        batch["reward"] *= self.extrinsic_reward_coef
        batch["reward"] += batch["intrinsic_reward"] * self.intrinsic_reward_coef
        
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

        info = {}
        # average intrinsic reward over the whole episode
        info["int_rew"] = []
        for i in range(self.num_target):
            intrinsic_reward_i = batch["intrinsic_reward"][batch["obs"]["target_index"][..., i]]
            info["int_rew"].append(f"{intrinsic_reward_i.mean():.3f}")
        info["int_rew"] = "/".join(info["int_rew"])
        # running mean
        if self.intrinsic_reward_norm:
            info["int_rms"] = []
            for i in range(self.num_target):
                info["int_rms"].append(f"{self.intrinsic_reward_rms[i].mean:.3f}")
            info["int_rms"] = "/".join(info["int_rms"])

        return batch, info

    def log_spec_reward(self, buffer: Buffer):
        logger.logkv("env/reward_i", buffer.get_stats_mean("intrinsic_reward"))
