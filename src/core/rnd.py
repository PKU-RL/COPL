from collections import defaultdict
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ..common.buffer import Buffer
from ..envs.clip import CLIPReward
from ..segment.segmineclip import preprocess
from ..utils import logger
from ..utils.utils import assign, multimap
from .base import PPOWorker
from .utils import RunningMeanStd, calculate_gae, explained_variance, stack_obs


class RNDPPOWorker(PPOWorker):
    def __init__(
        self,
        *args,
        extrinsic_reward_coef: float = 1.0,
        intrinsic_reward_coef: float = 0.0,
        intrinsic_reward_norm: bool = False,
        intrinsic_reward_fuse: str = "add", # not use
        rnd_loss_coef: float = 0.0,
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
        
        self.rnd_loss_coef = rnd_loss_coef
        
    @staticmethod
    def spec_parser(parser):
        parser.add_argument("--rnd_loss_coef", type=float, default=1.0)

    @staticmethod
    def spec_config(config, args):
        config["worker_kwargs"]["rnd_loss_coef"] = args.rnd_loss_coef
    
    @staticmethod
    def policy_fn():
        return "src.core.policy_rnd.RNDPPOPolicy"
    
    @staticmethod
    def exp_name(args)-> str:
        return "rnd"
    
    @torch.no_grad()
    def compute_intrinsic_reward(self, next_obs) -> np.ndarray:
        obs_rgb = preprocess(next_obs["obs_rgb"]).to(self.device).float()
        obs_emb = torch.as_tensor(next_obs["obs_emb"]).to(self.device).float()
        pred_obs_emb = self.policy.predictor_net(obs_rgb)
        rnd_reward = torch.norm(obs_emb - pred_obs_emb, dim=-1, p=2).cpu().numpy()
        return rnd_reward
    
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
            
            batch["intrinsic_reward"].append(self.compute_intrinsic_reward(next_obs_))
            
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
            if self.num_target == 1:
                self.intrinsic_reward_rms[0].update(batch["intrinsic_reward"])
                batch["intrinsic_reward"] /= np.sqrt(self.intrinsic_reward_rms[0].var + 1e-8)
            else:
                for i in range(self.num_target):
                    indices = batch["obs"]["target_index"][..., i]
                    intrinsic_reward_i = batch["intrinsic_reward"][indices]
                    self.intrinsic_reward_rms[i].update(intrinsic_reward_i)
                    batch["intrinsic_reward"][indices] /= np.sqrt(self.intrinsic_reward_rms[i].var + 1e-8)
                
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
        if self.num_target == 1:
            info["int_rew"] = f"{batch['intrinsic_reward'].mean():.3f}"
            if self.intrinsic_reward_norm:
                info["int_rms"] = f"{self.intrinsic_reward_rms[0].var[0]:.3f}"
        else:
            info["int_rew"] = []
            for i in range(self.num_target):
                intrinsic_reward_i = batch["intrinsic_reward"][batch["obs"]["target_index"][..., i]]
                info["int_rew"].append(f"{intrinsic_reward_i.mean():.3f}")
            info["int_rew"] = "/".join(info["int_rew"])
            if self.intrinsic_reward_norm:
                info["int_rms"] = []
                for i in range(self.num_target):
                    info["int_rms"].append(f"{self.intrinsic_reward_rms[i].var[0]:.3f}")
                info["int_rms"] = "/".join(info["int_rms"])

        return batch, info
    
    def learn(self, scheduler_step: int, buffer: Buffer):
        # Retrieve data from buffer
        if self.policy.is_recurrent:
            if self.is_truncated_recurrent:
                batch = buffer.get_all_truncated_recurrent(self.data_chunk_length)  # [N*T/L,L,...]
            else:
                batch = buffer.get_all_recurrent()  # [N,T,...]
        else:
            batch = buffer.get_all_feedforward()    # [N*T,1,...]
        # Build a dict to save training statistics
        stats_dict = defaultdict(list)
        # Minibatch training
        B, T = batch["reward"].shape[:2]
        indices = np.arange(B)
        minibatch_size = B // self.n_minibatches
        assert minibatch_size > 1
        # Get current clip range
        cur_clip_range = self.clip_range.value(step=scheduler_step)
        cur_vf_clip_range = self.vf_clip_range.value(step=scheduler_step)
        # Train for n_epochs
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, B, minibatch_size):
                end = start + minibatch_size
                sub_indices = indices[start:end]
                if self.policy.is_recurrent:
                    rnn_states = batch["rnn_states"][sub_indices].swapaxes(0, 1)
                else:
                    rnn_states = None
                self.optimizer.zero_grad()
                pg_loss, vf_loss, entropy, rnd_loss, extra_out = self.policy.loss(
                    obs=multimap(lambda x: x[sub_indices].swapaxes(0, 1), batch["obs"]),
                    advs=batch["adv"][sub_indices].swapaxes(0, 1),
                    firsts=batch["first"][sub_indices].swapaxes(0, 1),
                    actions=batch["action"][sub_indices].swapaxes(0, 1),
                    old_values=batch["value"][sub_indices].swapaxes(0, 1),
                    old_logpacs=batch["logpac"][sub_indices].swapaxes(0, 1),
                    rnn_states=rnn_states,
                    clip_range=cur_clip_range,
                    vf_clip_range=cur_vf_clip_range,
                    normalize_adv=self.normalize_adv,
                    normalize_adv_mt=self.normalize_adv_mt,
                )
                total_loss = (
                    pg_loss + self.vf_loss_coef * vf_loss - self.entropy_coef * entropy +
                    self.rnd_loss_coef * rnd_loss
                )
                total_loss.backward()
                self.pre_optim_step_hook()
                self.optimizer.step()
                # Saving statistics
                stats_dict["policy_loss"].append(pg_loss.item())
                stats_dict["value_loss"].append(vf_loss.item())
                stats_dict["entropy"].append(entropy.item())
                stats_dict["rnd_loss"].append(rnd_loss.item())
                stats_dict["total_loss"].append(total_loss.item())
                for key in extra_out:
                    stats_dict[key].append(extra_out[key].item())
        # Compute mean
        for key in stats_dict:
            stats_dict[key] = np.mean(stats_dict[key])
        # Compute explained variance
        stats_dict["explained_variance"] = explained_variance(
            y_pred=batch["value"], y_true=batch["value"] + batch["adv"]
        )
        return stats_dict
    
    def log_spec_reward(self, buffer: Buffer):
        logger.logkv("env/reward_i", buffer.get_stats_mean("intrinsic_reward"))