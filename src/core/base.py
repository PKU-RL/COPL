from collections import defaultdict
from typing import Callable, Dict, Optional, Tuple, Union

import gym
import numpy as np
import torch
from tqdm import tqdm

from ..common.agent import Actor, Learner, worker_class
from ..common.buffer import Buffer
from ..common.pcgrad import PCGrad
from ..utils.utils import multimap
from .utils import (calculate_gae, explained_variance, get_callable,
                    get_scheduler, stack_obs)


class PPOActor(Actor):
    def __init__(
        self,
        env_fn: Callable[..., gym.Env],
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        n_steps: int,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        stdout: bool = True,
    ):
        super().__init__(env_fn, env_kwargs, policy_fn, policy_kwargs, device)
        self.n_steps = n_steps
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.stdout = stdout

    def collect(
        self,
        scheduler_step: int,
        buffer: Buffer,
        deterministic: bool = False,
    ) -> dict:
        # Collect a batch of samples
        batch, last_obs, last_first, last_bad = self.collect_batch(scheduler_step, deterministic)
        # Compute advantage
        batch, info = self.process_batch(batch, last_obs, last_first, last_bad)
        # Organize axes
        for key in batch:
            batch[key] = multimap(lambda val: val.swapaxes(0, 1), batch[key])
        # Send data to buffer
        self.add_batch_to_buffer(
            scheduler_step=scheduler_step,
            batch=batch,
            size=len(batch["reward"]),
            buffer=buffer,
        )

        return info

    def collect_batch(self, epoch, deterministic=False) -> Tuple[dict, np.ndarray, np.ndarray]:
        """
        Collect a batch of trajectories
        """
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
            
            if self.stdout:
                tbar.set_postfix_str(f"Ret: {self.env.callmethod('get_ep_stat_mean', 'r'):.2f}")
        
        # Concatenate
        batch["reward"] = np.asarray(batch["reward"], dtype=np.float32)
        batch["obs"] = stack_obs(batch["obs"])
        batch["first"] = np.asarray(batch["first"], dtype=bool)
        batch["bad"] = np.asarray(batch["bad"], dtype=bool)
        batch["action"] = np.asarray(batch["action"])
        batch["value"] = np.asarray(batch["value"], dtype=np.float32)
        batch["logpac"] = np.asarray(batch["logpac"], dtype=np.float32)
        
        if self.policy.is_recurrent:
            # deprecated code for LSTM
            # if batch["rnn_states"][0] is None:
            #     rnn_states = tuple(self.policy.rnn_states)
            #     batch["rnn_states"][0] = tuple(torch.zeros_like(s) for s in rnn_states)
            # for i in range(self.n_steps):
            #     batch["rnn_states"][i] = torch.stack(batch["rnn_states"][i], dim=-1).cpu().numpy()
            # TODO: compare with both LSTM and GRU
            if batch["rnn_states"][0] is None:
                batch["rnn_states"][0] = torch.zeros_like(self.policy.rnn_states)
            batch["rnn_states"] = torch.stack(batch["rnn_states"], axis=0).cpu().numpy()
        
        return batch, next_obs, next_first, next_bad
    
    def collect_eval_episodes(self, n_episode, deterministic=False, deactivate_use=False):
        batch = defaultdict(list)
        
        cnt_episode = 0
        
        with tqdm(total=n_episode, desc="Testing episode") as pbar:
            while cnt_episode < n_episode:
                reward, obs, first, bad = self.env.observe()
                if self.policy.is_recurrent:
                    batch["rnn_states"].append(self.policy.rnn_states)
                action, _ = self.policy.act(
                    multimap(lambda val: np.expand_dims(np.array(val), axis=0), obs), first[None, ...],
                    deterministic=deterministic)
                batch["obs"].append(obs)

                # Deactivate use
                if deactivate_use:
                    second_act = action[..., 1]
                    second_act[second_act == 1] = 0
                    assert not np.any(action[..., 1] == 1)

                self.env.act(action.squeeze(0))
                reward, _, next_first, _ = self.env.observe()
                batch["reward"].append(reward)

                cnt_episode += np.sum(next_first)
                pbar.update(np.sum(next_first))
                pbar.set_postfix_str(f"Success: {self.env.callmethod('get_ep_stat_mean', 'r'):.2f}")
        
        return batch

    def process_batch(
        self, batch: dict, last_obs: np.ndarray, last_first: np.ndarray, last_bad: np.ndarray,
    ) -> Tuple[dict, dict]:
        """
        Process torche collected batch, e.g. computing advantages
        """
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


class PPOLearner(Learner):
    def __init__(
        self,
        policy_fn: str,
        policy_kwargs: dict,
        optimizer_fn: str,
        optimizer_kwargs: dict,
        n_epochs: int,
        n_minibatches: int,
        normalize_adv: bool = False,
        normalize_adv_mt: bool = False,
        clip_range: Union[Dict, float] = 0.2,
        vf_clip_range: Union[Dict, float] = 0.2,
        vf_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        is_truncated_recurrent: bool = True,
        data_chunk_length: int = 10,
        use_pcgrad: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(policy_fn, policy_kwargs, device)
        self.optimizer = get_callable(optimizer_fn)(
            params=self.policy.parameters(), **optimizer_kwargs
        )
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.normalize_adv = normalize_adv
        self.normalize_adv_mt = normalize_adv_mt
        self.clip_range = get_scheduler(clip_range)
        self.vf_clip_range = get_scheduler(vf_clip_range)
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.is_truncated_recurrent = is_truncated_recurrent
        self.data_chunk_length = data_chunk_length
        self.use_pcgrad = use_pcgrad
        if use_pcgrad:
            assert self.n_minibatches == 1, "Each update, we use the whole buffer in order to balance the samples from different tasks"
            self.optimizer = PCGrad(self.optimizer)

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
                pg_loss, vf_loss, entropy, extra_out = self.policy.loss(
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
                if self.use_pcgrad:
                    total_losses = [
                        pl + self.vf_loss_coef * vl - self.entropy_coef * e
                        for pl, vl, e in zip(pg_loss, vf_loss, entropy)
                    ]
                    self.optimizer.pc_backward(total_losses)
                    self.pre_optim_step_hook()
                    self.optimizer.step()
                    # Saving statistics
                    num_tasks = len(pg_loss)
                    stats_dict["policy_loss"].append(sum(pg_loss).item() / num_tasks)
                    stats_dict["value_loss"].append(sum(vf_loss).item() / num_tasks)
                    stats_dict["entropy"].append(entropy[0].item())
                    stats_dict["total_loss"].append(sum(total_losses).item() / num_tasks)
                else:
                    total_loss = (
                        pg_loss + self.vf_loss_coef * vf_loss - self.entropy_coef * entropy
                    )
                    total_loss.backward()
                    self.pre_optim_step_hook()
                    self.optimizer.step()
                    # Saving statistics
                    stats_dict["policy_loss"].append(pg_loss.item())
                    stats_dict["value_loss"].append(vf_loss.item())
                    stats_dict["entropy"].append(entropy.item())
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

    def pre_optim_step_hook(self):
        self.clip_gradient(max_norm=self.max_grad_norm)


class PPOWorker(worker_class(PPOActor, PPOLearner)):
    def __init__(
        self,
        env_fn: Callable[..., gym.Env],
        env_kwargs: dict,
        policy_fn: str,
        policy_kwargs: dict,
        optimizer_fn: str,
        optimizer_kwargs: dict,
        n_steps: int,
        n_epochs: int,
        n_minibatches: int,
        discount_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        normalize_adv: bool = False,
        normalize_adv_mt: bool = False,
        use_proper_time_limit: bool = True,
        clip_range: Union[Dict, float] = 0.2,
        vf_clip_range: Union[Dict, float] = 0.2,
        vf_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        truncated_bptt: int = 10,
        use_pcgrad: bool = False,
        device: Union[str, torch.device] = "cpu",
        stdout: bool = True,
        worker_weight: float = 1.0,
    ):
        if use_pcgrad:
            assert n_minibatches == 1, "Each update, we use the whole buffer in order to balance the samples from different tasks"
            policy_fn = "src.core.policy_pcgrad.PPOMultiDiscretePolicy"

        super().__init__(
            env_fn, env_kwargs, policy_fn, policy_kwargs, device, worker_weight
        )
        self.optimizer = get_callable(optimizer_fn)(
            params=self.policy.parameters(), **optimizer_kwargs
        )
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.normalize_adv_mt = normalize_adv_mt
        self.use_proper_time_limit = use_proper_time_limit
        self.clip_range = get_scheduler(clip_range)
        self.vf_clip_range = get_scheduler(vf_clip_range)
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.data_chunk_length = truncated_bptt
        self.is_truncated_recurrent = True if truncated_bptt > 0 else False

        self.use_pcgrad = use_pcgrad
        if use_pcgrad:
            self.optimizer = PCGrad(self.optimizer)

        self.stdout = stdout

    @staticmethod
    def spec_parser(parser):
        pass
    
    @staticmethod
    def spec_config(config, args):
        pass
    
    @staticmethod
    def policy_fn():
        return "src.core.policy.PPOMultiDiscretePolicy"
    
    @staticmethod
    def exp_name(args):
        return "ppo"
