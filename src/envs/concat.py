from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from ..others.mineclip_official import tokenize_batch
from ..segment.segmineclip import SegVideoCLIP, preprocess
from ..segment.utils import encode_text_with_prompt_ensemble
from ..utils.utils import concat, multimap, split
from .utils import _chunk_seq, allsame


class ConcatVectorEnv(object):
    """
    Concatenate multiple environments into a single environment.

    :param envs: list of environments to concatenate, must all have the same ac_space and ob_space
    """

    def __init__(
        self, 
        envs: Sequence[gym.Env], 
        clip_model: Optional[SegVideoCLIP] = None,
        target_list: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        pseudo_random: bool = False,
        fixed_target: bool = False,
        keep_raw_rgb: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        self.ob_space = envs[0].ob_space
        self.ac_space = envs[0].ac_space
        self.num = sum(env.num for env in envs)
        
        total_num = sum(env.num for env in envs)
        assert allsame([env.ac_space for env in envs])
        assert allsame([env.ob_space for env in envs])
        self.envs = envs
        self.device = device

        self.clip_model = clip_model
        self.initial_mobs = envs[0].gym_env.initial_mobs
        self.number_mobs = envs[0].gym_env.number_mobs
        self.target_name = envs[0].gym_env.target_names_vlm
        if self.initial_mobs is None:
            print("Mob in env: none")
        else:
            print("Mob in env: [", ", ".join(self.initial_mobs), "] x", self.number_mobs)
        print("Target in env:", ", ".join(self.target_name))
        if clip_model is not None and target_list is not None:
            self.init_texts(target_list)
            print("Index in keywords:", " ".join(map(str, self.target_index)))
        self.prompt_embeddings = None
        if prompts is not None:
            self.prompt_embeddings = self.clip_model.text_encoder(tokenize_batch(prompts).to(self.device))
        
        self.extend_obs_space()
        self._keep_raw_rgb = keep_raw_rgb
        # cache rew, obs, first, and bad arrays
        self.buf_rews, self.buf_obs, self.buf_firsts, self.buf_bads = None, None, None, None
        self.buf_infos = [{} for _ in range(self.num)]
        
        assert not (pseudo_random and fixed_target), "Cannot set both pseudo_random and fixed_target to True!"
        self.pseudo_random = pseudo_random
        self.fixed_target = fixed_target
        self._target_queue = deque(maxlen=len(self.target_name))
        self._target_queue.extend(np.random.permutation(len(self.target_name)).tolist())
        # reset the environments
        if self.pseudo_random:
            for env in self.envs:
                if len(self._target_queue) == 0:
                    self._target_queue.extend(np.random.permutation(len(self.target_name)).tolist())
                env.reset(target=self._target_queue.popleft())
        elif self.fixed_target:
            assert len(self.target_name) == self.num, "Number of environments must match number of targets!"
            for i, env in enumerate(self.envs):
                env.reset(target=i)
        else:
            for env in self.envs:
                env.reset()
        self.cur_target = [env.gym_env.cur_target for env in self.envs]
        self.cur_target_index = [self.target_index[i] for i in self.cur_target]
        
        # DEBUG
        # for index in range(self.num):
        #     print(f"Env {index} reset to {self.cur_target[index]} (list id: {self.cur_target_index[index]})")
            
    def extend_obs_space(self):
        if self.clip_model is not None:
            self.ob_space["obs_emb"] = spaces.Box(low=-1, high=1, shape=(512,), dtype=np.float32)
    
    @torch.no_grad()
    def init_texts(self, target_list: str):
        with open(target_list, "r") as f:
            all_texts = f.read().splitlines()
        for name in self.target_name:
            if name not in all_texts:
                all_texts.append(name)
        self.text_embeddings = encode_text_with_prompt_ensemble(self.clip_model, all_texts, self.device)
        self.target_index = [all_texts.index(name) for name in self.target_name]
    
    @torch.no_grad()
    def extend_clip_obs(self):
        assert self.buf_obs is not None
        assert self.clip_model is not None

        rgb_tensor = preprocess(np.asarray(self.buf_obs["obs_rgb"], dtype=np.uint8)).to(self.device)
        obs_emb, patch_logit, patch_prob, patch_label, patch_mask = \
            self.clip_model.encode_and_segment_image(rgb_tensor, self.cur_target_index, self.text_embeddings)
        self.buf_obs["obs_emb"] = obs_emb.cpu().numpy()
        self.buf_obs["patch_logit"] = patch_logit.cpu().numpy()
        self.buf_obs["patch_prob"] = patch_prob.cpu().numpy()
        self.buf_obs["patch_label"] = patch_label.cpu().numpy()
        self.buf_obs["patch_mask"] = patch_mask.cpu().numpy()

        if not self._keep_raw_rgb:
            self.buf_obs.pop("obs_rgb") # delete raw rgb to save memory
            
        # Add target text embedding into obs
        self.buf_obs["text_emb"] = np.array([self.text_embeddings[i].cpu().numpy() for i in self.cur_target_index])
        if self.prompt_embeddings is not None:
            self.buf_obs["prompt_emb"] = np.array([self.prompt_embeddings[i].cpu().numpy() for i in self.cur_target])
    
    @torch.no_grad()
    def extend_clip_obs_single(self, obs: dict, target_index: int, index: int):
        # rgb_tensor = preprocess(np.asarray(obs["obs_rgb"], dtype=np.uint8)[None]).to(self.device)
        rgb_tensor = preprocess(np.asarray(obs["obs_rgb"], dtype=np.uint8)).to(self.device)
        obs_emb, patch_logit, patch_prob, patch_label, patch_mask = \
            self.clip_model.encode_and_segment_image(rgb_tensor, [target_index], self.text_embeddings)
        obs["obs_emb"] = obs_emb.cpu().numpy()[0]
        obs["patch_logit"] = patch_logit.cpu().numpy()[0]
        obs["patch_prob"] = patch_prob.cpu().numpy()[0]
        obs["patch_label"] = patch_label.cpu().numpy()[0]
        obs["patch_mask"] = patch_mask.cpu().numpy()[0]
        
        if not self._keep_raw_rgb:
            obs.pop("obs_rgb")
        
        obs["text_emb"] = self.text_embeddings[target_index].cpu().numpy()
        if self.prompt_embeddings is not None:
            obs["prompt_emb"] = self.prompt_embeddings[index].cpu().numpy()
    
    def remake(self) -> None:
        for env in self.envs:
            env.remake()

    def observe(self) -> Tuple[Any, Any, Any, Any]:
        if self.buf_rews is None:
            rews, obs, firsts, bads = zip(*[env.observe() for env in self.envs])
            self.buf_rews, self.buf_firsts, self.buf_bads = map(np.concatenate, (rews, firsts, bads))
            self.buf_obs = concat(obs)
            if self.clip_model is not None:
                self.extend_clip_obs()
        return self.buf_rews, self.buf_obs, self.buf_firsts, self.buf_bads

    def get_info(self) -> List[Dict]:
        return self.buf_infos

    def act(self, ac: Any) -> None:
        split_ac = split(ac, sections=np.cumsum([env.num for env in self.envs]))
        rews, obs, firsts, bads = zip(*[env.act(a) for env, a in zip(self.envs, split_ac)])
        self.buf_rews, self.buf_firsts, self.buf_bads = map(np.concatenate, (rews, firsts, bads))

        obs = list(obs)
        reset_indices = np.where(self.buf_firsts)[0]
        for index in reset_indices:
            # 1. Add last observation of previous episode to info
            self.buf_infos[index] = self.envs[index].get_info()
            self.buf_infos[index]["reset_ob"] = deepcopy(obs[index])
            if self.clip_model is not None:
                self.extend_clip_obs_single(self.buf_infos[index]["reset_ob"], self.cur_target_index[index], self.cur_target[index])
            # 2. Select the next target and reset the environment
            if self.pseudo_random:
                if len(self._target_queue) == 0:
                    self._target_queue.extend(np.random.permutation(len(self.target_name)).tolist())
                self.envs[index].reset(target=self._target_queue.popleft())
            elif self.fixed_target:
                self.envs[index].reset(target=index)
            else:
                self.envs[index].reset()
            # 3. Update self.cur_target
            self.cur_target[index] = self.envs[index].gym_env.cur_target
            self.cur_target_index[index] = self.target_index[self.cur_target[index]]
            # DEBUG
            # print(f"Env {index} reset to {self.cur_target[index]} (list id: {self.cur_target_index[index]})")
            # 4. Update obs
            obs[index] = multimap(lambda val: np.expand_dims(np.array(val), axis=0), self.envs[index].last_ob)
        
        self.buf_obs = concat(obs)
        if self.clip_model is not None:
            self.extend_clip_obs()

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        sizes = [env.num for env in self.envs]
        chunked_args = [_chunk_seq(arg, sizes) for arg in args]
        chunked_kwargs = {k: _chunk_seq(v, sizes) for k, v in kwargs.items()}
        result = []
        for chunk_idx, env in enumerate(self.envs):
            env_args = [arg[chunk_idx] for arg in chunked_args]
            env_kwargs = {k: v[chunk_idx] for k, v in chunked_kwargs.items()}
            result.extend(env.callmethod(method, *env_args, **env_kwargs))
        return result
