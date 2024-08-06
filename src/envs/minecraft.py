import json
from collections import deque
from typing import Any, List, Optional, Tuple, Union

import gymnasium as gym
import minedojo
import numpy as np
import torch
from gymnasium import spaces
from minedojo.tasks.meta.multi_task import HARVEST_TARGET2VLM

from ..segment.segmineclip import SegVideoCLIP, preprocess
from ..segment.utils import encode_text_with_prompt_ensemble
from .utils import _parse_inventory_dict, preprocess_obs, transform_action


class MinecraftEnv(gym.Env):
    def __init__(
        self,
        # env setting
        image_size: Tuple = (160, 256),
        max_step: int = 500,
        biome: Optional[str] = None,
        camera_interval: int = 2,
        # multi-task mob setting
        multi_task_config: Optional[str] = None,
        # clip model
        clip_model: Optional[SegVideoCLIP] = None,
        target_list: Optional[str] = None,
        target_index: Optional[int] = None,
        # misc
        force_slow_reset_interval: Optional[int] = None,
        break_speed_multiplier: float  = 1.0,
        always_night: bool = False,
        device: Union[str, torch.device] = "cpu",
        seed: int = 0,
        **kwargs: Any,
    ):
        self.observation_size = (3, *image_size)
        self._observation_space = None
        self._action_space = None

        self.biome = biome
        self.image_size = image_size
        self.camera_interval = camera_interval
        self.max_step = max_step
        self.cur_step = 0
        self.device = device
        self.seed = seed
        self.break_speed_multiplier = break_speed_multiplier
        self.kwargs = kwargs

        self.clip_model = clip_model # use mineclip model to precompute embeddings
        self.cur_target = None

        with open(multi_task_config, "r") as f:
            config = json.load(f)
        self.initial_mobs = config["initial_mobs"]
        self.number_mobs = config["number_mobs"]
        self._mob_spawn_range = config["mob_spawn_range"]
        self.target_names = config["target_names"]
        self.target_names_vlm = []
        for t in self.target_names:
            if t in HARVEST_TARGET2VLM:
                self.target_names_vlm.append(HARVEST_TARGET2VLM[t])
            else:
                self.target_names_vlm.append(t)
        self.custom_commands = config["reset_cmds"]
        self.initial_inventory = config.get("initial_inventory", None)
        self.predef_target = target_index
        if clip_model is not None and target_list is not None:
            self.init_texts(target_list)

        # special task: FIND
        self._find_task = False
        if config.get("find_mob", None) is not None:
            self._find_task = True
            self._find_target = config["find_mob"]
        
        self._always_night = always_night
        self.remake_env()
        # self.task_prompt = self.base_env.task_prompt
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
        self._slow_reset_interval = force_slow_reset_interval
        self._slow_reset_counter = -1
    
    @property
    def observation_space(self) -> spaces.Space:
        if self._observation_space is not None:
            return self._observation_space
        
        obs_dict = {
            "compass": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
            "gps": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            "voxels": spaces.Box(low=0, high=31, shape=(27,), dtype=np.int32),
            "biome_id": spaces.Box(low=0, high=167, shape=(1,), dtype=np.int32),
            "prev_action": spaces.Box(low=0, high=107, shape=(1,), dtype=np.int32),
            "target_index": spaces.Box(low=0, high=1, shape=(len(self.target_names),), dtype=bool),
        }
        if self.clip_model is not None:
            obs_dict["obs_emb"] = spaces.Box(low=-1, high=1, shape=(512,), dtype=np.float32)
        else:
            obs_dict["obs_rgb"] = spaces.Box(low=0, high=255, shape=self.observation_size, dtype=np.uint8)
        self._observation_space = spaces.Dict(obs_dict)    

        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        if self._action_space is None:
            self._action_space = spaces.MultiDiscrete([12, 3])
        return self._action_space
    
    @torch.no_grad()
    def init_texts(self, target_list: str):
        with open(target_list, "r") as f:
            all_texts = f.read().splitlines()
        for t in self.target_names_vlm:
            if t not in all_texts:
                all_texts.append(t)
        self.text_embeddings = encode_text_with_prompt_ensemble(self.clip_model, all_texts, self.device)
        self.target_index = [all_texts.index(t) for t in self.target_names_vlm]
    
    @torch.no_grad()
    def encode_rgb(self, obs: dict):
        rgb_tensor = preprocess(np.asarray(obs["rgb"], dtype=np.uint8)).view(1, *self.observation_size).to(self.device)
        obs["obs_emb"] = self.clip_model.encode_image(rgb_tensor).cpu().numpy()

    def _find_distance_to_entity_if_in_sight(self, obs):
        assert self._find_task, "This function is only for FIND task!"
        in_sight, min_distance = False, None
        entities, distances = (
            obs["rays"]["entity_name"],
            obs["rays"]["entity_distance"],
        )
        entity_idx = np.where(entities == self._find_target)[0]
        if len(entity_idx) > 0:
            in_sight = True
            min_distance = np.min(distances[entity_idx])
        return in_sight, min_distance

    def remake_env(self):
        '''
        call this to reset all the blocks and trees
        should modify line 479 in minedojo/tasks/__init__.py, deep copy the task spec dict:
            import deepcopy
            task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
        '''
        if hasattr(self, 'base_env'):
            self.base_env.close()

        env_kwargs = {
            "task_id": "multi-task",
            "image_size": self.image_size,
        }
        if self.initial_mobs is not None:
            env_kwargs.update(
                initial_mobs = self.initial_mobs * self.number_mobs,
                initial_mob_spawn_range_low = (-self._mob_spawn_range, 1, -self._mob_spawn_range),
                initial_mob_spawn_range_high = (self._mob_spawn_range, 1, self._mob_spawn_range),
            )
        env_kwargs.update(
            target_names = self.target_names,
            target_quantities = 1,
            reward_weights = 1,
            fast_reset_random_teleport_range_high = 200,
            fast_reset_random_teleport_range_low = 0,
            specified_biome = self.biome,
            use_voxel = True,
            voxel_size = dict(xmin=-1,ymin=-1,zmin=-1,xmax=1,ymax=1,zmax=1),
            start_at_night = False,
            always_night = self._always_night,
            allow_mob_spawn = False,
            break_speed_multiplier = self.break_speed_multiplier,
        )
        if self.initial_inventory is not None:
            env_kwargs.update(initial_inventory = _parse_inventory_dict(self.initial_inventory))

        if self._find_task:
            env_kwargs.update(
                use_lidar = True,
                lidar_rays=[
                    (np.pi * pitch / 180, np.pi * yaw / 180, 999)
                    for pitch in np.arange(-30, 30, 6)
                    for yaw in np.arange(-60, 60, 10)
                ]
            )
        self.base_env = minedojo.make(**env_kwargs)
        self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')

    def reset(self, target: Optional[int] = None, **kwargs):
        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.base_env.unwrapped.execute_cmd(cmd)
            self.base_env.unwrapped.set_time(6000)
            self.base_env.unwrapped.set_weather("clear")
        self._first_reset = False
        self.prev_action = self.base_env.action_space.no_op()

        force_slow_reset = False
        if self._slow_reset_interval is not None:
            self._slow_reset_counter += 1
            if self._slow_reset_counter >= self._slow_reset_interval:
                force_slow_reset = True
                self._slow_reset_counter = 0
            
        if target is not None:
            # parameter target is more prior than self.predef_target
            self.cur_target = target
        elif self.predef_target is not None:
            self.cur_target = self.predef_target
        else:
            self.cur_target = np.random.choice(len(self.target_names))
        # DEBUG
        # print(f"Choose target: {self.target_names[self.cur_target]}")
        obs = self.base_env.reset(
            force_slow_reset=force_slow_reset,
            custom_cmds=self.custom_commands[self.cur_target]
        )
        self.cur_step = 0

        obs['prev_action'] = self.prev_action
        if self.clip_model is not None:
            self.encode_rgb(obs)

        obs = preprocess_obs(obs)
        obs["target_index"] = np.eye(len(self.target_names), dtype=bool)[self.cur_target]

        return obs, {}

    def step(self, action):
        action = transform_action(action, self.camera_interval)
        obs, reward, terminated, info = self.base_env.step(action)
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            truncated = True
        else:
            truncated = False
        
        info["target"] = self.cur_target
        info["target"] = np.eye(len(self.target_names), dtype=bool)[self.cur_target].tolist()
        if self._find_task:
            entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
            if entity_in_sight and distance < 3.0:
                reward = 1.
                terminated = True
            else:
                reward = 0.
                terminated = False
        else:
            if not info["success"][self.cur_target]:
                reward = 0.
            else:
                reward = 1.
        
        if self.clip_model is not None:
            self.encode_rgb(obs)

        obs['prev_action'] = self.prev_action
        self.prev_action = action # save the previous action for the agent's observation

        obs = preprocess_obs(obs)
        obs["target_index"] = np.eye(len(self.target_names), dtype=bool)[self.cur_target]

        return obs, reward, terminated, truncated, info
    
    def close(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()
            print("MineDojoEnv closed.")

    def __del__(self):
        self.close()


if __name__ == "__main__":
    env = MinecraftEnv(
        # biome="plains",
        biome="flat",
        device="cuda:0",
        seed=0,
    )

    buf_obs = []

    obs = env.reset()[0]
    buf_obs.append(obs["obs_rgb"])

    for _ in range(50):
        # obs = env.step(env.action_space.sample())[0]
        act = [10, 0]
        obs = env.step(act)[0]
        buf_obs.append(obs["obs_rgb"])

    print("Test passed.")

    buf_obs = np.stack(buf_obs)
    buf_obs = np.transpose(buf_obs, (0, 2, 3, 1))
    import imageio
    imageio.mimsave("test.mp4", buf_obs, fps=10)
    