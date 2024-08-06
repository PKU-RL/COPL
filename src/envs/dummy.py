from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .utils import transform_action


def equal(a, b, eps=1e-5):
    return np.abs(a - b) < eps


def cvt_action(act):
    if act[5] <= 1:
        return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
    elif act[5] == 3:
        return act[0] + 3*act[1] + 9*act[2] + 72
    else:
        raise Exception('Action[5] should be 0, 1, 3')


class DummyMinecraftEnv(gym.Env):
    def __init__(
        self,
        task_id,
        image_size=(160, 256),
        max_step=500,
        clip_model=None,
        target_name=None,
        **kwargs
    ):
        self.task_id = task_id
        self.observation_size = (3, *image_size)
        self._observation_space = None
        self._action_space = None
        self.max_step = max_step
        self.clip_model = clip_model
        self.target_name = target_name
        self.remake_env()

    @property
    def observation_space(self) -> spaces.Space:
        if self._observation_space is None:
            self._observation_space = spaces.Dict({
                    "compass": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                    "gps": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                    "voxels": spaces.Box(low=0, high=31, shape=(27,), dtype=np.int32),
                    "biome_id": spaces.Box(low=0, high=167, shape=(1,), dtype=np.int32),
                    "prev_action": spaces.Box(low=0, high=107, shape=(1,), dtype=np.int32),
            })
            if self.clip_model is not None:
                self._observation_space["obs_emb"] = spaces.Box(low=-1, high=1, shape=(512,), dtype=np.float32)
                if self.target_name is not None:
                    self._observation_space["target_emb"] = spaces.Box(low=-1, high=1, shape=(512,), dtype=np.float32)
                    self._observation_space["target_bbox"] = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
            else:
                self._observation_space["obs_rgb"] = spaces.Box(low=0, high=255, shape=self.observation_size, dtype=np.uint8)
        return self._observation_space

    @property
    def action_space(self) -> spaces.Space:
        if self._action_space is None:
            self._action_space = spaces.MultiDiscrete([12, 3])
        return self._action_space

    def remake_env(self):
        print('Environment remake: reset all the destroyed blocks!')
         
    def reset(self, **kwargs):
        self.cur_step = 0
        
        return self.observation_space.sample(), {}

    def step(self, action):
        self.cur_step += 1
        truncated = False
        if self.cur_step >= self.max_step:
            truncated = True
        
        return self.observation_space.sample(), 0, False, truncated, {}

    def close(self):
        print("MineDojoEnv closed.")
        
    def __del__(self):
        self.close()


class MoveDummyEnv(DummyMinecraftEnv):
    def __init__(
        self,
        task_id,
        image_size=(160, 256),
        max_step=500,
        clip_model=None,
        **kwargs
    ):
        """
        task_id: str
            dummy_env_move_random: return gps as moving randomly
            dummy_env_move_forward: return gpu as moving forward
            dummy_env_move_backward: return gpu as moving backward
            dummy_env_move_around: return gpu as moving around
            dummy_env_move_alter: return gpu as moving forward and left alternately
        """
        self._step_size = kwargs.get("step_size", 0.00025)
        super().__init__(task_id, image_size, max_step, clip_model, **kwargs)
    
    def _generate_obs(self):
        obs = self.observation_space.sample()
        
        if self.task_id in [
            "dummy_env_move_random",
            "dummy_env_move_forward", 
            "dummy_env_move_backward", 
            "dummy_env_move_around",
            "dummy_env_move_alter",
        ]:
            if self.cur_step == 0:
                obs["gps"] = self.observation_space["gps"].sample()
                if self.task_id == "dummy_env_move_around":
                    self._cur_goal = obs["gps"].copy()
                    self._cur_goal_num = 0
            elif self.task_id == "dummy_env_move_random":
                delta = self._step_size * np.eye(3)[np.random.randint(3)]
                obs["gps"] = self._last_gps + delta * np.random.choice([-1, 1])
            elif self.task_id == "dummy_env_move_forward":
                obs["gps"] = self._last_gps + np.array([self._step_size, 0, 0])
            elif self.task_id == "dummy_env_move_backward":
                obs["gps"] = self._last_gps - np.array([self._step_size, 0, 0])
            elif self.task_id == "dummy_env_move_around":
                if np.all(equal(self._last_gps, self._cur_goal)):
                    self._cur_goal_num += 1
                    delta = (self._cur_goal_num + 1) // 2 * 5 * self._step_size
                    delta *= (-1) ** ((self._cur_goal_num - 1) // 2)
                    if self._cur_goal_num % 2 == 1:
                        self._cur_goal[0] += delta
                    else:
                        self._cur_goal[2] += delta
                    # print("New goal:", self._cur_goal)
                
                obs["gps"] = self._last_gps.copy()
                dir = np.where(~equal(self._cur_goal, self._last_gps))[0]
                obs["gps"][dir] += self._step_size * np.sign(self._cur_goal[dir] - self._last_gps[dir])
            elif self.task_id == "dummy_env_move_alter":
                if self.cur_step % 2 == 1:
                    obs["gps"] = self._last_gps + np.array([self._step_size, 0, 0])
                else:
                    obs["gps"] = self._last_gps + np.array([0, 0, self._step_size])
                
        self._last_gps = obs["gps"].copy()
        return obs

    def reset(self, **kwargs):
        self.cur_step = 0
        
        return self._generate_obs(), {}

    def step(self, action):
        self.cur_step += 1
        truncated = False
        if self.cur_step >= self.max_step:
            truncated = True
        
        return self._generate_obs(), 0, False, truncated, {}


class GPSDummyEnv(DummyMinecraftEnv):
    def __init__(
        self,
        task_id,
        image_size=(160, 256),
        max_step=500,
        clip_model=None,
        **kwargs
    ):
        self._use_time_action =  task_id == "dummy_env_gps_ta"
        self._step_size = kwargs.get("step_size", 0.00025)
        super().__init__(task_id, image_size, max_step, clip_model, **kwargs)
    
    @property
    def observation_space(self) -> spaces.Space:
        if self._observation_space is None:
            self._observation_space = spaces.Dict({
                "gps": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            })
        if self._use_time_action:
            self._observation_space["gpst"] = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
            self._observation_space["prev_action"] = spaces.Box(low=0, high=107, shape=(1,), dtype=np.int32)
        return self._observation_space
    
    def _generate_obs(self):
        obs = {}
        if self.cur_step == 0:
            obs["gps"] = self._observation_space["gps"].sample()
        else:
            obs["gps"] = self.obs["gps"]
        if self._use_time_action:
            obs["gpst"] = np.concatenate([obs["gps"], [self.cur_step / self.max_step]])
            obs["prev_action"] = np.array([self._last_action])
        self.obs = deepcopy(obs)
        return obs

    def reset(self, **kwargs):
        self.cur_step = 0
        self._last_action = 0

        return self._generate_obs(), {}

    def step(self, action):
        self._last_action = cvt_action(transform_action(action))
        
        last_gps = self.obs["gps"]
        if action[0] == 1:  # forward
            self.obs["gps"] = last_gps + np.array([self._step_size, 0, 0])
        elif action[0] == 2:  # backward
            self.obs["gps"] = last_gps - np.array([self._step_size, 0, 0])
        elif action[0] == 3:  # left
            self.obs["gps"] = last_gps + np.array([0, 0, self._step_size])
        elif action[0] == 4:  # right
            self.obs["gps"] = last_gps - np.array([0, 0, self._step_size])

        self.cur_step += 1
        truncated = False
        if self.cur_step >= self.max_step:
            truncated = True
        
        return self._generate_obs(), 0, False, truncated, {}

    def close(self):
        print("MineDojoEnv closed.")
        
    def __del__(self):
        self.close()
