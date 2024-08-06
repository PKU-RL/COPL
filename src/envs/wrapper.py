from collections import defaultdict, deque
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np


class Wrapper(gym.Env):
    def __init__(self, env, ob_space=None, ac_space=None):
        self.ob_space = ob_space or env.ob_space
        self.ac_space = ac_space or env.ac_space
        self.num = env.num
        
        self.env = env
        
    def remake(self) -> None:
        return self.env.remake()

    def observe(self) -> Tuple[Any, Any, Any]:
        return self.env.observe()

    def get_info(self) -> List[Dict]:
        return self.env.get_info()

    def act(self, ac: Any) -> None:
        return self.env.act(ac)
    
    def close(self) -> None:
        return self.env.close()

    def callmethod(self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]):
        try:
            return getattr(self, method)(*args, **kwargs)
        except AttributeError:
            return self.env.callmethod(method, *args, **kwargs)


class EpisodeStatsWrapper(Wrapper):
    def __init__(self, env, buffer_size=100):
        super().__init__(env)
        self._buffer_size = buffer_size
        self.stats_buffers = defaultdict(partial(deque, maxlen=buffer_size))
        self.ep_returns = np.zeros(self.num, dtype=np.float)
        self.ep_lengths = np.zeros(self.num, dtype=np.int)
        self.ep_count = 0

    def act(self, ac):
        _, _, first, _ = self.observe()
        self.ep_returns[first] = 0
        self.ep_lengths[first] = 0
        self.env.act(ac)
        rew, _, first, _ = self.observe()
        self.ep_returns += rew
        self.ep_lengths += 1
        self.ep_count += first.sum()
        self.stats_buffers["r"].extend(self.ep_returns[first])
        self.stats_buffers["l"].extend(self.ep_lengths[first])
        self.stats_buffers["finish"].extend(np.logical_and(first, rew > 0)[first])

    def get_info(self):
        infos = self.env.get_info()
        _, _, first, _ = self.observe()
        for i, info in enumerate(infos):
            if first[i] and self.ep_lengths[i] > 0:
                ep_info = {
                    "r": self.ep_returns[i],
                    "l": self.ep_lengths[i],
                }
                info["episode"] = ep_info
        return infos

    def get_ep_stat_mean(self, key):
        return np.mean(self.stats_buffers[key]) if self.stats_buffers[key] else 0
    
    def reset_ep_stat(self):
        self.stats_buffers = defaultdict(partial(deque, maxlen=self._buffer_size))
        self.ep_returns = np.zeros(self.num, dtype=np.float)
        self.ep_lengths = np.zeros(self.num, dtype=np.int)
        self.ep_count = 0


class MTEpisodeStatsWrapper(Wrapper):
    def __init__(self, env, buffer_size=100):
        super().__init__(env)
        self._buffer_size = buffer_size
        self.stats_buffers = defaultdict(partial(deque, maxlen=buffer_size))
        self.ep_returns = np.zeros(self.num, dtype=np.float)
        self.ep_lengths = np.zeros(self.num, dtype=np.int)
        self.ep_count = 0

    def act(self, ac):
        _, _, first, _ = self.observe()
        self.ep_returns[first] = 0
        self.ep_lengths[first] = 0
        self.env.act(ac)
        rew, _, first, bad = self.observe()
        self.ep_returns += rew
        self.ep_lengths += 1
        self.ep_count += first.sum()
        self.stats_buffers["r"].extend(self.ep_returns[first])
        self.stats_buffers["l"].extend(self.ep_lengths[first])
        # SUCCESS: terminated but not truncated (kill one animal which is not the target)
        # CORRECT: terminated and rewarded (kill the target)
        self.stats_buffers["correct"].extend(np.logical_and(first, rew > 0)[first])
        self.stats_buffers["success"].extend(np.logical_and(first, np.logical_not(bad))[first])

        # Detailed stats
        if np.any(first):
            infos = self.env.get_info()
            indices = np.where(first)[0]
            for i in indices:
                self.stats_buffers["target_id"].append(infos[i]["target"])
                self.stats_buffers["success_id"].append(infos[i]["success"])

    def get_info(self):
        infos = self.env.get_info()
        _, _, first, _ = self.observe()
        for i, info in enumerate(infos):
            if first[i] and self.ep_lengths[i] > 0:
                ep_info = {
                    "r": self.ep_returns[i],
                    "l": self.ep_lengths[i],
                }
                info["episode"] = ep_info
        return infos

    def get_ep_stat_mean(self, key):
        return np.mean(self.stats_buffers[key]) if self.stats_buffers[key] else 0
    
    def get_task_stat(self):
        success = np.array(self.stats_buffers["success_id"])
        target = np.array(self.stats_buffers["target_id"])
        correct = success * target

        correct_rate = correct.sum(axis=0) / (target.sum(axis=0) + 1e-6)
        precisn_rate = correct.sum(axis=0) / ((target * success.sum(axis=-1, keepdims=True)).sum(axis=0) + 1e-6)
        return correct_rate, precisn_rate
    
    def reset_ep_stat(self):
        self.stats_buffers = defaultdict(partial(deque, maxlen=self._buffer_size))
        self.ep_returns = np.zeros(self.num, dtype=np.float)
        self.ep_lengths = np.zeros(self.num, dtype=np.int)
        self.ep_count = 0
