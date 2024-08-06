import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from ..utils.utils import multimap
from .concat import ConcatVectorEnv
from .subproc import SubprocEnv, SubprocVectorEnv


class FromGymEnv(object):
    def __init__(self, gym_env: gym.Env, render_mode=None):
        self.ob_space = gym_env.observation_space
        self.ac_space = gym_env.action_space
        self.num = 1
        self.gym_env = gym_env
        self.last_rew = 0.0
        self.last_first = True  # done
        self.last_bad = False   # time limit
        self.render_mode = render_mode
        
        self._init_reset = True
        
    def reset(self, target: Optional[int] = None, **kwargs) -> None:
        if self._init_reset:
            self._init_reset = False
            # Count time cost of initial reset
            start = time.perf_counter()
            self.last_ob, self.info = self.gym_env.reset(target=target, **kwargs)
            reset_time = round(time.perf_counter() - start)
            print(f"Reset took {reset_time // 60} mins and {reset_time % 60} seconds.")
        else:
            self.last_ob, self.info = self.gym_env.reset(target=target, **kwargs)
  
    def remake(self, **kwargs) -> None:
        self.gym_env.remake_env()
        start = time.perf_counter()
        self.last_ob, self.info = self.gym_env.reset(**kwargs)
        reset_time = round(time.perf_counter() - start)
        print(f"Reset took {reset_time // 60} mins and {reset_time % 60} seconds.")

    def observe(self) -> Tuple[Any, Any, Any, Any]:
        return (
            np.array([self.last_rew], "f"),
            multimap(lambda val: np.expand_dims(np.array(val), axis=0), self.last_ob),
            np.array([self.last_first], bool),
            np.array([self.last_bad], bool),
        )

    def get_info(self) -> List[Dict]:
        # self.observe()
        return self.info

    def act(self, ac: Any) -> None:
        # Check we got an action consistent with num_envs=1
        assert len(ac) == 1
        aczero = multimap(lambda x: x[0], ac)
        # Caution: in gymnasium, terminated won't be set to True if the episode is finished by time limit!
        self.last_ob, self.last_rew, terminated, truncated, self.info = self.gym_env.step(aczero)
        self.last_first = terminated or truncated
        self.last_bad = truncated
        if self.render_mode == "rgb_array":
            self.info["rgb"] = self.gym_env.render(mode="rgb_array")
        elif self.render_mode is not None:
            self.gym_env.render(mode=self.render_mode)
        # if self.last_first:
        #     reset_ob = self.last_ob
        #     self.last_ob, _ = self.gym_env.reset()
        #     self.info["reset_ob"] = reset_ob # last observation of previous episode
        return self.observe()   # return observation after taking action

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        call_args = [arg[0] for arg in args]
        call_kwargs = {k: v[0] for k, v in kwargs.items()}
        return [getattr(self.gym_env, method)(*call_args, **call_kwargs)]
    
    def callattr(self, attr: str) -> List[Any]:
        if hasattr(self, attr):
            return [getattr(self, attr)]
        return [getattr(self.gym_env, attr)]


def _make_gym_env(env_fn, env_kwargs, render_mode=None, seed=None, device="cpu"):
    gym_env = env_fn(seed=seed, device=device, **env_kwargs)
    env = FromGymEnv(gym_env, render_mode=render_mode)
    return env


def vectorize_gym(
    num,
    env_fn=None,
    env_kwargs=None,
    pseudo_random=False,
    fixed_target=False,
    use_subproc=True,
    keep_raw_rgb=False,
    serial_start=False,
    render_mode=None,
    seed=None,
    device="cpu",
):
    if env_fn is None:
        import gym

        env_fn = gym.make
    if env_kwargs is None:
        env_kwargs = {}

    if env_kwargs.get("clip_model", None) is not None:
        model_clip = env_kwargs.pop("clip_model")
        target_list = env_kwargs.pop("target_list", None)
        prompts = env_kwargs.pop("prompts", None)
    else:
        model_clip = None
        target_list = None
        prompts = None
        
    if env_kwargs["target_index"] is not None:
        print(f"Predefined target index: {env_kwargs['target_index']}.")
    
    biome_list = env_kwargs.pop("biome", None)
    if len(biome_list) == 1:
        print(f"Envs are in biome: {biome_list[0]}")
        biome_list *= num
    else:
        assert len(biome_list) == num, "Number of biomes should be 1 or equal to number of environments."
        print("Envs are in biomes:", ", ".join(biome_list))
    
    if use_subproc and num > 1:
        envs = []
        for idx in range(num):
            env_kwargs.update(biome=biome_list[idx])
            envs.append(
                SubprocEnv(
                    env_fn=_make_gym_env,
                    env_kwargs=dict(
                        env_fn=env_fn,
                        env_kwargs=env_kwargs,
                        render_mode=render_mode,
                        seed=seed + idx if seed is not None else None,
                        device=device
                    )
                )
            )

        return SubprocVectorEnv(envs, model_clip, target_list, prompts, pseudo_random, fixed_target, keep_raw_rgb, 
                                serial_start, device)
    
    envs = []
    for idx in range(num):
        env_kwargs.update(biome=biome_list[idx])
        envs.append(
            _make_gym_env(
                env_fn=env_fn,
                env_kwargs=env_kwargs,
                render_mode=render_mode,
                seed=seed + idx if seed is not None else None,
                device=device,
            )
        )
    
    return ConcatVectorEnv(envs, model_clip, target_list, prompts, pseudo_random, fixed_target, keep_raw_rgb, device)


def env_fn(task_id, clip_model=None, target_name=None, seed=None, device=None):
    from .dummy import DummyMinecraftEnv
    return DummyMinecraftEnv(task_id, clip_model=clip_model, target_name=target_name)


if __name__ == "__main__":
    from ..segment.segmineclip import load
    
    model, _ = load()
    
    evn_kwargs = {
        "task_id": "harvest_milk_with_empty_bucket_and_cow",
        "clip_model": model,
        "target_name": "cow",
    }
    
    venv = vectorize_gym(2, env_fn, evn_kwargs, use_subproc=False, device="cuda:0")
    print("ConcatEnv created.")

    _, obs, _, _ = venv.observe()
    venv.callmethod("close")

    evn_kwargs = {
        "task_id": "harvest_milk_with_empty_bucket_and_cow",
        "clip_model": model,
        "target_name": "cow",
    }

    venv = vectorize_gym(2, env_fn, evn_kwargs, use_subproc=True, device="cuda:0")
    print("SubprocEnv created.")

    _, obs, _, _ = venv.observe()
    venv.callmethod("close")
