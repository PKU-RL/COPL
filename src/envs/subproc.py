import contextlib
import multiprocessing as mp
import os
import pickle
import threading
import traceback
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces

from ..others.mineclip_official import tokenize_batch
from ..segment.segmineclip import SegVideoCLIP, preprocess
from ..segment.utils import encode_text_with_prompt_ensemble
from ..utils.utils import concat, multimap, split
from .utils import _chunk_seq, allsame

_clear_lock = threading.Lock()

CODECS = {"pickle": (pickle.dumps, pickle.loads)}

try:
    import cloudpickle
except ImportError:
    pass
else:
    CODECS["cloudpickle"] = (cloudpickle.dumps, cloudpickle.loads)


class SubprocError(Exception):
    pass


class SubprocEnv(object):
    """
    Create an environment in a subprocess using the provided function.

    :param env_fn: function to call to create the gym environment, defaults to `gym.make` 
    :param env_kwargs: keyword arguments to pass to env_fn
    :param daemon: if set to False, don't create daemon processes, the parent process will block
        when exiting until all non-daemon child processes have exited
    """

    def __init__(self, env_fn, env_kwargs=None, daemon=True):
        if env_kwargs is None:
            env_kwargs = {}
        # tensorflow is not fork safe, and fork doesn't work on all platforms anyway
        self._ctx = mp.get_context("spawn")
        self._p2c, c2p = self._ctx.Pipe()

        # pickle cannot pickle functions, so fallback to cloudpickle if pickle fails
        last_err = None
        for codec, (encode, decode) in CODECS.items():
            try:
                env_fn_serialized = encode(env_fn)
                env_kwargs_serialized = encode(env_kwargs)
            except Exception as e:
                last_err = e
            else:
                break
        else:
            raise Exception(
                f"all attempted encoders failed, tried: {', '.join(CODECS.keys())}. "
                f"Last error was:\n  {last_err}.\n\nIf you are pickling a function defined "
                f"inside of another function, try `pip install cloudpickle` to enable cloudpickle encoding"
            )

        self._child = self._ctx.Process(
            target=_worker,
            kwargs=dict(
                decode=decode,
                env_fn_serialized=env_fn_serialized,
                env_kwargs_serialized=env_kwargs_serialized,
                p2c=self._p2c,
                c2p=c2p,
            ),
            daemon=daemon,
        )
        # clear mpi vars to avoid issues with MPI_init being called in subprocesses
        with _clear_mpi_env_vars():
            self._child.start()
        # close child connection to avoid hangs when child exits unexpectedly
        c2p.close()
    
    def reset(self, target: Optional[int] = None) -> None:
        return self._call_method_in_worker("reset", target=target)
    
    def remake(self) -> None:
        return self._call_method_in_worker("remake")

    def observe(self) -> Tuple[Any, Any, Any, Any]:
        return self._call_method_in_worker("observe")

    def get_info(self) -> List[Dict]:
        return self._call_method_in_worker("get_info")

    def act(self, ac: Any) -> None:
        # self._call_method_in_worker_noreturn("act", ac=ac)
        return self._call_method_in_worker("act", ac=ac)

    def recv(self) -> Any:
        return self._recv_return_in_worker()

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        return self._call_method_in_worker("callmethod", method, *args, **kwargs)
    
    def callattr(self, attr: str) -> List[Any]:
        return self._call_method_in_worker("callattr", attr)

    def _communicate(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except (EOFError, ConnectionResetError):
            self._child.join()
            raise SubprocError(
                f"child process exited with exit code {self._child.exitcode}"
            )

    def _call_method_in_worker(self, method, *args, **kwargs):
        self._communicate(
            self._p2c.send,
            dict(send_response=True, method=method, args=args, kwargs=kwargs),
        )
    
    def _call_method_in_worker_noreturn(self, method, *args, **kwargs):
        self._communicate(
            self._p2c.send,
            dict(send_response=False, method=method, args=args, kwargs=kwargs),
        )
        
    def _recv_return_in_worker(self):
        result, err = self._communicate(self._p2c.recv)
        if err is not None:
            raise SubprocError from Exception("exception in subprocess:\n\n" + err)
        return result

    def close(self):
        if hasattr(self, "_child") and self._child is not None:
            if self._child.is_alive():
                self._p2c.send(None)
                self._child.join()
                self._p2c.close()
            self._child = None
            self._p2c = None

    def __del__(self):
        self.close()


def _worker(p2c, c2p, decode, env_fn_serialized, env_kwargs_serialized):
    try:
        p2c.close()
        result = None
        err = None
        try:
            env_fn = decode(env_fn_serialized)
            env_kwargs = decode(env_kwargs_serialized)
            env = env_fn(**env_kwargs)
        except Exception as e:
            err = traceback.format_exc()
            c2p.send((result, err))
            return
        else:
            result = (env.ob_space, env.ac_space, env.num)
            c2p.send((result, err))

        while True:
            msg = c2p.recv()
            if msg is None:
                # this is sent to tell the child to exit
                return

            result = None
            try:
                fn = getattr(env, msg["method"])
                result = fn(*msg["args"], **msg["kwargs"])
            except Exception as e:
                err = traceback.format_exc()
            if msg["send_response"]:
                c2p.send((result, err))
            # if send_response is False but an error occurred, the an error will be sent the next time
            # send_response is True
    except KeyboardInterrupt:
        print("Subproc worker: got KeyboardInterrupt")


class SubprocVectorEnv(object):
    def __init__(
        self, 
        envs: Sequence[SubprocEnv],
        clip_model: Optional[SegVideoCLIP] = None,
        target_list: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        pseudo_random: bool = False,
        fixed_target: bool = False,
        keep_raw_rgb: bool = False,
        serial_start: bool = False,
        device: Union[str, torch.device] = "cpu",
    ):
        ob_spaces, ac_spaces, nums = zip(
            *[env._recv_return_in_worker() for env in envs])
        assert allsame(ob_spaces)
        assert allsame(ac_spaces)
        
        self.envs = envs
        self.ob_space = ob_spaces[0]
        self.ac_space = ac_spaces[0]
        self.num = sum(nums)
        self.env_num = nums
        self.device = device

        self.clip_model = clip_model

        envs[0].callattr("initial_mobs")
        self.initial_mobs = envs[0]._recv_return_in_worker()[0]
        envs[0].callattr("number_mobs")
        self.number_mobs = envs[0]._recv_return_in_worker()[0]
        envs[0].callattr("target_names_vlm")
        self.target_name = envs[0]._recv_return_in_worker()[0]

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
        self.serial_start = serial_start
        # reset the environments
        if serial_start:
            # reset each environment serially to reduce network overhead
            print("Serially resetting environments...")
            if self.pseudo_random:
                for env in self.envs:
                    if len(self._target_queue) == 0:
                        self._target_queue.extend(np.random.permutation(len(self.target_name)).tolist())
                    env.reset(self._target_queue.popleft())
                    env._recv_return_in_worker()
            elif self.fixed_target:
                for i, env in enumerate(self.envs):
                    env.reset(target=i)
                    env._recv_return_in_worker()
            else:
                for env in self.envs:
                    env.reset()
                    env._recv_return_in_worker()
        else:
            print("Parallelly resetting environments...")
            if self.pseudo_random:
                for env in self.envs:
                    if len(self._target_queue) == 0:
                        self._target_queue.extend(np.random.permutation(len(self.target_name)).tolist())
                    env.reset(self._target_queue.popleft())
            elif self.fixed_target:
                for i, env in enumerate(self.envs):
                    env.reset(target=i)
            else:
                for env in self.envs:
                    env.reset()
            for env in self.envs:
                env._recv_return_in_worker()

        for env in self.envs:
            env.callattr("cur_target")
        self.cur_target = [env._recv_return_in_worker()[0] for env in self.envs]
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
        if self.serial_start:
            for env in self.envs:
                env.remake()
                env._recv_return_in_worker()
        else:
            for env in self.envs:
                env.remake()
            for env in self.envs:
                env._recv_return_in_worker()
    
    def observe(self) -> Tuple[Any, Any, Any, Any]:
        if self.buf_rews is None:
            for env in self.envs:
                env.observe()
            rews, obs, firsts, bads = zip(*[env._recv_return_in_worker() for env in self.envs])
            self.buf_rews, self.buf_firsts, self.buf_bads = map(np.concatenate, (rews, firsts, bads))
            self.buf_obs = concat(obs)
            if self.clip_model is not None:
                self.extend_clip_obs()
        return self.buf_rews, self.buf_obs, self.buf_firsts, self.buf_bads
    
    def get_info(self) -> List[Dict]:
        return self.buf_infos
        for env in self.envs:
            env.get_info()
        result = []
        for env in self.envs:
            result.append(env._recv_return_in_worker())
        return result
    
    def act(self, ac: Any) -> None:
        split_ac = split(ac, sections=np.cumsum(self.env_num))
        for env, a in zip(self.envs, split_ac):
            env.act(a)
        rews, obs, firsts, bads = zip(*[env._recv_return_in_worker() for env in self.envs])
        self.buf_rews, self.buf_firsts, self.buf_bads = map(np.concatenate, (rews, firsts, bads))

        obs = list(obs)
        reset_indices = np.where(self.buf_firsts)[0]
        for index in reset_indices:
            # 1. Add last observation of previous episode to info
            self.envs[index].get_info()
            self.buf_infos[index] = self.envs[index]._recv_return_in_worker()
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
            self.envs[index]._recv_return_in_worker()   # clear the buffer
            # 3. Update self.cur_target
            self.envs[index].callattr("cur_target")
            self.cur_target[index] = self.envs[index]._recv_return_in_worker()[0]
            self.cur_target_index[index] = self.target_index[self.cur_target[index]]
            # DEBUG
            # print(f"Env {index} reset to {self.cur_target[index]} (list id: {self.cur_target_index[index]})")
            # 4. Update obs
            self.envs[index].callattr("last_ob")
            last_ob = self.envs[index]._recv_return_in_worker()[0]
            obs[index] = multimap(lambda val: np.expand_dims(np.array(val), axis=0), last_ob)
        
        self.buf_obs = concat(obs)
        if self.clip_model is not None:
            self.extend_clip_obs()

    def callmethod(
        self, method: str, *args: Sequence[Any], **kwargs: Sequence[Any]
    ) -> List[Any]:
        sizes = self.env_num
        chunked_args = [_chunk_seq(arg, sizes) for arg in args]
        chunked_kwargs = {k: _chunk_seq(v, sizes) for k, v in kwargs.items()}
        for chunk_idx, env in enumerate(self.envs):
            env_args = [arg[chunk_idx] for arg in chunked_args]
            env_kwargs = {k: v[chunk_idx] for k, v in chunked_kwargs.items()}
            env.callmethod(method, *env_args, **env_kwargs)
        
        result = []
        for env in self.envs:
            result.extend(env._recv_return_in_worker())

        return result


@contextlib.contextmanager
def _clear_mpi_env_vars():
    """
    from mpi4py import MPI will call MPI_Init by default.  If we spawn a child process that also calls MPI_Init and
    has MPI environment variables defined, MPI will think that the child process is an MPI process just like the
    parent and do bad things such as hang or crash.

    This context manager is a hacky way to clear those environment variables temporarily such as when we are starting
    multiprocessing Processes.
    """
    with _clear_lock:
        removed_environment = {}
        for k, v in list(os.environ.items()):
            for prefix in ["OMPI_", "PMI_"]:
                if k.startswith(prefix):
                    removed_environment[k] = v
                    del os.environ[k]
        try:
            yield
        finally:
            os.environ.update(removed_environment)
