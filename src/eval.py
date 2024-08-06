import atexit
import datetime
import json
import os

import cv2
import gymnasium as gym
import numpy as np
import torch

FONT = cv2.FONT_HERSHEY_SIMPLEX

from .common.buffer import Buffer
from .config import load_config
from .core import PPO, RND, Focal
from .envs.clip import CLIPReward_MT
from .envs.minecraft import MinecraftEnv
from .envs.venv import vectorize_gym
from .envs.wrapper import MTEpisodeStatsWrapper
from .segment.segmineclip import load
from .utils.utils import multimap, split_episode

WORKERMAP = {
    "ppo": PPO,
    "focal": Focal,
    "rnd": RND,
}


def make_minecraft_env(**kwargs) -> gym.Env:
    return MinecraftEnv(**kwargs)
    

def make_vector_env(**kwargs):
    env = vectorize_gym(**kwargs)
    env = MTEpisodeStatsWrapper(env)
    return env


def eval(config):
    # Setup cuda
    if torch.cuda.is_available():
        print("Choose to use gpu...")
        config["worker_kwargs"]["device"] = "cuda:0"
        torch.set_num_threads(args.n_envs)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        print("Choose to use cpu...")
        config["worker_kwargs"]["device"] = "cpu"
        torch.set_num_threads(args.n_envs)
    config["worker_kwargs"]["env_kwargs"]["device"] = config["worker_kwargs"]["device"]
    
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Load CLIP model
    if args.use_clip:
        kwargs = {}
        if args.segment_config is not None:
            kwargs = load_config(args.segment_config)
        model_clip, _ = load(device=config["worker_kwargs"]["device"], **kwargs)
        print('MineCLIP model loaded.')
        config["worker_kwargs"]["env_kwargs"]["env_kwargs"]["clip_model"] = model_clip
        with open(args.multi_task_config, "r") as f:
            mt_config = json.load(f)
        task_prompts = mt_config["mineclip_prompts"]
        config["worker_kwargs"]["env_kwargs"]["env_kwargs"]["prompts"] = task_prompts
        
    if args.use_mineclip_reward:
        assert args.use_clip, "use_mineclip_reward must be used with use_clip"

        with open(args.multi_task_config, "r") as f:
            mt_config = json.load(f)
        task_prompts = mt_config["mineclip_prompts"]
        print('All task prompts:')
        for t in task_prompts:
            print(t)
        config["worker_kwargs"]["mineclip_reward"] = CLIPReward_MT(
            model_clip, task_prompts, device=config["worker_kwargs"]["device"], 
            num_frames=config["worker_kwargs"]["mineclip_num_frames"])
    config["worker_kwargs"].pop("mineclip_num_frames")
    
    worker = WORKERMAP[args.algo](**config["worker_kwargs"])
    atexit.register(worker.close)
    
    if args.load_model is not None:
        worker.policy.load_state_dict(
            torch.load(args.load_model, map_location=config["worker_kwargs"]["device"]),
            strict=False,
        )
    
    # Rollout worker
    batch, _, next_first, _ = worker.collect_batch(0, deterministic=args.eval_deterministic)
    
    mineclip_reward = None
    if args.use_mineclip_reward:
        mineclip_reward = worker.mineclip_reward.compute_reward(batch["obs"]["obs_emb"], 
                                                                batch["next_obs"]["obs_emb"],
                                                                batch["first"],
                                                                batch["target"])
    for key in batch:
        batch[key] = multimap(lambda val: val.swapaxes(0, 1), batch[key])
    
    # Get stats
    ret = worker.env.callmethod("get_ep_stat_mean", "r")
    len_ = worker.env.callmethod("get_ep_stat_mean", "l")
    correct = worker.env.callmethod("get_ep_stat_mean", "correct")
    success = worker.env.callmethod("get_ep_stat_mean", "success")
    precision = correct / (success + 1e-8)
    first = np.concatenate([batch["first"], next_first[..., None]], axis=1)
    
    print(f"Success: {correct:.2f}, "
          f"Precision: {precision:.2f}, "
          f"Length: {len_:.2f}")
        
    if args.save_gif or args.save_mp4:
        import imageio

        save_dir = args.save_dir or "data"
        save_dir = os.path.join(save_dir, "videos", "multi_task")
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        if args.save_tag is not None:
            save_dir = os.path.join(save_dir, args.save_tag + "-" + timestamp)
        else:
            save_dir = os.path.join(save_dir, timestamp)
        save_dir += f"-{args.target_index}"
        os.makedirs(save_dir, exist_ok=True)

        imgs = batch["obs"]["obs_rgb"]                  # [B, T, C, H, W]
        imgs = np.transpose(imgs, (0, 1, 3, 4, 2))      # [B, T, H, W, C]
        
        if not args.hide_info:
            # Show mineclip reward on the top-left corner
            if mineclip_reward is not None:
                mineclip_reward = mineclip_reward.swapaxes(0, 1)
                for b in range(imgs.shape[0]):
                    for t in range(imgs.shape[1]):
                        um = cv2.UMat(imgs[b, t])
                        cv2.putText(um, f"{mineclip_reward[b, t]:.3f}", (40, 20), FONT, 0.5, (255, 0, 0), 1)
                        imgs[b, t] = um.get()
            # Show Focal reward on the top-left corner and heat map
            elif "focal" in args.algo:
                B, T = imgs.shape[:2]
                for b in range(B):
                    for t in range(T):
                        um = cv2.UMat(imgs[b, t])
                        rew = batch["intrinsic_reward"][b, t]
                        cv2.putText(um, f"{rew:.3f}", (10, 20), FONT, 0.5, (255, 0, 0), 1)
                        imgs[b, t] = um.get()
                imgs_logit = batch["obs"]["patch_logit"].repeat(16, 2).repeat(16, 3)
                imgs_logit = ((imgs_logit[..., None].repeat(3, -1) - 0.2) / 0.2 * 255.).astype(np.uint8)
                imgs_prob = batch["obs"]["patch_prob"].repeat(16, 2).repeat(16, 3)
                imgs_prob = (imgs_prob[..., None].repeat(3, -1) * 255.).astype(np.uint8)
                imgs = np.concatenate([imgs, imgs_logit, imgs_prob], axis=3)

        imgs = split_episode(imgs, first)
        rew = split_episode(batch["reward"], first)
        for i, img in enumerate(imgs):
            if args.save_gif:
                imageio.mimsave(os.path.join(save_dir, f"{i}-{rew[i].sum():.1f}.gif"), img, fps=15)
            if args.save_mp4:
                imageio.mimsave(os.path.join(save_dir, f"{i}-{rew[i].sum():.1f}.mp4"), img, fps=15, macro_block_size=None)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # MineCraft
    parser.add_argument("--max_timesteps", type=int, default=500)
    parser.add_argument("--multi_task_config", type=str, default="src/config/env/multi_tasks/combat_1.json")
    parser.add_argument("--use_clip", action="store_false", default=True)
    parser.add_argument("--random_init", type=int, default=200)
    parser.add_argument("--biome_id", type=str, nargs="+", default=["plains"])
    parser.add_argument("--camera_interval", type=int, default=2)
    parser.add_argument("--target_list", type=str, default="src/envs/keyword.txt")
    parser.add_argument("--target_index", type=int, default=None, 
                        help="-1: pseudo-random; None: random; 0~: pre-defined")
    parser.add_argument("--segment_config", type=str, default=None)
    parser.add_argument("--force_slow_reset_interval", type=int, default=None)
    parser.add_argument("--break_speed_multiplier", type=float, default=1.0)
    parser.add_argument("--always_night", action="store_true")
    
    # general
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--use_parallel", action="store_true")
    parser.add_argument("--serial_start", action="store_true")
    parser.add_argument("--eval_deterministic", action="store_true")
    
    parser.add_argument("--extrinsic_reward_coef", type=float, default=1.0)
    # episodic curiosity
    parser.add_argument("--intrinsic_reward_coef", type=float, default=0.0)
    parser.add_argument("--intrinsic_reward_norm", "-irn", action="store_true")
    parser.add_argument("--intrinsic_reward_fuse", type=str, default="add",
                        choices=["add", "max", "min", "mul"])
    
    # mineclip reward
    parser.add_argument("--use_mineclip_reward", action="store_true")
    parser.add_argument("--mineclip_reward_coef", type=float, default=0.0)
    parser.add_argument("--mineclip_num_frames", type=int, default=16)
    
    # configs
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "focal", "rnd"])
    parser.add_argument("--extractor_config", type=str, default="src/config/clip/feats_all.json")
    parser.add_argument("--actor_critic_config", type=str, default="src/config/ac/rnn_policy.json")
    
    # misc
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--save_mp4", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_tag", type=str, default=None)
    parser.add_argument("--hide_info", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    
    args, unknown_args = parser.parse_known_args()
    
    worker = WORKERMAP[args.algo]
    
    worker.spec_parser(parser)
    args = parser.parse_args()
    
    # Build configuation
    config = {
        # Run
        "run_cfg": {
            "load_model": args.load_model,
            "save_gif": args.save_gif,
            "save_mp4": args.save_mp4,
            "save_dir": args.save_dir,
            "save_tag": args.save_tag,
            "hide_info": args.hide_info,
        },
        # Agent
        "worker_kwargs": {
            "env_fn": make_vector_env,
            "env_kwargs": {
                "env_fn": make_minecraft_env,
                "num": args.n_envs,
                "env_kwargs": {
                    "max_step": args.max_timesteps,
                    "multi_task_config": args.multi_task_config,
                    "clip_model": None,
                    "target_list": args.target_list,
                    "target_index": args.target_index,
                    "biome": args.biome_id,
                    "camera_interval": args.camera_interval,
                    "fast_reset_random_teleport_range_high":
                        None if args.random_init == 0 else args.random_init,
                    "fast_reset_random_teleport_range_low":
                        None if args.random_init == 0 else 0,
                    "force_slow_reset_interval": args.force_slow_reset_interval,
                    # "break_speed_multiplier": args.break_speed_multiplier,
                    "always_night": args.always_night,
                },
                "pseudo_random": args.target_index == -1,
                "fixed_target": args.target_index == -2,
                "use_subproc": args.use_parallel,
                "serial_start": args.serial_start,
                "keep_raw_rgb": args.save_gif or args.save_mp4,
                "seed": args.seed,
            },
            "optimizer_fn": "torch.optim.Adam",
            "optimizer_kwargs": {"lr": 0.0005},
            "n_steps": args.n_steps,
            "n_epochs": 1,
            "n_minibatches": 1,
            "extrinsic_reward_coef": args.extrinsic_reward_coef,
            # episodic curiosity
            "intrinsic_reward_coef": args.intrinsic_reward_coef,
            "intrinsic_reward_norm": args.intrinsic_reward_norm,
            "intrinsic_reward_fuse": args.intrinsic_reward_fuse,
            # mineclip reward
            "mineclip_reward": None,
            "mineclip_reward_coef": args.mineclip_reward_coef,
            "mineclip_num_frames": args.mineclip_num_frames,
        },
    }
    
    config["worker_kwargs"]["policy_fn"] = worker.policy_fn()
    config["worker_kwargs"]["policy_kwargs"] = load_config(args.actor_critic_config)
    config["worker_kwargs"]["policy_kwargs"].update(
        extractor_config=args.extractor_config,
        init_weight_fn=None,
    )
    
    worker.spec_config(config, args)
    
    eval(config)
