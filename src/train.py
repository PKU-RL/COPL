import atexit
import datetime
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from .common.buffer import Buffer
from .config import load_config
from .core import PPO, RND, Focal
from .envs.clip import CLIPReward_MT
from .envs.minecraft import MinecraftEnv
from .envs.venv import vectorize_gym
from .envs.wrapper import MTEpisodeStatsWrapper
from .segment.segmineclip import load
from .utils import logger

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


def train(config):
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
    
    # Setup logger
    log_dir = args.log_dir or "debug"  
    task_name = "multi_task"
    exp_name = WORKERMAP[args.algo].exp_name(args)
    if args.use_mineclip_reward:
        exp_name += "_mineclip"
    exp_name += f"_seed{args.seed}"
    exp_name = exp_name + "_" + datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, task_name, exp_name)
    logger.configure(dir=run_dir, format_strs=["csv", "stdout"] if args.stdout else ["csv"])
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)
    returns_log = os.path.join(run_dir, "returns.txt")
    success_log = os.path.join(run_dir, "success.txt")
    precisn_log = os.path.join(run_dir, "precisn.txt")
    
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
    
    # Create buffer
    buffer_size = worker.env.num * worker.n_steps
    buffer = Buffer(max_size=buffer_size, sequence_length=worker.n_steps)
    
    # Training
    n_iters = int(args.n_timesteps / worker.env.num / worker.n_steps)
    for i in range(n_iters):
        t_start = time.perf_counter()
        # Collect data
        info = worker.collect(scheduler_step=i, buffer=buffer)
        t_end_collect = time.perf_counter()
        
        # Learn on data
        stats_dict = worker.learn(scheduler_step=i, buffer=buffer)
        t_end_iter = time.perf_counter()
        
        # Get stats
        ret = worker.env.callmethod("get_ep_stat_mean", "r")
        len_ = worker.env.callmethod("get_ep_stat_mean", "l")
        correct = worker.env.callmethod("get_ep_stat_mean", "correct")
        success = worker.env.callmethod("get_ep_stat_mean", "success")
        precisn = correct / (success + 1e-8)

        correct_all, precisn_all = worker.env.callmethod("get_task_stat")
        correct_all = list(map(lambda x: f"{x:.2f}", correct_all))
        precisn_all = list(map(lambda x: f"{x:.2f}", precisn_all))

        # Logging
        logger.logkv("run/time_iter", t_end_iter - t_start)
        logger.logkv("run/fps", worker.env.num * worker.n_steps / (t_end_collect - t_start))
        logger.logkv("run/iter", i)
        logger.logkv("run/samples", (i + 1) * worker.env.num * worker.n_steps)
        logger.logkv("run/time_left", (n_iters - i - 1) * (t_end_iter - t_start))
        logger.logkv("env/return", ret)
        logger.logkv("env/length", len_)
        logger.logkv("env/success", correct)
        logger.logkv("env/precision", precisn)
        logger.logkv("env/success_all", "/".join(correct_all))
        logger.logkv("env/precision_all", "/".join(precisn_all))
        worker.log_spec_reward(buffer)
        if args.use_mineclip_reward:
            logger.logkv("env/mineclip_reward", buffer.get_stats_mean("mineclip_reward"))
        for key, value in stats_dict.items():
            logger.logkv("ppo/" + key, value)
        for key, value in info.items():
            logger.logkv("info/" + key, value)
        
        if i % config["run_cfg"]["log_interval"] == 0:
            logger.dumpkvs()
            # Save returns and success
            with open(returns_log, "a") as f:
                f.write("%f\n" % ret)
            with open(success_log, "a") as f:
                f.write("%f\n" % correct)
            with open(precisn_log, "a") as f:
                f.write("%f\n" % precisn)
        
        if i % config["run_cfg"]["save_interval"] == 0:
            cp_dir = os.path.join(run_dir, "checkpoints")
            if not os.path.exists(cp_dir):
                os.makedirs(cp_dir)
            torch.save(worker.policy.state_dict(), os.path.join(cp_dir, f"policy_{i}.pt"))

    # Save model
    torch.save(worker.policy.state_dict(), os.path.join(run_dir, "policy.pt"))


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
    parser.add_argument("--n_timesteps", type=int, default=int(4e7))
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=500)
    parser.add_argument("--use_parallel", action="store_true")
    parser.add_argument("--serial_start", action="store_true")
    
    # PPO
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--n_minibatches", type=int, default=4)
    parser.add_argument("--discount_gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--use_proper_time_limit", action="store_false", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--truncated_bptt", type=int, default=10)
    parser.add_argument("--ortho_init", action="store_false", default=True)
    parser.add_argument("--normalize_adv_mt", action="store_true")
    parser.add_argument("--use_pcgrad", action="store_true")
    
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
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stdout", action="store_false", default=True)
    
    args, unknown_args = parser.parse_known_args()
    
    worker = WORKERMAP[args.algo]
    
    worker.spec_parser(parser)
    args = parser.parse_args()
    
    # Build configuation
    config = {
        # Run
        "run_cfg": {
            "load_model": args.load_model,
            "log_dir": args.log_dir,
            "log_interval": args.log_interval,
            "save_interval": args.save_interval,
            "n_timesteps": args.n_timesteps,
            "stdout": args.stdout,
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
                    "break_speed_multiplier": args.break_speed_multiplier,
                    "always_night": args.always_night,
                },
                "pseudo_random": args.target_index == -1,
                "fixed_target": args.target_index == -2,
                "use_subproc": args.use_parallel,
                "serial_start": args.serial_start,
                "keep_raw_rgb": args.algo in ["rnd"],
                "seed": args.seed,
            },
            "optimizer_fn": "torch.optim.Adam",
            "optimizer_kwargs": {"lr": args.lr},
            "n_steps": args.n_steps,
            "n_epochs": args.n_epochs,
            "n_minibatches": args.n_minibatches,
            "discount_gamma": args.discount_gamma,
            "gae_lambda": args.gae_lambda,
            "use_proper_time_limit": args.use_proper_time_limit,
            "normalize_adv": True,
            "normalize_adv_mt": args.normalize_adv_mt,
            "clip_range": args.clip_range,
            "entropy_coef": args.entropy_coef,
            "max_grad_norm": args.max_grad_norm,
            "truncated_bptt": args.truncated_bptt,
            "use_pcgrad": args.use_pcgrad,
            "stdout": args.stdout,
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
        "segment_config": args.segment_config,
    }
    
    config["worker_kwargs"]["policy_fn"] = worker.policy_fn()
    config["worker_kwargs"]["policy_kwargs"] = load_config(args.actor_critic_config)
    config["worker_kwargs"]["policy_kwargs"].update(
        extractor_config=args.extractor_config,
        init_weight_fn="src.core.utils.orthogonal_init" if args.ortho_init else None,
    )
    
    worker.spec_config(config, args)
    
    train(config)
