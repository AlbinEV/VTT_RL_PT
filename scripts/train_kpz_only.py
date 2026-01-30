#!/usr/bin/env python3
"""
Training script for Kp_z only control.
No damping ratio, only stiffness control along Z axis.

Usage:
    ./isaaclab.sh -p scripts/train_kpz_only.py --num_envs 16 --max_epochs 200
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime


# External drive for data storage (PC is full)
EXTERNAL_DRIVE = str(_path_setup.DATA_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Kp_z only policy")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--max_epochs", type=int, default=200, help="Max training epochs")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--experiment_name", type=str, default="kpz_only", help="Experiment name")
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper


def main():
    print("\n" + "="*60)
    print("TRAINING: Kp_z ONLY CONTROL")
    print("="*60)
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Wrap for RL-Games
    wrapped_env = RlGamesVecEnvWrapper(env, rl_device="cuda:0", clip_obs=10.0, clip_actions=1.0)
    
    # Register environment
    env_name = "kpz_only_env"
    vecenv.register(env_name, lambda config_name, num_actors, **kwargs: wrapped_env)
    env_configurations.register(env_name, {
        "vecenv_type": env_name,
        "env_creator": lambda **kwargs: wrapped_env,
    })
    
    # Timestamp for run
    timestamp = datetime.now().strftime("%m-%d-%H-%M")
    run_name = f"{args.experiment_name}_{timestamp}"
    
    # Paths on external drive
    runs_dir = os.path.join(EXTERNAL_DRIVE, "runs", run_name)
    checkpoint_dir = os.path.join(EXTERNAL_DRIVE, "checkpoints", run_name)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"📁 Saving data to external drive: {EXTERNAL_DRIVE}")
    print(f"   Runs: {runs_dir}")
    print(f"   Checkpoints: {checkpoint_dir}")
    
    # RL-Games configuration
    rl_config = {
        "params": {
            "seed": 42,
            "algo": {"name": "a2c_continuous"},
            "model": {"name": "continuous_a2c_logstd"},
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0.0},
                        "fixed_sigma": True,
                    }
                },
                "mlp": {
                    "units": [512, 128, 64],
                    "activation": "elu",
                    "initializer": {"name": "default"},
                }
            },
            "config": {
                "name": run_name,
                "env_name": env_name,
                "device": "cuda:0",
                "device_name": "cuda:0",
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": args.num_envs,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "e_clip": 0.2,
                "clip_value": True,
                "entropy_coef": 0.0,
                "critic_coef": 2.0,
                "bounds_loss_coef": 0.0,
                "learning_rate": 5e-5,
                "lr_schedule": "linear",
                "schedule_type": "legacy",
                "kl_threshold": 0.016,
                "mini_epochs": 4,
                "minibatch_size": 64,
                "horizon_length": 64,
                "max_epochs": args.max_epochs,
                "score_to_win": 100000,
                "save_best_after": 20,
                "save_frequency": 50,
                "print_stats": True,
                "grad_norm": 1.0,
                "truncate_grads": True,
                "train_dir": runs_dir,
                "experiment_dir": checkpoint_dir,
                "player": {"render": False},
            }
        }
    }
    
    # Create and run
    runner = Runner()
    runner.load(rl_config)
    runner.reset()
    runner.run({"train": True})
    
    print(f"\n✅ Training complete!")
    print(f"   Checkpoints saved in: {checkpoint_dir}")
    print(f"   Runs saved in: {runs_dir}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
