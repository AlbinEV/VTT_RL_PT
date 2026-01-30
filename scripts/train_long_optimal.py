#!/usr/bin/env python3
"""
LONG Training script with OPTIMAL parameters from grid search.
Best config: grid_029 (reward 413.21)

Optimal Parameters:
    W_FORCE_TRACKING = 8.0
    FORCE_ERROR_SCALE = 0.7
    W_ADAPTIVE_KPZ = 8.0
    OPTIMAL_KPZ_CENTER = 0.4
    W_KPZ_CHANGE_PENALTY = 2.0
    W_CONTACT_BONUS = 40.0

Usage:
    ./isaaclab.sh -p scripts/train_long_optimal.py --max_epochs 500 --num_envs 16
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime


# External drive for data storage
EXTERNAL_DRIVE = str(_path_setup.DATA_ROOT)

# ============================================================================
# SET OPTIMAL REWARD HYPERPARAMETERS (from grid_029)
# ============================================================================
os.environ["W_FORCE_TRACKING"] = "8.0"
os.environ["FORCE_ERROR_SCALE"] = "0.7"
os.environ["W_ADAPTIVE_KPZ"] = "8.0"
os.environ["OPTIMAL_KPZ_CENTER"] = "0.4"
os.environ["W_KPZ_CHANGE_PENALTY"] = "2.0"
os.environ["W_CONTACT_BONUS"] = "40.0"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Kp_z policy with OPTIMAL parameters")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--max_epochs", type=int, default=500, help="Max training epochs")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--experiment_name", type=str, default="kpz_optimal", help="Experiment name")
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper


def main():
    print("\n" + "="*70)
    print("  LONG TRAINING: Kp_z CONTROL with OPTIMAL PARAMETERS")
    print("="*70)
    print("\n📊 OPTIMAL HYPERPARAMETERS (from grid_029, reward=413.21):")
    print(f"   W_FORCE_TRACKING    = {os.environ['W_FORCE_TRACKING']}")
    print(f"   FORCE_ERROR_SCALE   = {os.environ['FORCE_ERROR_SCALE']}")
    print(f"   W_ADAPTIVE_KPZ      = {os.environ['W_ADAPTIVE_KPZ']}")
    print(f"   OPTIMAL_KPZ_CENTER  = {os.environ['OPTIMAL_KPZ_CENTER']}")
    print(f"   W_KPZ_CHANGE_PENALTY= {os.environ['W_KPZ_CHANGE_PENALTY']}")
    print(f"   W_CONTACT_BONUS     = {os.environ['W_CONTACT_BONUS']}")
    print("="*70 + "\n")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    print(f"✅ Environment created")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Num envs: {args.num_envs}")
    print(f"   Max epochs: {args.max_epochs}")
    
    # Wrap for RL-Games
    wrapped_env = RlGamesVecEnvWrapper(env, rl_device="cuda:0", clip_obs=10.0, clip_actions=1.0)
    
    # Register environment
    env_name = "kpz_optimal_env"
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
    print(f"\n📁 Saving to external drive: {EXTERNAL_DRIVE}")
    print(f"   Runs: {runs_dir}")
    print(f"   Checkpoints: {checkpoint_dir}")
    
    # RL-Games configuration (same as grid search but with more epochs)
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
                "save_best_after": 50,
                "save_frequency": 100,  # Save every 100 epochs
                "print_stats": True,
                "grad_norm": 1.0,
                "truncate_grads": True,
                "train_dir": runs_dir,
                "experiment_dir": checkpoint_dir,
                "player": {"render": False},
            }
        }
    }
    
    print(f"\n🚀 Starting training...")
    print(f"   Expected duration: ~{args.max_epochs * 0.5:.0f} minutes (estimate)\n")
    
    # Create and run
    runner = Runner()
    runner.load(rl_config)
    runner.reset()
    runner.run({"train": True})
    
    print(f"\n" + "="*70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"="*70)
    print(f"   Epochs: {args.max_epochs}")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   Runs: {runs_dir}")
    print(f"\n📌 To evaluate:")
    print(f"   ./isaaclab.sh -p scripts/extract_trajectories.py \\")
    print(f"       --checkpoint \"{checkpoint_dir}/nn/{run_name}.pth\" \\")
    print(f"       --num_episodes 5 --headless")
    print(f"="*70 + "\n")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
