#!/usr/bin/env python3
"""
Minimal Training Script for Fixed Mode Polishing Environment
Uses RL-Games PPO algorithm with Isaac Lab

Author: Albin Bajrami
Institution: VTT Technical Research Centre of Finland and Università Politecnica delle Marche
"""
import _path_setup

import argparse
import os
import sys

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab initialization
parser = argparse.ArgumentParser(description="Train Fixed Mode Polishing Agent")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=500, help="Maximum training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--video", action="store_true", help="Record video during training")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest after simulation is initialized
import torch
import gymnasium as gym

# Import and register the environment
import robo_pp_fixed

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv

def main():
    """Main training function."""
    
    # Get package directory for config paths
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # RL-Games configuration
    rl_config = {
        "params": {
            "seed": 42,
            "algo": {
                "name": "a2c_continuous"
            },
            "model": {
                "name": "continuous_a2c_logstd"
            },
            "network": {
                "name": "actor_critic",
                "separate": False,
                "space": {
                    "continuous": {
                        "mu_activation": "None",
                        "sigma_activation": "None",
                        "mu_init": {"name": "default"},
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True
                    }
                },
                "mlp": {
                    "units": [256, 128, 64],
                    "activation": "elu",
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"}
                }
            },
            "config": {
                "name": "polish_fixed",
                "env_name": "Polish-Fixed-v0",
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": args_cli.num_envs,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 3e-4,
                "lr_schedule": "adaptive",
                "schedule_type": "standard",
                "kl_threshold": 0.008,
                "score_to_win": 100000,
                "max_epochs": args_cli.max_iterations,
                "save_best_after": 50,
                "save_frequency": 100,
                "print_stats": True,
                "grad_norm": 1.0,
                "entropy_coef": 0.0,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": 32,
                "minibatch_size": 16384,
                "mini_epochs": 4,
                "critic_coef": 2,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001
            }
        }
    }
    
    # Create environment factory
    def create_env(**kwargs):
        env = gym.make(
            "Polish-Fixed-v0",
            cfg=robo_pp_fixed.PolishEnvCfg(
                scene=robo_pp_fixed.PolishEnvCfg.scene.__class__(
                    num_envs=args_cli.num_envs
                )
            ),
            render_mode="rgb_array" if args_cli.video else None
        )
        return env
    
    # Register environment with RL-Games
    vecenv.register(
        "Polish-Fixed-v0",
        lambda config_name, num_actors, **kwargs: create_env()
    )
    env_configurations.register(
        "Polish-Fixed-v0",
        {"vecenv_type": "Polish-Fixed-v0", "env_creator": create_env}
    )
    
    # Create runner and train
    runner = Runner()
    runner.load(rl_config)
    
    if args_cli.checkpoint:
        runner.reset()
        runner.load_checkpoint(args_cli.checkpoint)
    
    print(f"\n{'='*60}")
    print(f"Training Fixed Mode Polishing Environment")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Max iterations: {args_cli.max_iterations}")
    print(f"{'='*60}\n")
    
    runner.run({"train": True})
    
    print("\nTraining complete!")
    
    # Close simulation
    simulation_app.close()

if __name__ == "__main__":
    main()
