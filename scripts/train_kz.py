#!/usr/bin/env python3
"""
Training Script for Kz-only Control (Thesis Section 4.6)
Uses RL-Games PPO with phase-dependent reward function.

This configuration trains an agent to control vertical stiffness (Kp_z) only,
matching the "OSC + Kz control" experiment from the thesis.

Author: Albin Bajrami
Institution: VTT Technical Research Centre of Finland and Università Politecnica delle Marche
"""
import _path_setup

import argparse
import os
import sys
import math
from datetime import datetime

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab initialization
parser = argparse.ArgumentParser(description="Train Kz-only Control Agent (Thesis Experiment)")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
parser.add_argument("--experiment_name", type=str, default="kz_only", help="Experiment name for logs")
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest after simulation is initialized
import gymnasium as gym

# Import Isaac Lab RL wrappers
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

# Import and register the environment
import robo_pp_fixed
from robo_pp_fixed import PolishEnvCfg
from robo_pp_fixed.cfg.scene_cfg import JustPushSceneCfg


def main():
    """Main training function for Kz-only control."""
    
    # Get package directory for config paths
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_root = os.path.join(pkg_dir, "logs", "rl_games", args_cli.experiment_name)
    os.makedirs(log_root, exist_ok=True)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Environment configuration
    env_cfg = PolishEnvCfg(
        scene=JustPushSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=2.0
        ),
        reward_type="kpz",  # Use Kz-only reward function
    )
    env_cfg.seed = args_cli.seed
    
    # RL-Games PPO configuration matching thesis Section 4.6
    agent_cfg = {
        "params": {
            "seed": args_cli.seed,
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
                        "sigma_init": {"name": "const_initializer", "val": 0},
                        "fixed_sigma": True
                    }
                },
                "mlp": {
                    "units": [512, 128, 64],  # Thesis: [512, 128, 64]
                    "activation": "elu",       # Thesis: ELU
                    "initializer": {"name": "default"},
                    "regularizer": {"name": "None"}
                }
            },
            "env": {
                "clip_observations": 100.0,
                "clip_actions": 1.0,
            },
            "config": {
                "name": args_cli.experiment_name,
                "env_name": "rlgpu",
                "device": "cuda:0",
                "device_name": "cuda:0",
                "multi_gpu": False,
                "ppo": True,
                "mixed_precision": False,
                "normalize_input": True,
                "normalize_value": True,
                "value_bootstrap": True,
                "num_actors": args_cli.num_envs,
                "reward_shaper": {"scale_value": 1.0},
                "normalize_advantage": True,
                # PPO hyperparameters from thesis Section 4.6
                "gamma": 0.99,
                "tau": 0.95,
                "learning_rate": 5e-5,
                "lr_schedule": "adaptive",
                "schedule_type": "standard",
                "kl_threshold": 0.008,
                "score_to_win": 100000,
                "max_epochs": args_cli.max_iterations,
                "save_best_after": 50,
                "save_frequency": 100,
                "print_stats": True,
                "grad_norm": 1.0,
                "entropy_coef": 0.002,
                "truncate_grads": True,
                "e_clip": 0.2,
                "horizon_length": 256,
                "minibatch_size": 256,
                "mini_epochs": 3,
                "critic_coef": 2,
                "clip_value": True,
                "seq_length": 4,
                "bounds_loss_coef": 0.0001,
                "train_dir": log_root,
                "full_experiment_name": log_dir,
            }
        }
    }
    
    # Create Isaac environment
    env = gym.make("Polish-Fixed-v0", cfg=env_cfg, render_mode=None)
    
    # Wrap for RL-Games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    
    # Register environment with RL-Games
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})
    
    # Update num_actors in config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    
    print(f"\n{'='*70}")
    print(f"  TRAINING: Kz-only Control (Thesis Section 4.6)")
    print(f"{'='*70}")
    print(f"  Reward function:    kpz (phase-dependent Kp_z control)")
    print(f"  Environments:       {args_cli.num_envs}")
    print(f"  Max iterations:     {args_cli.max_iterations}")
    print(f"  Log directory:      {os.path.join(log_root, log_dir)}")
    print(f"{'='*70}")
    print(f"  PPO Hyperparameters (Thesis):")
    print(f"    γ (gamma):        0.99")
    print(f"    λ (tau/GAE):      0.95")
    print(f"    ε (clip):         0.2")
    print(f"    lr:               5e-5 (adaptive)")
    print(f"    horizon:          256")
    print(f"    minibatch:        256")
    print(f"    epochs:           3")
    print(f"    entropy:          0.002")
    print(f"    network:          [512, 128, 64] ELU")
    print(f"{'='*70}\n")
    
    # Create runner and train
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    
    if args_cli.checkpoint:
        runner.run({"train": True, "play": False, "checkpoint": args_cli.checkpoint})
    else:
        runner.run({"train": True, "play": False})
    
    print("\n" + "="*70)
    print("  Training complete!")
    print("="*70)
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
