#!/usr/bin/env python3
"""
Training script for safe Kp_z control - avoids force peaks.
Lower learning rate (1e-4) and more exploration for smoother control.
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime


EXTERNAL_DRIVE = str(_path_setup.DATA_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--max_epochs", type=int, default=500)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--experiment_name", type=str, default="kpz_safe_v1")
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rl_games.torch_runner import Runner
from rl_games.common import vecenv, env_configurations
import yaml
import robo_pp_fixed

print("\n" + "="*60)
print("TRAINING: Safe Kp_z (LR=1e-4, Entropy=0.001)")
print("="*60)
print(f"Experiment: {args.experiment_name}")
print(f"Num envs: {args.num_envs} | Epochs: {args.max_epochs}")

timestamp = datetime.now().strftime("%m-%d-%H-%M")
exp_name = f"{args.experiment_name}_{timestamp}"
output_dir = os.path.join(EXTERNAL_DRIVE, "runs", exp_name) if os.path.exists(EXTERNAL_DRIVE) else os.path.join("outputs", exp_name)
os.makedirs(output_dir, exist_ok=True)

print(f"Output: {output_dir}\n")

# Environment name
env_name = "kpz_safe_env"

# RL-Games config with SAFE hyperparameters
config = {
    'params': {
        'seed': 42,
        'algo': {'name': 'a2c_continuous'},
        'model': {'name': 'continuous_a2c_logstd'},
        'network': {
            'name': 'actor_critic',
            'separate': False,
            'space': {
                'continuous': {
                    'mu_activation': 'None',
                    'sigma_activation': 'None',
                    'mu_init': {'name': 'default'},
                    'sigma_init': {'name': 'const_initializer', 'val': 0.0},
                    'fixed_sigma': True
                }
            },
            'mlp': {
                'units': [512, 256, 128],
                'activation': 'elu',
                'initializer': {'name': 'default'}
            }
        },
        'config': {
            'name': args.experiment_name,
            'env_name': env_name,
            'multi_gpu': False,
            'ppo': True,
            'mixed_precision': False,
            'normalize_input': True,
            'normalize_value': True,
            'reward_shaper': {'scale_value': 1.0},
            'normalize_advantage': True,
            'gamma': 0.99,
            'tau': 0.95,
            'learning_rate': 1e-4,  # LOWER for stability
            'lr_schedule': 'adaptive',
            'kl_threshold': 0.016,
            'max_epochs': args.max_epochs,
            'save_best_after': 50,
            'save_frequency': 50,
            'grad_norm': 1.0,
            'entropy_coef': 0.001,  # MORE exploration
            'truncate_grads': True,
            'e_clip': 0.2,
            'horizon_length': 64,
            'minibatch_size': 512,
            'mini_epochs': 8,
            'critic_coef': 2.0,
            'clip_value': True,
            'seq_len': 4,
            'bounds_loss_coef': 0.0001,
        }
    }
}

# Create environment
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg
from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

env_cfg = PolishEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = PolishEnv(cfg=env_cfg, render_mode=None)

print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

# Wrap for RL-Games
wrapped = RlGamesVecEnvWrapper(env, rl_device="cuda:0", clip_obs=10.0, clip_actions=1.0)

# Register
vecenv.register(env_name, lambda config_name, num_actors, **kwargs: wrapped)
env_configurations.register(env_name, {"vecenv_type": env_name, "env_config": {}})

# Save config
with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
    yaml.dump(config, f)

print("🚀 Starting training...\n")

runner = Runner()
runner.load(config)
runner.reset()

try:
    runner.run({'train': True, 'play': False, 'checkpoint': output_dir, 'sigma': None})
except KeyboardInterrupt:
    print("\n⚠️  Interrupted")
finally:
    print(f"\n✅ Done: {output_dir}")
    simulation_app.close()
