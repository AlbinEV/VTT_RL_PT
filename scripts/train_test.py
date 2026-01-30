#!/usr/bin/env python3
"""
Quick Training Test for Fixed Mode Polishing Environment
Tests that training loop works with contact forces

Author: Albin Bajrami
"""
import _path_setup

import argparse
import os
import sys
from pathlib import Path

# Isaac Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Fixed Mode Agent")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=50, help="Max training iterations")
parser.add_argument("--debug_interval", type=int, default=0, help="Debug print interval")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from datetime import datetime

# Add package to path
script_dir = Path(__file__).parent
pkg_dir = script_dir.parent

import robo_pp_fixed
from robo_pp_fixed.cfg.scene_cfg import JustPushSceneCfg

def main():
    print(f"\n{'='*70}")
    print("TRAINING TEST - Fixed Mode Polishing")
    print(f"{'='*70}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Max iterations: {args_cli.max_iterations}")
    print(f"{'='*70}\n")

    # Create environment
    cfg = robo_pp_fixed.PolishEnvCfg()
    cfg.debug_interval = args_cli.debug_interval
    cfg.scene = JustPushSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
        replicate_physics=True
    )

    env = gym.make("Polish-Fixed-v0", cfg=cfg, render_mode=None)
    
    print(f"✓ Environment created")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")

    # Simple training loop (without RL-Games for quick test)
    obs, info = env.reset()
    
    total_rewards = []
    contact_forces = []
    
    for iteration in range(args_cli.max_iterations):
        # Random policy (just testing the loop)
        actions = torch.rand(args_cli.num_envs, env.action_space.shape[0], device="cuda:0") * 2 - 1
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        # Get contact force from env
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'cube_sensor'):
            fz = unwrapped.cube_sensor.data.net_forces_w[:, 0, 2]  # Z-force
            mean_fz = fz.mean().item()
            contact_forces.append(mean_fz)
        
        mean_reward = rewards.mean().item()
        total_rewards.append(mean_reward)
        
        if iteration % 10 == 0:
            fz_str = f", Fz={mean_fz:+.2f}N" if contact_forces else ""
            print(f"  Iter {iteration:3d}: reward={mean_reward:+.4f}{fz_str}")
        
        # Handle resets
        done = terminated | truncated
        if done.any():
            # Environment auto-resets
            pass
    
    # Summary
    print(f"\n{'='*70}")
    print("TRAINING TEST COMPLETED")
    print(f"{'='*70}")
    print(f"  Iterations: {args_cli.max_iterations}")
    print(f"  Mean reward: {sum(total_rewards)/len(total_rewards):.4f}")
    if contact_forces:
        print(f"  Mean contact Fz: {sum(contact_forces)/len(contact_forces):.2f} N")
        print(f"  Max contact Fz: {min(contact_forces):.2f} N (negative = pushing down)")
    print(f"{'='*70}\n")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
