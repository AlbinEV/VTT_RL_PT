#!/usr/bin/env python3
"""
Minimal Play/Evaluation Script for Fixed Mode Polishing Environment
Runs a trained policy and displays debug information

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
parser = argparse.ArgumentParser(description="Play Fixed Mode Polishing Agent")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--debug", action="store_true", help="Enable debug prints (every 50 steps)")
parser.add_argument("--debug_interval", type=int, default=50, help="Debug print interval (when --debug)")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest after simulation is initialized
import torch
import gymnasium as gym
import numpy as np

# Import and register the environment
import robo_pp_fixed

def main():
    """Main play function."""
    
    # Determine debug interval
    debug_interval = args_cli.debug_interval if args_cli.debug else 0
    
    print(f"\n{'='*60}")
    print(f"Playing Fixed Mode Polishing Environment")
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Environments: {args_cli.num_envs}")
    print(f"Episodes: {args_cli.num_episodes}")
    print(f"Debug mode: {args_cli.debug} (interval={debug_interval})")
    print(f"{'='*60}\n")
    
    # Create environment with debug settings
    env = gym.make(
        "Polish-Fixed-v0",
        cfg=robo_pp_fixed.PolishEnvCfg(
            debug_interval=debug_interval,  # Enable debug prints if --debug
            scene=robo_pp_fixed.PolishEnvCfg.scene.__class__(
                num_envs=args_cli.num_envs
            )
        ),
        render_mode=None
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args_cli.checkpoint}")
    checkpoint = torch.load(args_cli.checkpoint, map_location="cuda:0")
    
    # Extract policy weights (adapt based on checkpoint format)
    if "model" in checkpoint:
        policy_weights = checkpoint["model"]
    else:
        policy_weights = checkpoint
    
    print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")
    
    # Statistics tracking
    episode_rewards = []
    episode_lengths = []
    completed_episodes = 0
    
    # Reset environment
    obs, info = env.reset()
    episode_reward = torch.zeros(args_cli.num_envs, device="cuda:0")
    episode_length = torch.zeros(args_cli.num_envs, device="cuda:0")
    
    step = 0
    while completed_episodes < args_cli.num_episodes:
        step += 1
        
        # Random action for now (replace with policy inference)
        action = env.action_space.sample()
        action = torch.tensor(action, device="cuda:0").unsqueeze(0).repeat(args_cli.num_envs, 1)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_length += 1
        
        # Handle episode termination
        done = terminated | truncated
        for i in range(args_cli.num_envs):
            if done[i]:
                completed_episodes += 1
                episode_rewards.append(episode_reward[i].item())
                episode_lengths.append(episode_length[i].item())
                
                print(f"\nEpisode {completed_episodes} completed:")
                print(f"  Reward: {episode_reward[i].item():.2f}")
                print(f"  Length: {episode_length[i].item():.0f}")
                
                episode_reward[i] = 0
                episode_length[i] = 0
                
                if completed_episodes >= args_cli.num_episodes:
                    break
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    print(f"Min/Max reward: {np.min(episode_rewards):.2f} / {np.max(episode_rewards):.2f}")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
