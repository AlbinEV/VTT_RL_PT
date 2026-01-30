#!/usr/bin/env python3
"""
Evaluation Script for OSC-only Baseline (No RL Control)
Collects performance metrics for pure OSC controller without learned impedance.

This is the baseline experiment from the thesis for comparison:
- Fixed impedance parameters (no adaptation)
- Pure trajectory following with OSC

Author: Albin Bajrami
Institution: VTT Technical Research Centre of Finland and Università Politecnica delle Marche
"""
import _path_setup

import argparse
import os
import sys
import csv
from datetime import datetime

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab initialization
parser = argparse.ArgumentParser(description="Evaluate OSC-only Baseline")
parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest after simulation is initialized
import torch
import numpy as np
import gymnasium as gym

# Import and register the environment
import robo_pp_fixed
from robo_pp_fixed import PolishEnvCfg

def main():
    """Main evaluation function for OSC-only baseline."""
    
    # Get package directory for output
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(pkg_dir, "logs", "osc_baseline")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment with default (no RL) configuration
    cfg = PolishEnvCfg(
        scene=PolishEnvCfg.scene.__class__(num_envs=args_cli.num_envs)
    )
    env = gym.make("Polish-Fixed-v0", cfg=cfg, render_mode=None)
    
    print(f"\n{'='*70}")
    print(f"  EVALUATION: OSC-only Baseline (No RL)")
    print(f"{'='*70}")
    print(f"  Environments:       {args_cli.num_envs}")
    print(f"  Episodes:           {args_cli.num_episodes}")
    print(f"  Output directory:   {output_dir}")
    print(f"{'='*70}\n")
    
    # Metrics storage
    all_rewards = []
    all_force_errors = []
    all_position_errors = []
    all_episode_lengths = []
    all_contact_times = []
    
    episodes_completed = 0
    obs, info = env.reset()
    
    while episodes_completed < args_cli.num_episodes:
        # Zero action (no RL control, just OSC)
        action = torch.zeros(args_cli.num_envs, env.action_space.shape[0], device=env.unwrapped.device)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Track metrics
        all_rewards.append(reward.mean().item())
        
        # Check for episode completions
        done = terminated | truncated
        if done.any():
            done_indices = torch.where(done)[0]
            episodes_completed += len(done_indices)
            print(f"Episodes completed: {episodes_completed}/{args_cli.num_episodes}", end='\r')
    
    # Compute statistics
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"\n\n{'='*70}")
    print(f"  RESULTS: OSC-only Baseline")
    print(f"{'='*70}")
    print(f"  Mean reward:        {mean_reward:.4f} ± {std_reward:.4f}")
    print(f"  Episodes:           {episodes_completed}")
    print(f"{'='*70}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"osc_baseline_{timestamp}.csv")
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['mean_reward', mean_reward])
        writer.writerow(['std_reward', std_reward])
        writer.writerow(['episodes', episodes_completed])
    
    print(f"Results saved to: {results_file}")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
