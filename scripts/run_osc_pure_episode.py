#!/usr/bin/env python3
"""
Run Pure OSC episode (zero actions = fixed impedance) and save data.
Uses the framework impedance values with agent actions = 0.

Usage:
    python scripts/run_osc_pure_episode.py --num_envs 1 --episodes 1 --headless
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime

# Isaac Lab imports MUST come first
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run Pure OSC episode")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
parser.add_argument("--output_dir", type=str, default="./data_osc_pure", help="Output directory")
parser.add_argument("--seed", type=int, default=0, help="Random seed for env rollouts")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import the rest
import torch
import numpy as np
import h5py
import json
import gymnasium as gym
import random


import robo_pp_fixed
from robo_pp_fixed import PolishEnvCfg


def collect_episode(env, episode_idx: int, output_dir: str, timestamp: str, seed: int):
    """Collect data for one episode with pure OSC (zero actions)."""
    
    # Data buffers
    data = {
        'frame': [],
        'sim_time': [],
        'ee_pos': [],
        'ee_quat': [],
        'joint_pos': [],
        'joint_vel': [],
        'fx': [],
        'fy': [],
        'fz': [],
        'fz_error': [],
        'kp': [],
        'zeta': [],
        'wpt_idx': [],
        'phase': [],
        'action': [],
        'reward': [],
    }
    
    obs, info = env.reset(seed=seed)
    done = False
    step = 0
    episode_reward = 0.0
    
    # Get unwrapped env
    unwrapped = env.unwrapped
    
    # Determine action dimension from environment
    action_dim = env.action_space.shape[0]
    print(f"  Action space: {env.action_space}")
    
    while not done:
        # Pure OSC: zero actions (use default impedance, no RL adjustment)
        action = torch.zeros((1, action_dim), device="cuda:0")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data
        data['frame'].append(step)
        data['sim_time'].append(step * 0.1)  # env_dt = sim_dt * decimation = 0.01 * 10 = 0.1s
        
        # EE pose from robot data
        if hasattr(unwrapped, 'robot') and hasattr(unwrapped, 'ee_body_idx'):
            ee_pos = unwrapped.robot.data.body_pos_w[0, unwrapped.ee_body_idx].cpu().numpy().copy()
            ee_quat = unwrapped.robot.data.body_quat_w[0, unwrapped.ee_body_idx].cpu().numpy().copy()
            data['ee_pos'].append(ee_pos)
            data['ee_quat'].append(ee_quat)
            
            joint_ids = unwrapped.joint_ids
            data['joint_pos'].append(unwrapped.robot.data.joint_pos[0, joint_ids].cpu().numpy().copy())
            data['joint_vel'].append(unwrapped.robot.data.joint_vel[0, joint_ids].cpu().numpy().copy())
        
        # Contact force - use cube_sensor (or carpet_sensor alias)
        if hasattr(unwrapped, 'cube_sensor'):
            fx_val = unwrapped.cube_sensor.data.net_forces_w[0, 0, 0].item()
            fy_val = unwrapped.cube_sensor.data.net_forces_w[0, 0, 1].item()
            fz_val = unwrapped.cube_sensor.data.net_forces_w[0, 0, 2].item()
            data['fx'].append(fx_val)
            data['fy'].append(fy_val)
            data['fz'].append(fz_val)
            data['fz_error'].append(fz_val - unwrapped.fz_target)
        
        # Stiffness - use dynamic_kp
        if hasattr(unwrapped, 'dynamic_kp'):
            data['kp'].append(unwrapped.dynamic_kp[0].cpu().numpy().copy())
        
        # Damping - use dynamic_zeta
        if hasattr(unwrapped, 'dynamic_zeta'):
            data['zeta'].append(unwrapped.dynamic_zeta[0].cpu().numpy().copy())
        
        # Waypoint index
        if hasattr(unwrapped, 'wpt_idx'):
            data['wpt_idx'].append(unwrapped.wpt_idx[0].item())
        
        # Phase (env internal phase: 0=descent, 1=contact, 2=rise)
        if hasattr(unwrapped, 'phase'):
            data['phase'].append(unwrapped.phase[0].item())
        
        data['action'].append(action[0].cpu().numpy().copy())
        reward_val = reward[0].item() if hasattr(reward[0], 'item') else float(reward[0])
        data['reward'].append(reward_val)
        episode_reward += reward_val
        
        done = terminated.any().item() if hasattr(terminated, 'any') else terminated
        done = done or (truncated.any().item() if hasattr(truncated, 'any') else truncated)
        step += 1
        
        # Safety limit
        if step > 500:
            print(f"  Warning: Episode exceeded 500 steps, terminating")
            break
    
    # Save to H5
    os.makedirs(output_dir, exist_ok=True)
    filename = f"osc_pure_{timestamp}_ep{episode_idx:02d}.h5"
    filepath = os.path.join(output_dir, filename)
    
    with h5py.File(filepath, 'w') as f:
        for key, values in data.items():
            if values:
                arr = np.array(values)
                f.create_dataset(key, data=arr, compression='gzip')
        
        # Metadata
        f.attrs['episode_idx'] = episode_idx
        f.attrs['total_steps'] = step
        f.attrs['episode_reward'] = episode_reward
        f.attrs['timestamp'] = timestamp
        f.attrs['controller'] = 'OSC_pure'
        f.attrs['action_type'] = 'zero_actions'
        f.attrs['env_dt'] = 0.1
    
    # Also save JSON summary
    json_file = filepath.replace('.h5', '.json')
    summary = {
        'episode_idx': episode_idx,
        'total_steps': step,
        'episode_reward': float(episode_reward),
        'timestamp': timestamp,
        'seed': seed,
        'controller': 'OSC_pure',
        'env_dt': 0.1,
        'mean_fz': float(np.mean([f for f in data['fz'] if f != 0])) if data['fz'] else 0,
        'max_fz': float(np.min(data['fz'])) if data['fz'] else 0,  # max negative = strongest push
        'mean_fx': float(np.mean(data['fx'])) if data['fx'] else 0,
        'mean_fy': float(np.mean(data['fy'])) if data['fy'] else 0,
    }
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Episode {episode_idx}: {step} steps, reward={episode_reward:.2f}")
    print(f"    Mean Fz (contact): {summary['mean_fz']:.2f} N")
    print(f"    Peak Fz: {summary['max_fz']:.2f} N")
    print(f"    Saved to: {filename}")
    
    return step, episode_reward


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print(f"\n{'='*60}")
    print(f"  Pure OSC Episode Collection")
    print(f"  Timestamp: {timestamp}")
    print(f"{'='*60}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Mode:       Zero RL actions (pure OSC impedance)")
    print(f"{'='*60}\n")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    env = gym.make("Polish-Fixed-v0", cfg=env_cfg, render_mode=None)
    
    total_steps = 0
    total_reward = 0.0
    
    for ep in range(args.episodes):
        steps, reward = collect_episode(env, ep, args.output_dir, timestamp, args.seed + ep)
        total_steps += steps
        total_reward += reward
    
    env.close()
    simulation_app.close()
    
    print(f"\n{'='*60}")
    print(f"  Collection Complete!")
    print(f"  Total episodes: {args.episodes}")
    print(f"  Total steps:    {total_steps}")
    print(f"  Total reward:   {total_reward:.2f}")
    print(f"  Avg reward/ep:  {total_reward/args.episodes:.2f}")
    print(f"  Files saved to: {args.output_dir}/osc_pure_{timestamp}_*.h5")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
