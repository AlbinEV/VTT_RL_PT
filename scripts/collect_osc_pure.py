#!/usr/bin/env python3
"""
Collect data with Pure OSC (zero actions = fixed impedance).
Uses the original framework impedance values:
  - Kp = [3500, 1900, 3500, 460, 460, 410] N/m or Nm/rad
  - ζ  = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] (critically damped)

Usage:
    python scripts/collect_osc_pure.py --num_envs 1 --episodes 3 --headless
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime

# Isaac Lab imports MUST come first
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect data with Pure OSC control")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to collect")
parser.add_argument("--output_dir", type=str, default="./data_osc_pure", help="Output directory")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import the rest
import torch
import numpy as np
import h5py
import gymnasium as gym


import robo_pp_fixed
from robo_pp_fixed import PolishEnvCfg


def collect_episode(env, episode_idx: int, output_dir: str):
    """Collect data for one episode with pure OSC (zero actions)."""
    
    # Data buffers
    data = {
        'frame': [],
        'sim_time': [],
        'ee_pos': [],
        'ee_quat': [],
        'joint_pos': [],
        'joint_vel': [],
        'tau': [],
        'fz': [],
        'fz_error': [],
        'kp': [],
        'zeta': [],
        'wpt_idx': [],
        'phase': [],
        'action': [],
        'reward': [],
    }
    
    obs, info = env.reset()
    done = False
    step = 0
    episode_reward = 0.0
    
    # Get unwrapped env
    unwrapped = env.unwrapped
    
    while not done:
        # Pure OSC: zero actions (use default impedance)
        action = torch.zeros((1, 4), device="cuda:0")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Collect data - using correct attribute names
        data['frame'].append(step)
        data['sim_time'].append(step * 0.1)  # dt = 0.1
        
        # EE pose from robot data
        if hasattr(unwrapped, 'robot'):
            ee_pos = unwrapped.robot.data.body_pos_w[0, -1].cpu().numpy().copy()
            ee_quat = unwrapped.robot.data.body_quat_w[0, -1].cpu().numpy().copy()
            data['ee_pos'].append(ee_pos)
            data['ee_quat'].append(ee_quat)
            data['joint_pos'].append(unwrapped.robot.data.joint_pos[0].cpu().numpy().copy())
            data['joint_vel'].append(unwrapped.robot.data.joint_vel[0].cpu().numpy().copy())
        
        # Torques
        if hasattr(unwrapped, '_osc_torques'):
            data['tau'].append(unwrapped._osc_torques[0].cpu().numpy().copy())
        
        # Contact force - use _fz_ema
        if hasattr(unwrapped, '_fz_ema'):
            fz_val = unwrapped._fz_ema[0].item()
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
        
        # Phase
        if hasattr(unwrapped, 'phase_ctrl'):
            data['phase'].append(unwrapped.phase_ctrl.phase[0].item())
        
        data['action'].append(action[0].cpu().numpy().copy())
        reward_val = reward[0].item() if hasattr(reward[0], 'item') else float(reward[0])
        data['reward'].append(reward_val)
        episode_reward += reward_val
        
        done = terminated.any().item() if hasattr(terminated, 'any') else terminated
        step += 1
    
    # Save to H5
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"episode_{episode_idx:04d}_{timestamp}.h5"
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
        f.attrs['kp_nominal'] = [3500, 1900, 3500, 460, 460, 410]
        f.attrs['zeta_nominal'] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    
    print(f"  Episode {episode_idx}: {step} steps, reward={episode_reward:.2f}, saved to {filename}")
    
    return step, episode_reward


def main():
    # Original framework impedance values
    KP_NOMINAL = [3500, 1900, 3500, 460, 460, 410]  # N/m or Nm/rad
    ZETA_NOMINAL = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # Critically damped
    
    print(f"\n{'='*60}")
    print(f"  Pure OSC Data Collection")
    print(f"{'='*60}")
    print(f"  Kp nominal: {KP_NOMINAL}")
    print(f"  ζ nominal:  {ZETA_NOMINAL}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Output:     {args.output_dir}")
    print(f"{'='*60}\n")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    env = gym.make("Polish-Fixed-v0", cfg=env_cfg, render_mode=None)
    
    total_steps = 0
    total_reward = 0.0
    
    for ep in range(args.episodes):
        steps, reward = collect_episode(env, ep, args.output_dir)
        total_steps += steps
        total_reward += reward
    
    env.close()
    
    print(f"\n{'='*60}")
    print(f"  Collection Complete!")
    print(f"  Total episodes: {args.episodes}")
    print(f"  Total steps:    {total_steps}")
    print(f"  Total reward:   {total_reward:.2f}")
    print(f"  Avg reward/ep:  {total_reward/args.episodes:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
