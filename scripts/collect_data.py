#!/usr/bin/env python3
"""
Data Collection Script for Fixed Mode Polishing
Target: Maintain contact at -20 N
Collects: force, trajectory, orientation, impedance for analysis

Author: Albin Bajrami
"""
import _path_setup

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect polishing data with target force")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--num_steps", type=int, default=1000, help="Steps per run")
parser.add_argument("--num_runs", type=int, default=100, help="Number of runs")
parser.add_argument("--target_fz", type=float, default=-20.0, help="Target contact force [N]")
parser.add_argument("--output_dir", type=str, default="./data_collection", help="Output dir")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import h5py
import gymnasium as gym

import robo_pp_fixed
from robo_pp_fixed.cfg.scene_cfg import JustPushSceneCfg


def collect_run(env, run_idx, num_steps, target_fz, output_dir):
    """Collect data for a single run."""
    
    # Data buffers
    data = {
        "frame": [],
        "sim_time": [],
        "phase": [],
        # End-effector
        "ee_pos": [],
        "ee_quat": [],
        "ee_lin_vel": [],
        "ee_ang_vel": [],
        # Target waypoint index
        "wpt_idx": [],
        # Force
        "contact_force": [],
        "fz": [],
        "fz_error": [],  # fz - target_fz
        # Joints
        "joint_pos": [],
        "joint_vel": [],
        # Impedance
        "kp": [],
        "zeta": [],
        # Control
        "action": [],
        "tau": [],
        # Reward
        "reward": [],
    }
    
    obs, info = env.reset()
    unwrapped = env.unwrapped
    
    total_reward = 0.0
    contact_count = 0
    
    for step in range(num_steps):
        # Random action (or could use trained policy)
        action = torch.rand(1, env.action_space.shape[0], device="cuda:0") * 2 - 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward[0].item()
        
        # Collect data
        data["frame"].append(unwrapped.frame)
        data["sim_time"].append(unwrapped.frame * unwrapped.physics_dt)
        data["phase"].append(unwrapped.phase[0].item())
        
        # End-effector state
        ee_idx = unwrapped.ee_body_idx
        body_state = unwrapped.robot.data.body_state_w[0, ee_idx]
        data["ee_pos"].append(body_state[:3].cpu().numpy())
        data["ee_quat"].append(body_state[3:7].cpu().numpy())
        data["ee_lin_vel"].append(body_state[7:10].cpu().numpy())
        data["ee_ang_vel"].append(body_state[10:13].cpu().numpy())
        
        # Waypoint index
        data["wpt_idx"].append(unwrapped.wpt_idx[0].item())
        
        # Contact force
        if hasattr(unwrapped, 'cube_sensor'):
            force = unwrapped.cube_sensor.data.net_forces_w[0, 0].cpu().numpy()
            fz = force[2]
        else:
            force = np.zeros(3)
            fz = 0.0
        
        data["contact_force"].append(force)
        data["fz"].append(fz)
        data["fz_error"].append(fz - target_fz)
        
        if abs(fz) > 0.5:
            contact_count += 1
        
        # Joint state
        data["joint_pos"].append(unwrapped.robot.data.joint_pos[0].cpu().numpy())
        data["joint_vel"].append(unwrapped.robot.data.joint_vel[0].cpu().numpy())
        
        # Impedance
        data["kp"].append(unwrapped.dynamic_kp[0].cpu().numpy())
        data["zeta"].append(unwrapped.dynamic_zeta[0].cpu().numpy())
        
        # Action and torque
        data["action"].append(action[0].cpu().numpy())
        if hasattr(unwrapped, 'tau2'):
            data["tau"].append(unwrapped.tau2[0].cpu().numpy())
        else:
            data["tau"].append(np.zeros(7))
        
        data["reward"].append(reward[0].item())
        
        # Handle reset
        done = terminated | truncated
        if done.any():
            obs, info = env.reset()
    
    # Save to HDF5
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"run_{run_idx:04d}_{timestamp}.h5"
    
    with h5py.File(filename, "w") as f:
        # Metadata
        f.attrs["run_idx"] = run_idx
        f.attrs["num_steps"] = num_steps
        f.attrs["target_fz"] = target_fz
        f.attrs["total_reward"] = total_reward
        f.attrs["contact_ratio"] = contact_count / num_steps
        f.attrs["timestamp"] = timestamp
        
        # Data
        for key, values in data.items():
            arr = np.array(values)
            f.create_dataset(key, data=arr, compression="gzip")
    
    return {
        "total_reward": total_reward,
        "mean_fz": np.mean(data["fz"]),
        "min_fz": np.min(data["fz"]),
        "max_fz": np.max(data["fz"]),
        "contact_ratio": contact_count / num_steps,
    }


def main():
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("DATA COLLECTION - Fixed Mode Polishing")
    print(f"{'='*70}")
    print(f"  Target force: {args_cli.target_fz} N")
    print(f"  Runs: {args_cli.num_runs}")
    print(f"  Steps per run: {args_cli.num_steps}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")
    
    # Create environment
    cfg = robo_pp_fixed.PolishEnvCfg()
    cfg.debug_interval = 0
    cfg.scene = JustPushSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
        replicate_physics=True
    )
    
    env = gym.make("Polish-Fixed-v0", cfg=cfg, render_mode=None)
    print(f"✓ Environment created\n")
    
    # Statistics
    all_stats = []
    
    for run in range(args_cli.num_runs):
        stats = collect_run(env, run, args_cli.num_steps, args_cli.target_fz, output_dir)
        all_stats.append(stats)
        
        print(f"Run {run+1:3d}/{args_cli.num_runs}: "
              f"reward={stats['total_reward']:+8.2f}, "
              f"Fz={stats['mean_fz']:+6.2f}N (min={stats['min_fz']:+.1f}, max={stats['max_fz']:+.1f}), "
              f"contact={stats['contact_ratio']*100:.1f}%")
    
    # Summary
    print(f"\n{'='*70}")
    print("COLLECTION COMPLETED")
    print(f"{'='*70}")
    print(f"  Total runs: {args_cli.num_runs}")
    print(f"  Mean reward: {np.mean([s['total_reward'] for s in all_stats]):.2f}")
    print(f"  Mean Fz: {np.mean([s['mean_fz'] for s in all_stats]):.2f} N")
    print(f"  Mean contact ratio: {np.mean([s['contact_ratio'] for s in all_stats])*100:.1f}%")
    print(f"  Files saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    # Save summary
    summary_file = output_dir / "summary.npz"
    np.savez(summary_file, 
             rewards=[s['total_reward'] for s in all_stats],
             mean_fz=[s['mean_fz'] for s in all_stats],
             min_fz=[s['min_fz'] for s in all_stats],
             contact_ratio=[s['contact_ratio'] for s in all_stats])
    print(f"Summary saved to: {summary_file}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
