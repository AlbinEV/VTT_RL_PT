#!/usr/bin/env python3
"""
Test Run Script for Fixed Mode Polishing Environment
Runs in headless mode and saves episode data to HDF5

Author: Albin Bajrami
Institution: VTT Technical Research Centre of Finland and Università Politecnica delle Marche
"""
import _path_setup

import argparse
import os
import sys
from pathlib import Path

# Isaac Lab imports - must be before other imports
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab initialization
parser = argparse.ArgumentParser(description="Test Fixed Mode Environment with HDF5 logging")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
parser.add_argument("--debug_interval", type=int, default=100, help="Debug print interval (0=disabled)")
parser.add_argument("--output_dir", type=str, default="./test_output", help="Output directory for HDF5 files")

# Add headless by default for testing
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True  # Force headless mode for test

# Launch Isaac Lab app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import the rest after simulation is initialized
import torch
import numpy as np
import gymnasium as gym
from datetime import datetime

# Import and register the environment
import robo_pp_fixed
from robo_pp_fixed.cfg.scene_cfg import JustPushSceneCfg
from robo_pp_fixed.utils import EpisodeLogger


def main():
    """Main test function."""
    
    print(f"\n{'='*70}")
    print(f"TEST RUN - Fixed Mode Polishing Environment")
    print(f"{'='*70}")
    print(f"  Environments: {args_cli.num_envs}")
    print(f"  Steps: {args_cli.num_steps}")
    print(f"  Debug interval: {args_cli.debug_interval}")
    print(f"  Output dir: {args_cli.output_dir}")
    print(f"  Mode: HEADLESS")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom config with specified num_envs
    cfg = robo_pp_fixed.PolishEnvCfg()
    cfg.debug_interval = args_cli.debug_interval
    cfg.scene = JustPushSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
        replicate_physics=True
    )
    
    # Create environment with custom config
    env = gym.make(
        "Polish-Fixed-v0",
        cfg=cfg,
        render_mode=None
    )
    
    print(f"✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Get path to log_vars.yaml from installed package
    pkg_root = Path(robo_pp_fixed.__file__).resolve().parent
    yaml_cfg = pkg_root / "data" / "log_vars.yaml"
    
    # Create episode logger
    logger = EpisodeLogger(
        run_dir=output_dir,
        yaml_cfg=str(yaml_cfg),
        chunk_size=100,
        start_idx=0,
        attrs={
            "experiment": "test_run",
            "date": datetime.now().isoformat(),
            "num_envs": args_cli.num_envs,
        }
    )
    print(f"✓ Episode logger initialized")
    
    # Reset environment
    obs, info = env.reset()
    print(f"✓ Environment reset")
    print(f"  Observation shape: {obs['policy'].shape if isinstance(obs, dict) else obs.shape}")
    
    # Statistics tracking
    total_reward = torch.zeros(args_cli.num_envs, device="cuda:0")
    episode_rewards = []
    
    # Run test loop
    print(f"\n--- Starting test loop for {args_cli.num_steps} steps ---\n")
    
    for step in range(args_cli.num_steps):
        # Random action
        action = env.action_space.sample()
        action = torch.tensor(action, device="cuda:0").unsqueeze(0).repeat(args_cli.num_envs, 1)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Get internal state for logging (env 0 only)
        unwrapped = env.unwrapped
        log_data = {
            "frame": unwrapped.frame,
            "phase": unwrapped.phase[0].item() if hasattr(unwrapped, "phase") else 0,
            "sim_time": unwrapped.frame * unwrapped.physics_dt if hasattr(unwrapped, "physics_dt") else step * 0.01,
            "joint_pos": unwrapped.robot.data.joint_pos[0].cpu().numpy() if hasattr(unwrapped, "robot") else np.zeros(7),
            "joint_vel": unwrapped.robot.data.joint_vel[0].cpu().numpy() if hasattr(unwrapped, "robot") else np.zeros(7),
            "kp": unwrapped.dynamic_kp[0].cpu().numpy() if hasattr(unwrapped, "dynamic_kp") else np.zeros(6),
            "zeta": unwrapped.dynamic_zeta[0].cpu().numpy() if hasattr(unwrapped, "dynamic_zeta") else np.zeros(6),
            "action": action[0].cpu().numpy(),
            "reward": reward[0].item(),
        }
        
        # Add ee_pos if available
        if hasattr(unwrapped, "robot") and hasattr(unwrapped.robot.data, "body_state_w"):
            try:
                ee_idx = unwrapped.ee_body_idx if hasattr(unwrapped, "ee_body_idx") else -1
                if ee_idx >= 0:
                    log_data["ee_pos"] = unwrapped.robot.data.body_state_w[0, ee_idx, :3].cpu().numpy()
                    log_data["ee_quat"] = unwrapped.robot.data.body_state_w[0, ee_idx, 3:7].cpu().numpy()
            except Exception:
                pass
        
        # Log data
        logger.log(log_data)
        
        # Print progress
        if step % 100 == 0:
            print(f"  Step {step:4d}/{args_cli.num_steps}: reward={reward[0].item():+.4f}, total={total_reward[0].item():+.2f}")
        
        # Handle episode termination
        done = terminated | truncated
        if done.any():
            for i in range(args_cli.num_envs):
                if done[i]:
                    episode_rewards.append(total_reward[i].item())
                    print(f"\n  Episode completed at step {step}: reward={total_reward[i].item():.2f}")
                    total_reward[i] = 0
                    logger.next_episode()
    
    # Close logger
    logger.close()
    
    # Print summary
    print(f"\n{'='*70}")
    print("TEST COMPLETED")
    print(f"{'='*70}")
    print(f"  Total steps: {args_cli.num_steps}")
    print(f"  Episodes completed: {len(episode_rewards)}")
    if episode_rewards:
        print(f"  Mean episode reward: {np.mean(episode_rewards):.2f}")
    print(f"  Final accumulated reward (env 0): {total_reward[0].item():.2f}")
    print(f"  Output saved to: {output_dir.absolute()}")
    
    # List output files
    h5_files = list(output_dir.glob("*.h5"))
    print(f"  HDF5 files created: {len(h5_files)}")
    for f in h5_files:
        print(f"    - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    print(f"{'='*70}\n")
    
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
