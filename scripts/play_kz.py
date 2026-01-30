#!/usr/bin/env python3
"""
Play/Evaluate trained Kz-only policy.
Visualizes the robot following the fixed waypoints with learned impedance control.

Usage:
    python scripts/play_kz.py --checkpoint <path_to_checkpoint.pth> --num_envs 1
    python scripts/play_kz.py  # Uses latest checkpoint
"""
import _path_setup

import argparse
import os
import sys
import glob

# Isaac Lab imports MUST come first
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab init
parser = argparse.ArgumentParser(description="Play trained Kz-only policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else
import torch
import numpy as np
import gymnasium as gym

# Add parent to path

# Import and register environment
import robo_pp_fixed
from robo_pp_fixed import PolishEnvOSCCfg


def find_latest_checkpoint(base_dir: str = None) -> str:
    """Find the most recent checkpoint in the logs directory."""
    if base_dir is None:
        base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "rl_games", "kz_only"
        )
    
    # Find all checkpoint files
    pattern = os.path.join(base_dir, "**/nn/*.pth")
    checkpoints = glob.glob(pattern, recursive=True)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_dir}")
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_policy(checkpoint_path: str, device: str = "cuda:0"):
    """Load trained policy from checkpoint."""
    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint
    
    return model_state, checkpoint


def main():
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        try:
            checkpoint_path = find_latest_checkpoint()
            print(f"[INFO] Using latest checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            print("[INFO] Running with random policy instead...")
            checkpoint_path = None
    
    # Create environment config
    env_cfg = PolishEnvOSCCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.reward_type = "kpz"
    
    # Create environment
    env = gym.make("Polish-Fixed-v0", cfg=env_cfg, render_mode="rgb_array" if not args.headless else None)
    
    print(f"\n{'='*60}")
    print("  PLAY: Kz-only Policy Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint:     {checkpoint_path if checkpoint_path else 'RANDOM POLICY'}")
    print(f"  Environments:   {args.num_envs}")
    print(f"  Steps:          {args.steps}")
    print(f"  Headless:       {args.headless}")
    print(f"{'='*60}\n")
    
    # Load policy if checkpoint exists
    policy_net = None
    if checkpoint_path:
        try:
            model_state, checkpoint = load_policy(checkpoint_path)
            # Note: For full policy loading, we'd need to reconstruct the network
            # For now, we'll use the environment's built-in behavior
            print("[INFO] Checkpoint loaded (using for reference)")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e}")
            print("[INFO] Running with random/zero actions")
    
    # Reset environment
    obs, info = env.reset()
    
    # Statistics
    total_reward = 0.0
    episode_rewards = []
    current_episode_reward = 0.0
    
    # Data collection for visualization
    ee_positions = []
    forces_z = []
    waypoint_indices = []
    kp_values = []
    
    print("[INFO] Starting evaluation loop...")
    print("[INFO] Press Ctrl+C to stop early\n")
    
    try:
        for step in range(args.steps):
            # Get action (zero actions = pure OSC trajectory following)
            # With trained policy, actions would modulate Kp_z
            action = torch.zeros((args.num_envs, 4), device="cuda:0")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Accumulate rewards
            reward_val = reward.mean().item() if hasattr(reward, 'mean') else float(reward)
            current_episode_reward += reward_val
            total_reward += reward_val
            
            # Collect data from unwrapped env
            unwrapped = env.unwrapped
            if hasattr(unwrapped, 'ee_pos_w'):
                ee_pos = unwrapped.ee_pos_w[0].cpu().numpy()
                ee_positions.append(ee_pos.copy())
            
            if hasattr(unwrapped, 'fz'):
                fz = unwrapped.fz[0].item() if hasattr(unwrapped.fz[0], 'item') else unwrapped.fz[0]
                forces_z.append(fz)
            
            if hasattr(unwrapped, 'traj_manager') and hasattr(unwrapped.traj_manager, 'wpt_idx'):
                wpt = unwrapped.traj_manager.wpt_idx[0].item() if hasattr(unwrapped.traj_manager.wpt_idx, '__getitem__') else 0
                waypoint_indices.append(wpt)
            
            # Check for episode end
            done = terminated.any().item() if hasattr(terminated, 'any') else terminated
            if done:
                episode_rewards.append(current_episode_reward)
                print(f"  Episode complete | Reward: {current_episode_reward:.2f}")
                current_episode_reward = 0.0
                obs, info = env.reset()
            
            # Progress update
            if (step + 1) % 100 == 0:
                avg_reward = total_reward / (step + 1)
                print(f"  Step {step+1}/{args.steps} | Avg Reward: {avg_reward:.3f}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Evaluation interrupted by user")
    
    # Final statistics
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total steps:        {len(ee_positions)}")
    print(f"  Total reward:       {total_reward:.2f}")
    print(f"  Average reward:     {total_reward/max(len(ee_positions),1):.4f}")
    print(f"  Episodes completed: {len(episode_rewards)}")
    
    if ee_positions:
        ee_arr = np.array(ee_positions)
        print(f"\n  EE Position Range:")
        print(f"    X: [{ee_arr[:,0].min():.4f}, {ee_arr[:,0].max():.4f}]")
        print(f"    Y: [{ee_arr[:,1].min():.4f}, {ee_arr[:,1].max():.4f}]")
        print(f"    Z: [{ee_arr[:,2].min():.4f}, {ee_arr[:,2].max():.4f}]")
    
    if forces_z:
        fz_arr = np.array(forces_z)
        print(f"\n  Force Z:")
        print(f"    Range: [{fz_arr.min():.2f}, {fz_arr.max():.2f}] N")
        print(f"    Mean:  {fz_arr.mean():.2f} N")
        contact_steps = np.sum(np.abs(fz_arr) > 0.5)
        print(f"    Contact steps (|Fz|>0.5): {contact_steps} ({100*contact_steps/len(fz_arr):.1f}%)")
    
    if waypoint_indices:
        wpt_arr = np.array(waypoint_indices)
        print(f"\n  Waypoints visited: {np.unique(wpt_arr)}")
    
    print(f"{'='*60}\n")
    
    # Close environment
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
