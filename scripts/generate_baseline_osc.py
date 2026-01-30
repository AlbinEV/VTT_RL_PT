#!/usr/bin/env python3
"""
Generate PURE OSC baseline trajectories with fixed optimal parameters.
NO RL, just standard OSC with Kp_z and damping from grid search results.

Usage:
    ./isaaclab.sh -p scripts/generate_baseline_osc.py --kp_z 5192 --episodes 3
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime
import h5py
import numpy as np


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate Pure OSC Baseline")
parser.add_argument("--kp_z", type=float, default=5192.0, help="Fixed Kp_z value (N/m)")
parser.add_argument("--damping", type=float, default=1.0, help="Damping ratio")
parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
parser.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
parser.add_argument("--output_dir", type=str, 
                    default=str(_path_setup.DATA_ROOT / "baseline_osc_pure"),
                    help="Output directory")
parser.add_argument("--headless", action="store_true", default=True)

args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"osc_baseline_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("  PURE OSC BASELINE GENERATION")
    print("=" * 80)
    print(f"  Kp_z (fixed): {args.kp_z:.1f} N/m")
    print(f"  Damping ratio: {args.damping}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Output: {output_dir}")
    print("=" * 80 + "\n")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = 1
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    # Reset environment to get initial obs
    obs_dict, _ = env.reset()
    
    # Force fixed Kp_z in the OSC controller
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    else:
        base_env = env
        
    # Set fixed impedance parameters
    if hasattr(base_env, 'osc_kp'):
        base_env.osc_kp[:, 2] = args.kp_z  # Fixed Kp_z
        print(f"✓ Set fixed Kp_z = {args.kp_z:.1f} N/m")
    
    if hasattr(base_env, 'osc_zeta'):
        base_env.osc_zeta[:, 2] = args.damping
        print(f"✓ Set fixed damping = {args.damping}")
    
    all_results = []
    
    for episode in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"  EPISODE {episode + 1}/{args.episodes}")
        print(f"{'='*60}")
        
        obs_dict, _ = env.reset()
        
        # Storage
        episode_data = {
            'timestamps': [],
            'contact_forces': [],
            'kp_z_values': [],
            'damping_values': [],
            'tcp_positions': [],
            'actions': [],
            'rewards': [],
        }
        
        for step in range(args.max_steps):
            # PURE OSC: zero actions (no RL modification)
            action = torch.zeros((1, 1), device=env.device, dtype=torch.float32)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            
            # Collect data
            episode_data['timestamps'].append(step * env.cfg.sim.dt)
            episode_data['actions'].append(action.cpu().numpy()[0, 0])
            episode_data['rewards'].append(reward.cpu().numpy()[0])
            
            # Extract contact force
            if hasattr(base_env, 'net_contact_force'):
                fz = base_env.net_contact_force[0, 2].cpu().item()
            else:
                fz = 0.0
            episode_data['contact_forces'].append(fz)
            
            # Extract current Kp_z
            if hasattr(base_env, 'osc_kp'):
                kp_z = base_env.osc_kp[0, 2].cpu().item()
            else:
                kp_z = args.kp_z
            episode_data['kp_z_values'].append(kp_z)
            
            # Extract damping
            if hasattr(base_env, 'osc_zeta'):
                damping = base_env.osc_zeta[0, 2].cpu().item()
            else:
                damping = args.damping
            episode_data['damping_values'].append(damping)
            
            # TCP position
            if hasattr(base_env, 'ee_pos'):
                tcp_pos = base_env.ee_pos[0, :3].cpu().numpy()
            else:
                tcp_pos = np.zeros(3)
            episode_data['tcp_positions'].append(tcp_pos)
            
            if step % 50 == 0:
                print(f"  Step {step:3d}: Fz={fz:7.2f}N, Kp_z={kp_z:.1f}, Reward={reward.item():.2f}")
            
            if terminated or truncated:
                print(f"  → Episode terminated at step {step}")
                break
        
        # Convert to numpy arrays
        for key in episode_data:
            episode_data[key] = np.array(episode_data[key])
        
        # Statistics
        mean_fz = np.mean(episode_data['contact_forces'])
        mae_fz = np.mean(np.abs(episode_data['contact_forces'] + 20.0))
        mean_kp = np.mean(episode_data['kp_z_values'])
        std_kp = np.std(episode_data['kp_z_values'])
        duration = episode_data['timestamps'][-1]
        
        print(f"\n  STATISTICS:")
        print(f"    Duration: {duration:.2f}s")
        print(f"    Mean Force: {mean_fz:.2f} N")
        print(f"    MAE from -20N: {mae_fz:.2f} N")
        print(f"    Kp_z: {mean_kp:.1f} ± {std_kp:.1f} N/m")
        print(f"    Actions: all zero (OSC pure)")
        
        # Save episode
        episode_file = os.path.join(output_dir, f"episode_{episode}.npz")
        np.savez(
            episode_file,
            **episode_data
        )
        print(f"  ✓ Saved: {episode_file}")
        
        # Also save as H5 for compatibility
        h5_file = os.path.join(output_dir, f"episode_{episode}.h5")
        with h5py.File(h5_file, 'w') as f:
            for key, value in episode_data.items():
                f.create_dataset(key, data=value)
        print(f"  ✓ Saved: {h5_file}")
        
        all_results.append({
            'episode': episode,
            'duration': duration,
            'mean_fz': mean_fz,
            'mae_fz': mae_fz,
            'mean_kp_z': mean_kp,
            'std_kp_z': std_kp,
        })
    
    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total episodes: {len(all_results)}")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Average across episodes:")
    print(f"    Mean Force: {np.mean([r['mean_fz'] for r in all_results]):.2f} N")
    print(f"    MAE: {np.mean([r['mae_fz'] for r in all_results]):.2f} N")
    print(f"    Kp_z: {np.mean([r['mean_kp_z'] for r in all_results]):.1f} N/m (fixed)")
    print(f"{'='*80}\n")
    
    # Save summary
    summary_file = os.path.join(output_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("PURE OSC BASELINE SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Kp_z (fixed): {args.kp_z:.1f} N/m\n")
        f.write(f"  Damping: {args.damping}\n")
        f.write(f"  Episodes: {args.episodes}\n\n")
        f.write(f"Results:\n")
        for r in all_results:
            f.write(f"  Episode {r['episode']}: MAE={r['mae_fz']:.2f}N, Duration={r['duration']:.2f}s\n")
        f.write(f"\nAverage MAE: {np.mean([r['mae_fz'] for r in all_results]):.2f} N\n")
    
    print(f"✓ Summary saved: {summary_file}\n")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
