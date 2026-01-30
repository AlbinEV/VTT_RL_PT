#!/usr/bin/env python3
"""
OSC Parameter Optimizer with comprehensive data logging
200 trials, random search for Kp_z and zeta_z
"""
import _path_setup

import argparse
import os
import sys
from datetime import datetime
import json
import time
import csv

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OSC Parameter Optimizer")
parser.add_argument("--num_trials", type=int, default=200)
parser.add_argument("--episodes_per_trial", type=int, default=2)
parser.add_argument("--fz_target", type=float, default=-20.0)
parser.add_argument("--output_dir", type=str, default="./osc_optimization")
parser.add_argument("--search_type", type=str, default="random", choices=["grid", "random"])

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
from itertools import product


from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg


def format_time(seconds):
    """Format seconds to mm:ss or hh:mm:ss"""
    if seconds >= 3600:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def evaluate_parameters(env: PolishEnv, kp_z: float, zeta_z: float, 
                       fz_target: float, num_episodes: int = 2,
                       trial_idx: int = 0) -> dict:
    """Evaluate a Kp_z, zeta_z combination with comprehensive logging."""
    
    metrics = {
        'trial_idx': trial_idx,
        'kp_z': kp_z,
        'zeta_z': zeta_z,
        'fz_target': fz_target,
        'episodes': [],
        'force_errors': [],
        'force_values': [],
        'rewards': [],
        'max_wpt_reached': [],
        'completion': [],
        'steps_per_episode': [],
        'contact_steps': [],
    }
    
    max_wpt = env.traj_mgr.T - 1
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        # Set parameters
        env.dynamic_kp[:, 2] = kp_z
        if hasattr(env, 'dynamic_zeta'):
            env.dynamic_zeta[:, 2] = zeta_z
        
        done = False
        episode_forces = []
        episode_positions = []
        episode_reward = 0.0
        max_wpt_this_ep = 0
        step_count = 0
        contact_count = 0
        
        episode_data = {
            'episode_idx': ep,
            'kp_z': kp_z,
            'zeta_z': zeta_z,
            'timesteps': [],
        }
        
        while not done:
            action = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
            
            current_wpt = env.wpt_idx[0].item()
            max_wpt_this_ep = max(max_wpt_this_ep, current_wpt)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            current_wpt = env.wpt_idx[0].item()
            max_wpt_this_ep = max(max_wpt_this_ep, current_wpt)
            
            step_count += 1
            
            # Get position
            if hasattr(env, '_ee_pos_w'):
                pos = env._ee_pos_w[0].cpu().numpy().tolist()
            else:
                pos = [0, 0, 0]
            
            # Get force in contact phases
            fz = 0.0
            phase = 0
            if hasattr(env, 'phase_ctrl'):
                phase = env.phase_ctrl.phase[0].item()
                if phase >= 3:
                    fz = env._fz_ema[0].item()
                    episode_forces.append(fz)
                    contact_count += 1
            
            # Log timestep data (every 10 steps to save space)
            if step_count % 10 == 0:
                episode_data['timesteps'].append({
                    'step': step_count,
                    'wpt_idx': current_wpt,
                    'phase': phase,
                    'fz': fz,
                    'pos_x': pos[0],
                    'pos_y': pos[1],
                    'pos_z': pos[2],
                    'reward': reward[0].item(),
                })
            
            episode_reward += reward[0].item()
            done = terminated.any().item() or truncated.any().item()
        
        completed = max_wpt_this_ep >= max_wpt
        
        # Episode summary
        episode_data['total_steps'] = step_count
        episode_data['contact_steps'] = contact_count
        episode_data['max_wpt_reached'] = max_wpt_this_ep
        episode_data['completed'] = completed
        episode_data['total_reward'] = episode_reward
        if episode_forces:
            episode_data['fz_mean'] = np.mean(episode_forces)
            episode_data['fz_std'] = np.std(episode_forces)
            episode_data['fz_min'] = np.min(episode_forces)
            episode_data['fz_max'] = np.max(episode_forces)
        
        metrics['episodes'].append(episode_data)
        metrics['rewards'].append(episode_reward)
        metrics['max_wpt_reached'].append(max_wpt_this_ep)
        metrics['completion'].append(completed)
        metrics['steps_per_episode'].append(step_count)
        metrics['contact_steps'].append(contact_count)
        
        if episode_forces:
            metrics['force_values'].extend(episode_forces)
            errors = [abs(f - fz_target) for f in episode_forces]
            metrics['force_errors'].extend(errors)
    
    # Aggregate metrics
    if metrics['force_errors']:
        metrics['force_error_mean'] = float(np.mean(metrics['force_errors']))
        metrics['force_error_std'] = float(np.std(metrics['force_errors']))
        metrics['force_stability'] = float(np.var(metrics['force_values']))
        metrics['force_mean'] = float(np.mean(metrics['force_values']))
        metrics['force_min'] = float(np.min(metrics['force_values']))
        metrics['force_max'] = float(np.max(metrics['force_values']))
    else:
        metrics['force_error_mean'] = float('inf')
        metrics['force_error_std'] = float('inf')
        metrics['force_stability'] = float('inf')
        metrics['force_mean'] = 0.0
        metrics['force_min'] = 0.0
        metrics['force_max'] = 0.0
    
    metrics['completion_rate'] = float(np.mean(metrics['completion']))
    metrics['avg_max_wpt'] = float(np.mean(metrics['max_wpt_reached']))
    metrics['avg_reward'] = float(np.mean(metrics['rewards']))
    metrics['avg_steps'] = float(np.mean(metrics['steps_per_episode']))
    metrics['avg_contact_steps'] = float(np.mean(metrics['contact_steps']))
    
    # Composite score
    metrics['score'] = (
        0.5 * metrics['force_error_mean'] +
        0.3 * np.sqrt(max(0, metrics['force_stability'])) +
        0.2 * (1.0 - metrics['completion_rate']) * 50
    )
    
    return metrics


def generate_search_space(search_type: str, num_trials: int) -> list:
    if search_type == "grid":
        # Grid search
        n_kp = int(np.sqrt(num_trials))
        n_zeta = num_trials // n_kp
        kp_z_values = np.linspace(500, 6000, n_kp)
        zeta_z_values = np.linspace(0.3, 2.0, n_zeta)
        combos = list(product(kp_z_values, zeta_z_values))
        return combos[:num_trials]
    else:
        # Random search with broader range
        combos = []
        for _ in range(num_trials):
            kp_z = np.random.uniform(200, 8000)
            zeta_z = np.random.uniform(0.2, 2.5)
            combos.append((kp_z, zeta_z))
        return combos


def main():
    total_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"  OSC Parameter Optimizer - Comprehensive Logging")
    print(f"{'='*80}")
    print(f"  Search type:     {args.search_type}")
    print(f"  Num trials:      {args.num_trials}")
    print(f"  Episodes/trial:  {args.episodes_per_trial}")
    print(f"  Target Fz:       {args.fz_target} N")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Timestamp:       {timestamp}")
    print(f"{'='*80}\n")
    
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = 1
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    env.fz_target = args.fz_target
    
    search_space = generate_search_space(args.search_type, args.num_trials)
    
    results = []
    all_episodes = []
    best_score = float('inf')
    best_params = None
    best_trial_idx = -1
    
    # CSV file for quick analysis
    csv_file = os.path.join(args.output_dir, f"trials_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial', 'kp_z', 'zeta_z', 'score', 'fz_error_mean', 'fz_error_std',
            'fz_mean', 'fz_min', 'fz_max', 'fz_stability', 'completion_rate',
            'avg_max_wpt', 'avg_reward', 'avg_steps', 'avg_contact_steps',
            'trial_time_s', 'total_time_s'
        ])
    
    print(f"Starting optimization with {len(search_space)} trials...")
    print(f"Total waypoints: {env.traj_mgr.T}\n")
    print(f"{'#':<4} {'Kp_z':<7} {'ζ':<5} {'Score':<6} {'Fz_err':<6} {'Fz_avg':<7} {'Wpt':<4} {'Done':<5} {'Time':<9} {'Total':<10} {'ETA':<10}")
    print("-" * 85)
    
    for i, (kp_z, zeta_z) in enumerate(search_space):
        trial_start = time.time()
        
        try:
            metrics = evaluate_parameters(
                env, kp_z, zeta_z, args.fz_target, 
                num_episodes=args.episodes_per_trial,
                trial_idx=i
            )
            
            trial_time = time.time() - trial_start
            total_time = time.time() - total_start
            
            # Estimate ETA
            avg_trial_time = total_time / (i + 1)
            remaining_trials = len(search_space) - (i + 1)
            eta = avg_trial_time * remaining_trials
            
            # Store results
            results.append(metrics)
            all_episodes.extend(metrics['episodes'])
            
            # Update CSV
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    i, kp_z, zeta_z, metrics['score'], metrics['force_error_mean'],
                    metrics['force_error_std'], metrics['force_mean'], 
                    metrics['force_min'], metrics['force_max'],
                    metrics['force_stability'], metrics['completion_rate'],
                    metrics['avg_max_wpt'], metrics['avg_reward'],
                    metrics['avg_steps'], metrics['avg_contact_steps'],
                    trial_time, total_time
                ])
            
            star = ""
            if metrics['score'] < best_score:
                best_score = metrics['score']
                best_params = (kp_z, zeta_z)
                best_trial_idx = i
                star = "★"
            
            print(f"{i+1:<4} {kp_z:<7.0f} {zeta_z:<5.2f} {metrics['score']:<6.2f} "
                  f"{metrics['force_error_mean']:<6.2f} {metrics['force_mean']:<7.1f} "
                  f"{metrics['avg_max_wpt']:<4.0f} {metrics['completion_rate']*100:<5.0f}% "
                  f"{format_time(trial_time):<9} {format_time(total_time):<10} {format_time(eta):<10} {star}")
            
            # Save detailed JSON every 20 trials
            if (i + 1) % 20 == 0:
                checkpoint_file = os.path.join(args.output_dir, f"checkpoint_{timestamp}_trial{i+1}.json")
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'config': {
                            'search_type': args.search_type,
                            'num_trials': args.num_trials,
                            'fz_target': args.fz_target,
                            'trials_completed': i + 1,
                            'total_time_seconds': total_time,
                        },
                        'best_params': {
                            'trial_idx': best_trial_idx,
                            'kp_z': best_params[0] if best_params else None,
                            'zeta_z': best_params[1] if best_params else None,
                            'score': best_score,
                        },
                        'results_summary': [{
                            'trial': r['trial_idx'],
                            'kp_z': r['kp_z'],
                            'zeta_z': r['zeta_z'],
                            'score': r['score'],
                            'force_error_mean': r['force_error_mean'],
                            'force_mean': r['force_mean'],
                            'completion_rate': r['completion_rate'],
                        } for r in results]
                    }, f, indent=2)
                print(f"  [Checkpoint saved: {checkpoint_file}]")
                
        except Exception as e:
            trial_time = time.time() - trial_start
            total_time = time.time() - total_start
            print(f"{i+1:<4} {kp_z:<7.0f} {zeta_z:<5.2f} ERROR: {str(e)[:40]}")
    
    env.close()
    
    total_elapsed = time.time() - total_start
    
    # Save final comprehensive results
    final_results_file = os.path.join(args.output_dir, f"final_results_{timestamp}.json")
    with open(final_results_file, 'w') as f:
        json.dump({
            'config': {
                'search_type': args.search_type,
                'num_trials': args.num_trials,
                'episodes_per_trial': args.episodes_per_trial,
                'fz_target': args.fz_target,
                'total_time_seconds': total_elapsed,
                'timestamp': timestamp,
            },
            'best_params': {
                'trial_idx': best_trial_idx,
                'kp_z': best_params[0] if best_params else None,
                'zeta_z': best_params[1] if best_params else None,
                'score': best_score,
            },
            'all_results': [{
                'trial_idx': r['trial_idx'],
                'kp_z': r['kp_z'],
                'zeta_z': r['zeta_z'],
                'score': r['score'],
                'force_error_mean': r['force_error_mean'],
                'force_error_std': r['force_error_std'],
                'force_mean': r['force_mean'],
                'force_min': r['force_min'],
                'force_max': r['force_max'],
                'force_stability': r['force_stability'],
                'completion_rate': r['completion_rate'],
                'avg_max_wpt': r['avg_max_wpt'],
                'avg_reward': r['avg_reward'],
                'avg_steps': r['avg_steps'],
                'avg_contact_steps': r['avg_contact_steps'],
            } for r in results]
        }, f, indent=2)
    
    # Save all episode data for detailed analysis
    episodes_file = os.path.join(args.output_dir, f"all_episodes_{timestamp}.json")
    with open(episodes_file, 'w') as f:
        json.dump(all_episodes, f)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total time: {format_time(total_elapsed)}")
    print(f"  Avg time/trial: {format_time(total_elapsed/len(search_space))}")
    print(f"  Trials completed: {len(results)}/{len(search_space)}")
    
    if best_params:
        print(f"\n  BEST PARAMETERS (Trial {best_trial_idx}):")
        print(f"    Kp_z  = {best_params[0]:.1f} N/m")
        print(f"    ζ_z   = {best_params[1]:.3f}")
        print(f"    Score = {best_score:.3f}")
    
    # Top 10
    print(f"\n  Top 10 configurations:")
    print(f"  {'-'*65}")
    sorted_results = sorted(results, key=lambda x: x['score'])[:10]
    for rank, r in enumerate(sorted_results, 1):
        print(f"  {rank:2}. Trial {r['trial_idx']:3} | Kp_z={r['kp_z']:6.0f}, ζ={r['zeta_z']:.2f} | "
              f"Score={r['score']:.2f}, Fz={r['force_mean']:.1f}N, Err={r['force_error_mean']:.2f}")
    
    print(f"\n  Output files:")
    print(f"    - CSV:      {csv_file}")
    print(f"    - Results:  {final_results_file}")
    print(f"    - Episodes: {episodes_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
