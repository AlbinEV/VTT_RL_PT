#!/usr/bin/env python3
"""
Visualize evaluation results from eval_kz_dz.py.
Creates comparison plots between baseline and RL policy.

Usage:
    python scripts/plot_eval_results.py eval_results/eval_baseline_* eval_results/eval_rl_*
    python scripts/plot_eval_results.py --dir eval_results/eval_rl_20260106_*
"""
import _path_setup

import argparse
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_episode_data(npz_file):
    """Load episode data from npz file."""
    data = np.load(npz_file)
    return {key: data[key] for key in data.files}


def load_all_episodes(result_dir):
    """Load all episodes from a result directory."""
    episode_files = sorted(glob.glob(os.path.join(result_dir, "episode_*.npz")))
    episodes = []
    for f in episode_files:
        episodes.append(load_episode_data(f))
    return episodes


def plot_single_run(result_dir, output_file=None):
    """Plot results from a single evaluation run."""
    
    episodes = load_all_episodes(result_dir)
    if not episodes:
        print(f"No episodes found in {result_dir}")
        return
    
    # Determine mode from directory name
    mode = "RL" if "eval_rl" in result_dir else "Baseline (OSC)"
    
    n_episodes = len(episodes)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Color map for episodes
    colors = plt.cm.viridis(np.linspace(0, 1, n_episodes))
    
    # 1. Force Z over time
    ax1 = fig.add_subplot(gs[0, 0])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01  # Convert to seconds (assuming 100Hz)
        ax1.plot(t, ep["fz"], color=colors[i], alpha=0.7, label=f"Ep {i+1}")
    ax1.axhline(y=-20.0, color='r', linestyle='--', linewidth=2, label="Target (-20N)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Force Z (N)")
    ax1.set_title(f"Force Z - {mode}")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Force error over time
    ax2 = fig.add_subplot(gs[0, 1])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        ax2.plot(t, ep["fz_error"], color=colors[i], alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.axhline(y=2, color='orange', linestyle=':', linewidth=1, label="±2N band")
    ax2.axhline(y=-2, color='orange', linestyle=':', linewidth=1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Force Error (N)")
    ax2.set_title(f"Force Error (Fz - Target) - {mode}")
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Stiffness Kp_z over time
    ax3 = fig.add_subplot(gs[1, 0])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        ax3.plot(t, ep["kp_z"], color=colors[i], alpha=0.7)
    ax3.axhline(y=5000, color='gray', linestyle='--', linewidth=1, label="Initial (5000)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Stiffness Kp_z (N/m)")
    ax3.set_title(f"Stiffness Kp_z - {mode}")
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Damping ratio over time
    ax4 = fig.add_subplot(gs[1, 1])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        ax4.plot(t, ep["dz"], color=colors[i], alpha=0.7)
    ax4.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, label="Initial (0.9)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Damping Ratio ζ_z")
    ax4.set_title(f"Damping Ratio ζ_z - {mode}")
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. RL Actions (if available)
    ax5 = fig.add_subplot(gs[2, 0])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        if "action_dkp" in ep:
            ax5.plot(t, ep["action_dkp"], color=colors[i], alpha=0.7)
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Action ΔKp")
    ax5.set_title(f"RL Action: Delta Stiffness - {mode}")
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        if "action_ddz" in ep:
            ax6.plot(t, ep["action_ddz"], color=colors[i], alpha=0.7)
    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Action Δζ")
    ax6.set_title(f"RL Action: Delta Damping - {mode}")
    ax6.grid(True, alpha=0.3)
    
    # 6. Cumulative reward
    ax7 = fig.add_subplot(gs[3, 0])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        cumsum = np.cumsum(ep["reward"])
        ax7.plot(t, cumsum, color=colors[i], alpha=0.7, label=f"Ep {i+1}: {cumsum[-1]:.0f}")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Cumulative Reward")
    ax7.set_title(f"Cumulative Reward - {mode}")
    ax7.legend(loc='upper left', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # 7. Position Z and Velocity Z
    ax8 = fig.add_subplot(gs[3, 1])
    for i, ep in enumerate(episodes):
        t = ep["timesteps"] * 0.01
        ax8.plot(t, ep["pos_z"], color=colors[i], alpha=0.7)
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Position Z (m)")
    ax8.set_title(f"End-Effector Z Position - {mode}")
    ax8.grid(True, alpha=0.3)
    
    plt.suptitle(f"Evaluation Results: {os.path.basename(result_dir)}", fontsize=14, fontweight='bold')
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        output_file = os.path.join(result_dir, "evaluation_plots.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    
    plt.close()
    
    return output_file


def plot_comparison(baseline_dir, rl_dir, output_file=None):
    """Compare baseline vs RL results side by side."""
    
    baseline_eps = load_all_episodes(baseline_dir)
    rl_eps = load_all_episodes(rl_dir)
    
    if not baseline_eps or not rl_eps:
        print("Need both baseline and RL results for comparison")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Average over episodes
    def average_data(episodes, key):
        min_len = min(len(ep[key]) for ep in episodes)
        stacked = np.array([ep[key][:min_len] for ep in episodes])
        return np.mean(stacked, axis=0), np.std(stacked, axis=0)
    
    t_base = baseline_eps[0]["timesteps"][:min(len(ep["timesteps"]) for ep in baseline_eps)] * 0.01
    t_rl = rl_eps[0]["timesteps"][:min(len(ep["timesteps"]) for ep in rl_eps)] * 0.01
    
    # Force comparison
    ax = axes[0, 0]
    fz_base_mean, fz_base_std = average_data(baseline_eps, "fz")
    fz_rl_mean, fz_rl_std = average_data(rl_eps, "fz")
    ax.fill_between(t_base, fz_base_mean - fz_base_std, fz_base_mean + fz_base_std, alpha=0.3, color='blue')
    ax.fill_between(t_rl[:len(fz_rl_mean)], fz_rl_mean - fz_rl_std, fz_rl_mean + fz_rl_std, alpha=0.3, color='orange')
    ax.plot(t_base, fz_base_mean, 'b-', linewidth=2, label='Baseline (OSC)')
    ax.plot(t_rl[:len(fz_rl_mean)], fz_rl_mean, 'orange', linewidth=2, label='RL Policy')
    ax.axhline(y=-20.0, color='r', linestyle='--', linewidth=2, label='Target')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force Z (N)")
    ax.set_title("Force Z Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Force error comparison
    ax = axes[0, 1]
    err_base_mean, err_base_std = average_data(baseline_eps, "fz_error")
    err_rl_mean, err_rl_std = average_data(rl_eps, "fz_error")
    ax.fill_between(t_base, err_base_mean - err_base_std, err_base_mean + err_base_std, alpha=0.3, color='blue')
    ax.fill_between(t_rl[:len(err_rl_mean)], err_rl_mean - err_rl_std, err_rl_mean + err_rl_std, alpha=0.3, color='orange')
    ax.plot(t_base, np.abs(err_base_mean), 'b-', linewidth=2, label='Baseline')
    ax.plot(t_rl[:len(err_rl_mean)], np.abs(err_rl_mean), 'orange', linewidth=2, label='RL')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Force Error| (N)")
    ax.set_title("Absolute Force Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Stiffness comparison
    ax = axes[1, 0]
    kp_base_mean, kp_base_std = average_data(baseline_eps, "kp_z")
    kp_rl_mean, kp_rl_std = average_data(rl_eps, "kp_z")
    ax.fill_between(t_base, kp_base_mean - kp_base_std, kp_base_mean + kp_base_std, alpha=0.3, color='blue')
    ax.fill_between(t_rl[:len(kp_rl_mean)], kp_rl_mean - kp_rl_std, kp_rl_mean + kp_rl_std, alpha=0.3, color='orange')
    ax.plot(t_base, kp_base_mean, 'b-', linewidth=2, label='Baseline')
    ax.plot(t_rl[:len(kp_rl_mean)], kp_rl_mean, 'orange', linewidth=2, label='RL')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Stiffness Kp_z (N/m)")
    ax.set_title("Stiffness Kp_z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Damping comparison
    ax = axes[1, 1]
    dz_base_mean, dz_base_std = average_data(baseline_eps, "dz")
    dz_rl_mean, dz_rl_std = average_data(rl_eps, "dz")
    ax.fill_between(t_base, dz_base_mean - dz_base_std, dz_base_mean + dz_base_std, alpha=0.3, color='blue')
    ax.fill_between(t_rl[:len(dz_rl_mean)], dz_rl_mean - dz_rl_std, dz_rl_mean + dz_rl_std, alpha=0.3, color='orange')
    ax.plot(t_base, dz_base_mean, 'b-', linewidth=2, label='Baseline')
    ax.plot(t_rl[:len(dz_rl_mean)], dz_rl_mean, 'orange', linewidth=2, label='RL')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Damping Ratio ζ_z")
    ax.set_title("Damping Ratio ζ_z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reward comparison
    ax = axes[2, 0]
    rew_base_mean, _ = average_data(baseline_eps, "reward")
    rew_rl_mean, _ = average_data(rl_eps, "reward")
    ax.plot(t_base, np.cumsum(rew_base_mean), 'b-', linewidth=2, label=f'Baseline: {np.sum(rew_base_mean):.0f}')
    ax.plot(t_rl[:len(rew_rl_mean)], np.cumsum(rew_rl_mean), 'orange', linewidth=2, label=f'RL: {np.sum(rew_rl_mean):.0f}')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[2, 1]
    ax.axis('off')
    
    # Compute metrics
    metrics_text = f"""
    SUMMARY STATISTICS
    {'='*40}
    
    Force Error (RMSE):
      Baseline: {np.sqrt(np.mean(err_base_mean**2)):.3f} N
      RL:       {np.sqrt(np.mean(err_rl_mean**2)):.3f} N
    
    Mean |Force Error|:
      Baseline: {np.mean(np.abs(err_base_mean)):.3f} N
      RL:       {np.mean(np.abs(err_rl_mean)):.3f} N
    
    Final Force (mean):
      Baseline: {fz_base_mean[-1]:.2f} N
      RL:       {fz_rl_mean[-1]:.2f} N
    
    Total Reward (mean):
      Baseline: {np.sum(rew_base_mean):.0f}
      RL:       {np.sum(rew_rl_mean):.0f}
    
    Stiffness Range:
      Baseline: {kp_base_mean.min():.0f} - {kp_base_mean.max():.0f}
      RL:       {kp_rl_mean.min():.0f} - {kp_rl_mean.max():.0f}
    
    Damping Range:
      Baseline: {dz_base_mean.min():.3f} - {dz_base_mean.max():.3f}
      RL:       {dz_rl_mean.min():.3f} - {dz_rl_mean.max():.3f}
    """
    ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle("Baseline vs RL Policy Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {output_file}")
    else:
        plt.savefig("comparison_baseline_vs_rl.png", dpi=150, bbox_inches='tight')
        print("Saved comparison plot to comparison_baseline_vs_rl.png")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("dirs", nargs="*", help="Result directories to plot")
    parser.add_argument("--dir", type=str, help="Single directory to plot")
    parser.add_argument("--compare", action="store_true", help="Compare two directories")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    
    if args.dir:
        # Single directory
        plot_single_run(args.dir, args.output)
    elif len(args.dirs) == 1:
        # Single directory from positional
        plot_single_run(args.dirs[0], args.output)
    elif len(args.dirs) == 2 and args.compare:
        # Compare two directories
        plot_comparison(args.dirs[0], args.dirs[1], args.output)
    elif args.dirs:
        # Plot each directory
        for d in args.dirs:
            if os.path.isdir(d):
                plot_single_run(d)
    else:
        # Find latest results
        eval_dirs = glob.glob("eval_results/eval_*")
        if eval_dirs:
            latest = max(eval_dirs, key=os.path.getctime)
            print(f"Plotting latest: {latest}")
            plot_single_run(latest, args.output)
        else:
            print("No evaluation results found. Run eval_kz_dz.py first.")


if __name__ == "__main__":
    main()
