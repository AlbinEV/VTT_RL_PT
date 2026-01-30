#!/usr/bin/env python3
"""
Comparative analysis: Baseline OSC vs RL-trained policy
Runs both approaches and compares force tracking performance.

Usage:
    ./isaaclab.sh -p scripts/compare_baseline_vs_rl.py --checkpoint <path> --episodes 5
"""
import _path_setup

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare Baseline vs RL policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
parser.add_argument("--num_episodes", type=int, default=5, help="Episodes per approach")
parser.add_argument("--output", type=str, default=str(_path_setup.DATA_ROOT / "comparisons"), 
                    help="Output directory")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym


# ============================================================================
# POLICY NETWORK
# ============================================================================

class PolicyNetwork(torch.nn.Module):
    """MLP policy matching rl_games architecture."""
    
    def __init__(self, obs_dim: int = 3456, action_dim: int = 1, hidden_sizes: list = [512, 128, 64]):
        super().__init__()
        
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(torch.nn.ELU())
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, action_dim))
        
        self.mlp = torch.nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.mlp(obs)


def load_policy(checkpoint_path: str, device: str = "cuda:0"):
    """Load trained policy from rl_games checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint
    
    policy = PolicyNetwork(obs_dim=3456, action_dim=1, hidden_sizes=[512, 128, 64])
    
    # Map rl_games weights
    new_state = {}
    layer_map = {
        'a2c_network.actor_mlp.0': 'mlp.0',
        'a2c_network.actor_mlp.2': 'mlp.2', 
        'a2c_network.actor_mlp.4': 'mlp.4',
        'a2c_network.mu': 'mlp.6',
    }
    
    for rl_key, local_key in layer_map.items():
        for suffix in ['.weight', '.bias']:
            full_rl_key = rl_key + suffix
            full_local_key = local_key + suffix
            if full_rl_key in model_state:
                new_state[full_local_key] = model_state[full_rl_key]
    
    policy.load_state_dict(new_state)
    policy.to(device)
    policy.eval()
    
    return policy


def run_episodes(env, policy=None, num_episodes=5, device="cuda:0"):
    """Run episodes with baseline (policy=None) or trained policy."""
    
    all_data = {
        'forces': [],
        'kp_z_values': [],
        'actions': [],
        'rewards': [],
        'tcp_positions': [],
        'timestamps': []
    }
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        ep_forces = []
        ep_kpz = []
        ep_actions = []
        ep_rewards = []
        ep_tcp = []
        ep_time = []
        
        while not done:
            if policy is None:
                # BASELINE: zero actions (OSC nominal)
                action = torch.zeros(1, 1, device=device)
                action_np = np.zeros(1)
            else:
                # RL POLICY: trained actions
                if isinstance(obs, dict):
                    obs_tensor = obs.get("policy", obs.get("obs", None))
                    if obs_tensor is None:
                        obs_tensor = list(obs.values())[0]
                else:
                    obs_tensor = obs
                
                if not isinstance(obs_tensor, torch.Tensor):
                    obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32, device=device)
                
                if obs_tensor.dim() == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    action_raw = policy(obs_tensor)
                    action = torch.tanh(action_raw)  # Apply tanh!
                
                action_np = action.cpu().numpy().flatten()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() if hasattr(terminated, 'any') else terminated
            
            # Extract data
            env_unwrapped = env.unwrapped
            
            # TCP position
            if hasattr(env_unwrapped, 'robot'):
                tcp_pos = env_unwrapped.robot.data.body_pos_w[0, -1].cpu().numpy()
            else:
                tcp_pos = np.zeros(3)
            
            # Contact force
            if hasattr(env_unwrapped, 'cube_sensor'):
                fz = float(env_unwrapped.cube_sensor.data.net_forces_w[0, 0, 2].cpu().numpy())
            else:
                fz = 0.0
            
            # Kp_z (dynamic)
            if hasattr(env_unwrapped, 'dynamic_kp'):
                kp_z = float(env_unwrapped.dynamic_kp[0, 2].cpu().numpy())
            else:
                kp_z = 5000.0
            
            ep_forces.append(fz)
            ep_kpz.append(kp_z)
            ep_actions.append(action_np[0] if len(action_np) > 0 else 0.0)
            ep_rewards.append(float(reward[0]) if hasattr(reward, '__getitem__') else float(reward))
            ep_tcp.append(tcp_pos)
            ep_time.append(step * 0.01)
            
            step += 1
            if step > 1000:  # Safety limit
                break
        
        all_data['forces'].append(ep_forces)
        all_data['kp_z_values'].append(ep_kpz)
        all_data['actions'].append(ep_actions)
        all_data['rewards'].append(ep_rewards)
        all_data['tcp_positions'].append(ep_tcp)
        all_data['timestamps'].append(ep_time)
    
    return all_data


def compute_metrics(data, target_force=-20.0):
    """Compute performance metrics."""
    
    all_forces = []
    all_rewards = []
    
    for ep_forces in data['forces']:
        all_forces.extend(ep_forces)
    
    for ep_rewards in data['rewards']:
        all_rewards.append(sum(ep_rewards))
    
    forces = np.array(all_forces)
    
    metrics = {
        'force_mean': np.mean(forces),
        'force_std': np.std(forces),
        'force_mae': np.mean(np.abs(forces - target_force)),
        'force_rmse': np.sqrt(np.mean((forces - target_force)**2)),
        'force_min': np.min(forces),
        'force_max': np.max(forces),
        'reward_mean': np.mean(all_rewards),
        'reward_std': np.std(all_rewards),
    }
    
    return metrics


def plot_comparison(baseline_data, rl_data, output_dir):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Force tracking comparison
    for i, ep_forces in enumerate(baseline_data['forces'][:3]):
        t = baseline_data['timestamps'][i]
        axes[0, 0].plot(t, ep_forces, 'b-', alpha=0.4, linewidth=1)
    
    for i, ep_forces in enumerate(rl_data['forces'][:3]):
        t = rl_data['timestamps'][i]
        axes[0, 0].plot(t, ep_forces, 'r-', alpha=0.4, linewidth=1)
    
    axes[0, 0].axhline(y=-20.0, color='k', linestyle='--', label='Target')
    axes[0, 0].plot([], [], 'b-', label='Baseline OSC', linewidth=2)
    axes[0, 0].plot([], [], 'r-', label='RL Policy', linewidth=2)
    axes[0, 0].set_ylabel('Contact Force Fz [N]')
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_title('Force Tracking Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Kp_z evolution
    for i, ep_kpz in enumerate(baseline_data['kp_z_values'][:3]):
        t = baseline_data['timestamps'][i]
        axes[0, 1].plot(t, ep_kpz, 'b-', alpha=0.4, linewidth=1)
    
    for i, ep_kpz in enumerate(rl_data['kp_z_values'][:3]):
        t = rl_data['timestamps'][i]
        axes[0, 1].plot(t, ep_kpz, 'r-', alpha=0.4, linewidth=1)
    
    axes[0, 1].plot([], [], 'b-', label='Baseline', linewidth=2)
    axes[0, 1].plot([], [], 'r-', label='RL Policy', linewidth=2)
    axes[0, 1].set_ylabel('Kp_z [N/m]')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_title('Stiffness Adaptation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Force histogram
    baseline_forces = np.concatenate(baseline_data['forces'])
    rl_forces = np.concatenate(rl_data['forces'])
    
    axes[1, 0].hist(baseline_forces, bins=50, alpha=0.5, color='blue', label='Baseline', density=True)
    axes[1, 0].hist(rl_forces, bins=50, alpha=0.5, color='red', label='RL Policy', density=True)
    axes[1, 0].axvline(x=-20.0, color='k', linestyle='--', label='Target')
    axes[1, 0].set_xlabel('Contact Force Fz [N]')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Force Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics comparison bar chart
    baseline_metrics = compute_metrics(baseline_data)
    rl_metrics = compute_metrics(rl_data)
    
    metrics_names = ['MAE', 'RMSE', 'Std']
    baseline_values = [baseline_metrics['force_mae'], baseline_metrics['force_rmse'], baseline_metrics['force_std']]
    rl_values = [rl_metrics['force_mae'], rl_metrics['force_rmse'], rl_metrics['force_std']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, baseline_values, width, label='Baseline', color='blue', alpha=0.7)
    axes[1, 1].bar(x + width/2, rl_values, width, label='RL Policy', color='red', alpha=0.7)
    axes[1, 1].set_ylabel('Force Error [N]')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_vs_rl_comparison.png'), dpi=150)
    print(f"✅ Saved comparison plot")


def main():
    """Main comparison function."""
    
    import robo_pp_fixed
    from robo_pp_fixed import PolishEnvCfg
    
    device = "cuda:0"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"comparison_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  BASELINE vs RL POLICY COMPARISON")
    print("="*70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Episodes per approach: {args.num_episodes}")
    print(f"Output: {output_dir}\n")
    
    # Load trained policy
    print("Loading trained policy...")
    policy = load_policy(args.checkpoint, device)
    print("✅ Policy loaded\n")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = 1
    env = gym.make("Polish-Fixed-v0", cfg=env_cfg, render_mode=None)
    
    # Run BASELINE (no RL, pure OSC)
    print("="*70)
    print("Running BASELINE (pure OSC, zero actions)...")
    print("="*70)
    baseline_data = run_episodes(env, policy=None, num_episodes=args.num_episodes, device=device)
    baseline_metrics = compute_metrics(baseline_data)
    
    print("\n📊 BASELINE METRICS:")
    print(f"  Force Mean:     {baseline_metrics['force_mean']:.2f} N")
    print(f"  Force MAE:      {baseline_metrics['force_mae']:.2f} N")
    print(f"  Force RMSE:     {baseline_metrics['force_rmse']:.2f} N")
    print(f"  Force Std:      {baseline_metrics['force_std']:.2f} N")
    print(f"  Reward Mean:    {baseline_metrics['reward_mean']:.2f}")
    
    # Run RL POLICY
    print("\n" + "="*70)
    print("Running RL POLICY (trained agent)...")
    print("="*70)
    rl_data = run_episodes(env, policy=policy, num_episodes=args.num_episodes, device=device)
    rl_metrics = compute_metrics(rl_data)
    
    print("\n📊 RL POLICY METRICS:")
    print(f"  Force Mean:     {rl_metrics['force_mean']:.2f} N")
    print(f"  Force MAE:      {rl_metrics['force_mae']:.2f} N")
    print(f"  Force RMSE:     {rl_metrics['force_rmse']:.2f} N")
    print(f"  Force Std:      {rl_metrics['force_std']:.2f} N")
    print(f"  Reward Mean:    {rl_metrics['reward_mean']:.2f}")
    
    # Comparison
    print("\n" + "="*70)
    print("  IMPROVEMENT ANALYSIS")
    print("="*70)
    
    mae_improvement = ((baseline_metrics['force_mae'] - rl_metrics['force_mae']) / baseline_metrics['force_mae']) * 100
    rmse_improvement = ((baseline_metrics['force_rmse'] - rl_metrics['force_rmse']) / baseline_metrics['force_rmse']) * 100
    std_improvement = ((baseline_metrics['force_std'] - rl_metrics['force_std']) / baseline_metrics['force_std']) * 100
    reward_improvement = ((rl_metrics['reward_mean'] - baseline_metrics['reward_mean']) / abs(baseline_metrics['reward_mean'])) * 100
    
    print(f"\nForce MAE:      {mae_improvement:+.1f}% {'✅' if mae_improvement > 0 else '❌'}")
    print(f"Force RMSE:     {rmse_improvement:+.1f}% {'✅' if rmse_improvement > 0 else '❌'}")
    print(f"Force Std:      {std_improvement:+.1f}% {'✅' if std_improvement > 0 else '❌'}")
    print(f"Reward:         {reward_improvement:+.1f}% {'✅' if reward_improvement > 0 else '❌'}")
    
    # Save data (flatten lists)
    np.savez(
        os.path.join(output_dir, 'baseline_data.npz'),
        forces=np.concatenate(baseline_data['forces']),
        kp_z=np.concatenate(baseline_data['kp_z_values']),
        actions=np.concatenate(baseline_data['actions']),
        rewards=[sum(r) for r in baseline_data['rewards']]
    )
    
    np.savez(
        os.path.join(output_dir, 'rl_data.npz'),
        forces=np.concatenate(rl_data['forces']),
        kp_z=np.concatenate(rl_data['kp_z_values']),
        actions=np.concatenate(rl_data['actions']),
        rewards=[sum(r) for r in rl_data['rewards']]
    )
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE vs RL POLICY COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        f.write("BASELINE METRICS:\n")
        for key, value in baseline_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nRL POLICY METRICS:\n")
        for key, value in rl_metrics.items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nIMPROVEMENTS:\n")
        f.write(f"  MAE:  {mae_improvement:+.2f}%\n")
        f.write(f"  RMSE: {rmse_improvement:+.2f}%\n")
        f.write(f"  Std:  {std_improvement:+.2f}%\n")
        f.write(f"  Reward: {reward_improvement:+.2f}%\n")
    
    # Create plots
    print("\nCreating comparison plots...")
    plot_comparison(baseline_data, rl_data, output_dir)
    
    print("\n" + "="*70)
    print(f"✅ COMPARISON COMPLETE!")
    print("="*70)
    print(f"Results saved in: {output_dir}")
    print(f"  - baseline_data.npz")
    print(f"  - rl_data.npz")
    print(f"  - metrics.txt")
    print(f"  - baseline_vs_rl_comparison.png")
    print("="*70 + "\n")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
