#!/usr/bin/env python3
"""
Evaluate best policy from grid search and compare with baseline.
"""
import _path_setup

import argparse
import os
import sys
import json
from datetime import datetime


from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate best Kp_z policy")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--steps", type=int, default=2000, help="Steps per evaluation")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--output_dir", type=str, default=str(_path_setup.DATA_ROOT / "eval_results"))
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import numpy as np
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg


class PolicyNetwork(nn.Module):
    """MLP policy matching rl_games A2C architecture."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # Architecture: 3456 -> 512 -> 128 -> 64 -> 1
        self.actor_mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
        )
        self.mu = nn.Linear(64, act_dim)
        
    def forward(self, obs):
        x = self.actor_mlp(obs)
        return self.mu(x)


def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: str = "cuda:0"):
    """Load policy from rl_games checkpoint."""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]
    
    # Create policy
    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    
    # Map weights
    new_state = {
        "actor_mlp.0.weight": model_state["a2c_network.actor_mlp.0.weight"],
        "actor_mlp.0.bias": model_state["a2c_network.actor_mlp.0.bias"],
        "actor_mlp.2.weight": model_state["a2c_network.actor_mlp.2.weight"],
        "actor_mlp.2.bias": model_state["a2c_network.actor_mlp.2.bias"],
        "actor_mlp.4.weight": model_state["a2c_network.actor_mlp.4.weight"],
        "actor_mlp.4.bias": model_state["a2c_network.actor_mlp.4.bias"],
        "mu.weight": model_state["a2c_network.mu.weight"],
        "mu.bias": model_state["a2c_network.mu.bias"],
    }
    
    policy.load_state_dict(new_state)
    policy.eval()
    
    # Get normalization params
    obs_mean = model_state["running_mean_std.running_mean"].float().to(device)
    obs_var = model_state["running_mean_std.running_var"].float().to(device)
    
    print(f"✅ Policy loaded successfully")
    return policy, obs_mean, obs_var


def run_evaluation(env, policy, obs_mean, obs_var, num_steps: int, use_policy: bool = True, device="cuda:0"):
    """Run evaluation and collect data."""
    
    obs, info = env.reset()
    
    data = {
        "ee_positions": [],
        "forces_z": [],
        "kp_z_values": [],
        "rewards": [],
        "actions": [],
    }
    
    total_reward = 0.0
    
    for step in range(num_steps):
        if isinstance(obs, dict):
            obs_tensor = obs.get("policy", obs.get("obs", list(obs.values())[0]))
        else:
            obs_tensor = obs
        
        if not isinstance(obs_tensor, torch.Tensor):
            obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32, device=device)
        
        if use_policy and policy is not None:
            obs_norm = (obs_tensor - obs_mean) / torch.sqrt(obs_var + 1e-8)
            obs_norm = torch.clamp(obs_norm, -10.0, 10.0)
            
            with torch.no_grad():
                action = policy(obs_norm)
                action = torch.tanh(action)
        else:
            action = torch.zeros((args.num_envs, env.action_space.shape[0]), device=device)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        reward_val = reward.mean().item()
        total_reward += reward_val
        data["rewards"].append(reward_val)
        data["actions"].append(action[0].cpu().numpy().tolist())
        
        try:
            if hasattr(env, 'ee_pos_w'):
                data["ee_positions"].append(env.ee_pos_w[0].cpu().numpy().tolist())
            if hasattr(env, 'fz'):
                data["forces_z"].append(float(env.fz[0].item()))
            if hasattr(env, 'osc_kp'):
                data["kp_z_values"].append(float(env.osc_kp[0, 2].item()))
        except:
            pass
        
        done = terminated.any().item() if hasattr(terminated, 'any') else bool(terminated)
        if done:
            obs, info = env.reset()
        
        if (step + 1) % 500 == 0:
            print(f"  Step {step+1}/{num_steps} | Avg Reward: {total_reward/(step+1):.3f}")
    
    return data, total_reward


def compute_metrics(data: dict, force_target: float = -20.0):
    """Compute evaluation metrics."""
    metrics = {}
    
    if data["rewards"]:
        metrics["total_reward"] = sum(data["rewards"])
        metrics["avg_reward"] = np.mean(data["rewards"])
    
    if data["forces_z"]:
        fz = np.array(data["forces_z"])
        metrics["force_mean"] = float(np.mean(fz))
        metrics["force_std"] = float(np.std(fz))
        metrics["force_mae"] = float(np.mean(np.abs(fz - force_target)))
        metrics["force_rmse"] = float(np.sqrt(np.mean((fz - force_target)**2)))
        metrics["contact_ratio"] = float(np.mean(np.abs(fz) > 1.0))
    
    if data["kp_z_values"]:
        kp = np.array(data["kp_z_values"])
        metrics["kp_z_mean"] = float(np.mean(kp))
        metrics["kp_z_std"] = float(np.std(kp))
        metrics["kp_z_min"] = float(np.min(kp))
        metrics["kp_z_max"] = float(np.max(kp))
    
    return metrics


def print_comparison(baseline_metrics: dict, policy_metrics: dict):
    """Print comparison table."""
    print("\n" + "="*70)
    print("  COMPARISON: BASELINE vs TRAINED POLICY")
    print("="*70)
    
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Policy':>15} {'Improvement':>12}")
    print("-"*70)
    
    for key in baseline_metrics:
        base_val = baseline_metrics[key]
        pol_val = policy_metrics.get(key, 0)
        
        if "mae" in key or "rmse" in key or "std" in key:
            # Lower is better
            improvement = ((base_val - pol_val) / abs(base_val) * 100) if base_val != 0 else 0
            imp_str = f"{improvement:+.1f}%"
        else:
            improvement = ((pol_val - base_val) / abs(base_val) * 100) if base_val != 0 else 0
            imp_str = f"{improvement:+.1f}%"
        
        print(f"  {key:<23} {base_val:>15.3f} {pol_val:>15.3f} {imp_str:>12}")
    
    print("="*70)


def main():
    print("\n" + "="*60)
    print("  POLICY EVALUATION - Kp_z Only Control")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"\n📊 Obs dim: {obs_dim}, Action dim: {act_dim}")
    
    policy, obs_mean, obs_var = load_policy(args.checkpoint, obs_dim, act_dim)
    
    print("\n" + "="*60)
    print("  BASELINE EVALUATION (zero actions = fixed OSC)")
    print("="*60)
    baseline_data, _ = run_evaluation(env, None, None, None, args.steps, use_policy=False)
    baseline_metrics = compute_metrics(baseline_data)
    
    print("\n" + "="*60)
    print("  POLICY EVALUATION (adaptive Kp_z control)")
    print("="*60)
    policy_data, _ = run_evaluation(env, policy, obs_mean, obs_var, args.steps, use_policy=True)
    policy_metrics = compute_metrics(policy_data)
    
    print_comparison(baseline_metrics, policy_metrics)
    
    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "baseline_metrics": baseline_metrics,
        "policy_metrics": policy_metrics,
    }
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    np.savez(os.path.join(output_dir, "baseline_data.npz"),
             **{k: np.array(v) for k, v in baseline_data.items() if v})
    np.savez(os.path.join(output_dir, "policy_data.npz"),
             **{k: np.array(v) for k, v in policy_data.items() if v})
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
