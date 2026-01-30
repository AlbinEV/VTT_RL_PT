#!/usr/bin/env python3
"""
Evaluation script for kz_dz RL policy.
Runs episodes and saves trajectories, forces, actions for analysis.

Usage:
    python scripts/eval_kz_dz.py --checkpoint runs/kz_dz_final_06-21-14-10/nn/kz_dz_final.pth --num_episodes 5
    python scripts/eval_kz_dz.py --baseline --num_episodes 5  # Pure OSC without RL
"""
import _path_setup

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Add paths

from isaaclab.app import AppLauncher

# Parse args before Isaac
parser = argparse.ArgumentParser(description="Evaluate kz_dz policy")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file")
parser.add_argument("--baseline", action="store_true", help="Run pure OSC baseline (no RL)")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory")
args = parser.parse_args()

# Launch Isaac
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# Now import Isaac stuff
import gymnasium as gym
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg

# RL-Games imports for loading policy
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd


class PolicyNetwork(torch.nn.Module):
    """Simple MLP policy network matching RL-Games structure."""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[512, 128, 64]):
        super().__init__()
        
        # Build MLP with same structure as RL-Games
        # RL-Games: actor_mlp.0 (Linear), actor_mlp.1 (ELU), actor_mlp.2 (Linear), etc.
        self.actor_mlp = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_sizes[0]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            torch.nn.ELU(),
        )
        self.mu = torch.nn.Linear(hidden_sizes[2], action_dim)
        
        # Running mean/std for observation normalization
        self.obs_mean = None
        self.obs_var = None
        
    def forward(self, obs):
        # Ensure obs is float32 (the network weights are float32)
        obs = obs.float()
        
        # Normalize observations if we have stats
        if self.obs_mean is not None and self.obs_var is not None:
            obs_mean = self.obs_mean.float()
            obs_var = self.obs_var.float()
            obs = (obs - obs_mean) / torch.sqrt(obs_var + 1e-5)
            # Clamp normalized obs like RL-Games does
            obs = torch.clamp(obs, -5.0, 5.0)
        
        x = self.actor_mlp(obs)
        return self.mu(x)


def load_checkpoint(checkpoint_path, obs_dim, action_dim, device):
    """Load RL-Games checkpoint and extract policy."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create policy network with same architecture
    policy = PolicyNetwork(obs_dim, action_dim, hidden_sizes=[512, 128, 64]).to(device)
    
    # Extract model state
    model_state = checkpoint.get('model', checkpoint)
    
    # Map RL-Games weights to our network
    # RL-Games uses indices 0,2,4 for Linear layers (1,3,5 are ELU)
    weight_mapping = {
        'a2c_network.actor_mlp.0.weight': 'actor_mlp.0.weight',
        'a2c_network.actor_mlp.0.bias': 'actor_mlp.0.bias',
        'a2c_network.actor_mlp.2.weight': 'actor_mlp.2.weight',
        'a2c_network.actor_mlp.2.bias': 'actor_mlp.2.bias',
        'a2c_network.actor_mlp.4.weight': 'actor_mlp.4.weight',
        'a2c_network.actor_mlp.4.bias': 'actor_mlp.4.bias',
        'a2c_network.mu.weight': 'mu.weight',
        'a2c_network.mu.bias': 'mu.bias',
    }
    
    # Build state dict for our model
    new_state_dict = {}
    for rl_key, our_key in weight_mapping.items():
        if rl_key in model_state:
            new_state_dict[our_key] = model_state[rl_key]
            print(f"  Loaded: {rl_key} -> {our_key}")
    
    # Load weights
    try:
        policy.load_state_dict(new_state_dict, strict=True)
        print(f"[INFO] Successfully loaded all policy weights from {checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load weights: {e}")
        return None, None
    
    # Load observation normalization stats
    if 'running_mean_std.running_mean' in model_state:
        policy.obs_mean = model_state['running_mean_std.running_mean'].to(device)
        policy.obs_var = model_state['running_mean_std.running_var'].to(device)
        print(f"[INFO] Loaded observation normalization stats")
    
    return policy, checkpoint
    
    return policy, running_mean


def run_evaluation(args):
    """Run evaluation episodes and collect data."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "baseline" if args.baseline else "rl"
    output_dir = os.path.join(args.output_dir, f"eval_{mode}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = 1  # Single env for evaluation
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"[INFO] Obs dim: {obs_dim}, Action dim: {action_dim}")
    
    # Load policy if not baseline
    policy = None
    if not args.baseline and args.checkpoint:
        policy, _ = load_checkpoint(args.checkpoint, obs_dim, action_dim, device)
        policy.eval()
    
    # Storage for all episodes
    all_episodes_data = []
    
    # Run episodes
    for ep in range(args.num_episodes):
        print(f"\n[INFO] Episode {ep+1}/{args.num_episodes}")
        
        # Reset environment
        obs_dict, info = env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        
        # Episode data storage
        episode_data = {
            "timesteps": [],
            "fz": [],           # Measured force Z
            "fz_target": [],    # Target force
            "fz_error": [],     # Force error
            "kp_z": [],         # Stiffness Z
            "dz": [],           # Damping ratio Z
            "pos_z": [],        # End-effector Z position
            "vel_z": [],        # End-effector Z velocity
            "action_dkp": [],   # RL action delta Kp
            "action_ddz": [],   # RL action delta damping
            "reward": [],       # Step reward
        }
        
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # Get action
            if args.baseline or policy is None:
                # Baseline: zero action (use OSC defaults)
                action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
            else:
                # RL policy - IMPORTANT: policy outputs raw action values
                try:
                    with torch.no_grad():
                        if isinstance(obs, torch.Tensor):
                            obs_tensor = obs.float().to(device)  # Ensure float32 and correct device
                        else:
                            obs_tensor = torch.FloatTensor(obs).to(device)
                        
                        # Add batch dim if needed
                        if obs_tensor.dim() == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                        
                        # Run policy
                        raw_action = policy(obs_tensor)  # (1, action_dim)
                        
                        # IMPORTANT: Apply tanh to get actions in [-1, 1] like during training
                        action = torch.tanh(raw_action)
                        
                        # Debug: print first action
                        if step == 0:
                            print(f"  [DEBUG] First obs shape: {obs_tensor.shape}")
                            print(f"  [DEBUG] Raw action: {raw_action}, Tanh: {action}")
                except Exception as e:
                    if step == 0:
                        print(f"[WARN] Policy error: {e}, using zero action")
                    action = torch.zeros((1, action_dim), dtype=torch.float32, device=device)
            
            # Ensure action is tensor on correct device with batch dim
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32, device=device)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
            done = terminated or truncated
            
            # Extract data from environment internal state
            # Force Z (smoothed EMA value)
            if hasattr(env, '_fz_ema'):
                fz = env._fz_ema[0].cpu().item()
            elif hasattr(env, 'cube_sensor'):
                fz = env.cube_sensor.data.net_forces_w[0, 0, 2].cpu().item()
            else:
                fz = 0.0
            fz_target = -20.0  # Target force
            
            # Get current impedance parameters from environment (dynamic values)
            if hasattr(env, 'dynamic_kp'):
                kp_z = env.dynamic_kp[0, 2].cpu().item()
            elif hasattr(env, '_osc'):
                kp_z = env._osc.stiffness_env[0, 2].cpu().item()
            else:
                kp_z = 5000.0
                
            if hasattr(env, 'dynamic_zeta'):
                dz = env.dynamic_zeta[0, 2].cpu().item()
            elif hasattr(env, '_osc'):
                dz = env._osc.damping_ratio_env[0, 2].cpu().item()
            else:
                dz = 0.9
            
            # Get EE position/velocity from robot state
            if hasattr(env, 'robot'):
                ee_state = env.robot.data.body_state_w[:, env.robot.find_bodies("EE")[0][0], :]
                pos_z = ee_state[0, 2].cpu().item()
                vel_z = ee_state[0, 9].cpu().item()  # velocity z
            else:
                pos_z = 0.0
                vel_z = 0.0
            
            # Store data
            episode_data["timesteps"].append(step)
            episode_data["fz"].append(float(fz))
            episode_data["fz_target"].append(fz_target)
            episode_data["fz_error"].append(float(fz - fz_target))
            episode_data["kp_z"].append(float(kp_z))
            episode_data["dz"].append(float(dz))
            episode_data["pos_z"].append(float(pos_z))
            episode_data["vel_z"].append(float(vel_z))
            # Action is (1, action_dim) tensor
            episode_data["action_dkp"].append(float(action[0, 0].item()) if action.shape[1] > 0 else 0.0)
            episode_data["action_ddz"].append(float(action[0, 1].item()) if action.shape[1] > 1 else 0.0)
            episode_data["reward"].append(float(reward) if not isinstance(reward, torch.Tensor) else reward.item())
            
            step += 1
            total_reward += float(reward) if not isinstance(reward, torch.Tensor) else reward.item()
            
            # Progress every 100 steps
            if step % 100 == 0:
                act_kp = float(action[0, 0].item()) if action.shape[1] > 0 else 0.0
                act_dz = float(action[0, 1].item()) if action.shape[1] > 1 else 0.0
                print(f"  Step {step}: Fz={fz:.2f}N, Kp={kp_z:.0f}, ζ={dz:.3f}, R={total_reward:.1f}, act=[{act_kp:.4f}, {act_dz:.4f}]")
        
        print(f"  Episode complete: {step} steps, Total reward: {total_reward:.2f}")
        
        # Convert to numpy arrays
        for key in episode_data:
            episode_data[key] = np.array(episode_data[key])
        
        all_episodes_data.append(episode_data)
        
        # Save episode data
        ep_file = os.path.join(output_dir, f"episode_{ep:02d}.npz")
        np.savez(ep_file, **episode_data)
        print(f"  Saved to {ep_file}")
    
    # Save summary
    summary = {
        "mode": mode,
        "checkpoint": args.checkpoint if not args.baseline else "baseline",
        "num_episodes": args.num_episodes,
        "total_rewards": [np.sum(ep["reward"]) for ep in all_episodes_data],
        "mean_fz_error": [np.mean(np.abs(ep["fz_error"])) for ep in all_episodes_data],
        "final_fz": [ep["fz"][-1] for ep in all_episodes_data],
    }
    
    summary_file = os.path.join(output_dir, "summary.npz")
    np.savez(summary_file, **summary)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Mode: {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Mean total reward: {np.mean(summary['total_rewards']):.2f} ± {np.std(summary['total_rewards']):.2f}")
    print(f"Mean |Fz error|: {np.mean(summary['mean_fz_error']):.3f} N")
    print(f"Final Fz values: {[f'{x:.2f}' for x in summary['final_fz']]}")
    print(f"Results saved to: {output_dir}")
    
    # Close environment
    env.close()
    
    return output_dir


if __name__ == "__main__":
    output_dir = run_evaluation(args)
    simulation_app.close()
