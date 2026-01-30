#!/usr/bin/env python3
"""
Extract trajectories from trained policy.
Saves position, force, Kp_z, and zeta_z values over time.

Usage:
    ./isaaclab.sh -p scripts/extract_trajectories.py --checkpoint <path> --output <dir>
"""
import _path_setup

import argparse
import os
import sys

# Isaac Lab imports MUST come first  
from isaaclab.app import AppLauncher

# Parse arguments before Isaac Lab init
parser = argparse.ArgumentParser(description="Extract trajectories from trained policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
parser.add_argument("--output", type=str, default=str(_path_setup.DATA_ROOT / "trajectories"), 
                    help="Output directory for trajectory data")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to record")
parser.add_argument("--seed", type=int, default=0, help="Random seed for env and policy rollouts")

# Append AppLauncher CLI args (includes --headless automatically)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import everything else
import torch
import numpy as np
import gymnasium as gym
import json
from datetime import datetime
import random

# Add parent to path

# ============================================================================
# POLICY NETWORK (must match training architecture)
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


def _infer_obs_dim(model_state, default_obs_dim: int = 3456) -> int:
    key = "a2c_network.actor_mlp.0.weight"
    if key in model_state:
        return int(model_state[key].shape[1])
    for k, v in model_state.items():
        if k.endswith("actor_mlp.0.weight"):
            return int(v.shape[1])
    return default_obs_dim


def _infer_action_dim(model_state, default_action_dim: int = 1) -> int:
    key = "a2c_network.mu.weight"
    if key in model_state:
        return int(model_state[key].shape[0])
    for k, v in model_state.items():
        if k.endswith("mu.weight"):
            return int(v.shape[0])
    return default_action_dim


def load_policy(checkpoint_path: str, device: str = "cuda:0"):
    """Load trained policy from rl_games checkpoint."""
    print(f"\n[INFO] Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint
    
    obs_dim = _infer_obs_dim(model_state, default_obs_dim=3456)
    action_dim = _infer_action_dim(model_state, default_action_dim=1)

    # Create policy network
    policy = PolicyNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=[512, 128, 64])
    
    # Map rl_games weights to our network
    # rl_games uses: a2c_network.actor_mlp.0.weight, etc
    new_state = {}
    
    # Layer mapping: 
    # rl_games actor_mlp.0 -> mlp.0 (Linear 3456->512)
    # rl_games actor_mlp.2 -> mlp.2 (Linear 512->128)  
    # rl_games actor_mlp.4 -> mlp.4 (Linear 128->64)
    # rl_games mu.weight -> mlp.6.weight (Linear 64->1)
    
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
                print(f"  Mapped {full_rl_key} -> {full_local_key}")
    
    policy.load_state_dict(new_state)
    policy.to(device)
    policy.eval()
    
    return policy, action_dim


def main():
    """Main function to extract trajectories."""
    
    # Import environment AFTER Isaac Lab is initialized
    try:
        import robo_pp_fixed
        import robo_pp_fixed.Polish_Env_OSC as polish_env
        from robo_pp_fixed import PolishEnvCfg
        env_id = "Polish-Fixed"
    except ImportError as e:
        print(f"[ERROR] Cannot import robo_pp_fixed: {e}")
        print("Make sure the VTT_RL_PT extension is installed or scripts/_path_setup.py can locate source/vtt_rl_pt")
        return
    
    device = "cuda:0"

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load policy
    policy, action_dim = load_policy(args.checkpoint, device)

    if action_dim > 1:
        polish_env.KP_DELTA = True
        polish_env.DAMPING_DELTA = True
        polish_env.OBS_DAMPING = True
        polish_env.TRAIN_AXES = [2]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"traj_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Extracting Trajectories")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Episodes: {args.num_episodes}")
    print(f"{'='*60}\n")
    
    # Create environment
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    if action_dim > 1:
        env_cfg.reward_type = "kz_dz"
    
    env = gym.make(env_id, cfg=env_cfg, render_mode=None)
    
    # Get actual environment timestep (respects decimation)
    env_dt = env.unwrapped.cfg.decimation * env.unwrapped.cfg.sim.dt
    print(f"[CONFIG] Environment timestep: {env_dt:.3f}s (decimation={env.unwrapped.cfg.decimation}, sim_dt={env.unwrapped.cfg.sim.dt:.4f})")
    
    # Storage for all episodes
    all_episodes = []
    
    for ep in range(args.num_episodes):
        print(f"\n[Episode {ep+1}/{args.num_episodes}]")
        
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        step = 0
        
        # Episode storage
        episode_data = {
            "episode": ep,
            "timestamps": [],
            "tcp_positions": [],
            "tcp_orientations": [],
            "contact_forces": [],
            "contact_forces_xyz": [],
            "contact_forces_xy": [],
            "kp_z_values": [],
            "zeta_z_values": [],
            "actions": [],
            "action_dkp": [],
            "action_ddz": [],
            "rewards": [],
        }
        
        while not done:
            # Get observation tensor
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
            
            # Get action from policy
            with torch.no_grad():
                action = policy(obs_tensor)
            action = torch.clamp(action, -1.0, 1.0)
            
            action_np = action.cpu().numpy().flatten()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() if hasattr(terminated, 'any') else terminated
            
            # Extract data from environment
            try:
                env_unwrapped = env.unwrapped
                
                # TCP position (end-effector)
                if hasattr(env_unwrapped, 'robot') and hasattr(env_unwrapped.robot, 'data'):
                    ee_body_idx = env_unwrapped.ee_body_idx
                    tcp_pos = env_unwrapped.robot.data.body_pos_w[0, ee_body_idx].cpu().numpy().tolist()
                else:
                    tcp_pos = [0, 0, 0]
                
                # Contact force from carpet sensor
                if hasattr(env_unwrapped, 'carpet_sensor'):
                    env_unwrapped.carpet_sensor.update(env_unwrapped.physics_dt)
                    f_ext = env_unwrapped.carpet_sensor.data.net_forces_w[0, 0]  # (3,) world frame
                    contact_fx = float(f_ext[0].cpu().numpy())
                    contact_fy = float(f_ext[1].cpu().numpy())
                    contact_fz = float(f_ext[2].cpu().numpy())
                elif hasattr(env_unwrapped, '_fz_ema'):
                    contact_fx = 0.0
                    contact_fy = 0.0
                    contact_fz = float(env_unwrapped._fz_ema[0].cpu().numpy())
                else:
                    contact_fx = 0.0
                    contact_fy = 0.0
                    contact_fz = 0.0
                
                # Kp_z value (Z-axis stiffness)
                if hasattr(env_unwrapped, 'dynamic_kp'):
                    kp_z = float(env_unwrapped.dynamic_kp[0, 2].cpu().numpy())  # Z-axis is index 2
                else:
                    kp_z = 0.0

                # zeta_z value (Z-axis damping ratio)
                if hasattr(env_unwrapped, 'dynamic_zeta'):
                    zeta_z = float(env_unwrapped.dynamic_zeta[0, 2].cpu().numpy())
                else:
                    zeta_z = 0.0
                    
            except Exception as e:
                print(f"[WARNING] Error extracting data: {e}")
                tcp_pos = [0, 0, 0]
                contact_fz = 0.0
                kp_z = 0.0
                zeta_z = 0.0
            
            # Store data
            episode_data["timestamps"].append(step * env_dt)  # Use actual env dt (respects decimation)
            episode_data["tcp_positions"].append(tcp_pos)
            episode_data["contact_forces"].append(contact_fz)
            episode_data["contact_forces_xyz"].append([contact_fx, contact_fy, contact_fz])
            episode_data["contact_forces_xy"].append((contact_fx ** 2 + contact_fy ** 2) ** 0.5)
            episode_data["kp_z_values"].append(kp_z)
            episode_data["zeta_z_values"].append(zeta_z)
            episode_data["actions"].append(action_np.tolist())
            episode_data["action_dkp"].append(float(action_np[0]) if action_np.size > 0 else 0.0)
            episode_data["action_ddz"].append(float(action_np[1]) if action_np.size > 1 else 0.0)
            episode_data["rewards"].append(float(reward[0]) if hasattr(reward, '__getitem__') else float(reward))
            
            step += 1
            
            if step % 100 == 0:
                print(
                    f"  Step {step}: Fz={contact_fz:.2f}N, Kp_z={kp_z:.1f}, "
                    f"zeta_z={zeta_z:.2f}, reward={episode_data['rewards'][-1]:.3f}"
                )
        
        print(f"  Episode completed: {step} steps, total_reward={sum(episode_data['rewards']):.2f}")
        all_episodes.append(episode_data)
    
    # Save data
    output_file = os.path.join(output_dir, "trajectories.json")
    with open(output_file, 'w') as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "num_episodes": args.num_episodes,
            "seed": args.seed,
            "timestamp": timestamp,
            "episodes": all_episodes
        }, f, indent=2)
    
    print(f"\n[SUCCESS] Saved trajectories to: {output_file}")
    
    # Also save as numpy for easier analysis
    for i, ep_data in enumerate(all_episodes):
        np.savez(
            os.path.join(output_dir, f"episode_{i}.npz"),
            timestamps=np.array(ep_data["timestamps"]),
            tcp_positions=np.array(ep_data["tcp_positions"]),
            contact_forces=np.array(ep_data["contact_forces"]),
            contact_forces_xyz=np.array(ep_data["contact_forces_xyz"]),
            contact_forces_xy=np.array(ep_data["contact_forces_xy"]),
            kp_z_values=np.array(ep_data["kp_z_values"]),
            zeta_z_values=np.array(ep_data["zeta_z_values"]),
            actions=np.array(ep_data["actions"]),
            action_dkp=np.array(ep_data["action_dkp"]),
            action_ddz=np.array(ep_data["action_ddz"]),
            rewards=np.array(ep_data["rewards"])
        )
    
    print(f"[SUCCESS] Saved {len(all_episodes)} episodes as .npz files")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    all_forces = []
    all_kpz = []
    all_zeta = []
    all_rewards = []
    
    for ep in all_episodes:
        all_forces.extend(ep["contact_forces"])
        all_kpz.extend(ep["kp_z_values"])
        all_zeta.extend(ep.get("zeta_z_values", []))
        all_rewards.append(sum(ep["rewards"]))
    
    forces = np.array(all_forces)
    kpz = np.array(all_kpz)
    
    print(f"Contact Force (Fz):")
    print(f"  Mean: {np.mean(forces):.2f} N")
    print(f"  Std:  {np.std(forces):.2f} N")
    print(f"  Min:  {np.min(forces):.2f} N")
    print(f"  Max:  {np.max(forces):.2f} N")
    
    print(f"\nKp_z Values:")
    print(f"  Mean: {np.mean(kpz):.1f}")
    print(f"  Std:  {np.std(kpz):.1f}")
    print(f"  Min:  {np.min(kpz):.1f}")
    print(f"  Max:  {np.max(kpz):.1f}")

    if all_zeta:
        zeta = np.array(all_zeta)
        print(f"\nZeta_z Values:")
        print(f"  Mean: {np.mean(zeta):.2f}")
        print(f"  Std:  {np.std(zeta):.2f}")
        print(f"  Min:  {np.min(zeta):.2f}")
        print(f"  Max:  {np.max(zeta):.2f}")
    
    print(f"\nEpisode Rewards:")
    print(f"  Mean: {np.mean(all_rewards):.2f}")
    print(f"  Std:  {np.std(all_rewards):.2f}")
    
    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
