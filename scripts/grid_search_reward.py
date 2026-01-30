#!/usr/bin/env python3
"""
Grid Search for Reward Function Hyperparameters - Kp_z only training.

Runs 50 training simulations with different reward parameter combinations.
Each run is short (100 epochs) to quickly evaluate parameter effectiveness.

Usage:
    ./isaaclab.sh -p scripts/grid_search_reward.py --num_runs 50 --epochs_per_run 100
"""
import _path_setup

import argparse
import os
import sys
import json
import itertools
import random
from datetime import datetime
from pathlib import Path

# Add paths before Isaac imports

from isaaclab.app import AppLauncher

# Parse arguments before Isaac
parser = argparse.ArgumentParser(description="Grid Search for Reward Parameters")
parser.add_argument("--num_runs", type=int, default=50, help="Total number of runs")
parser.add_argument("--epochs_per_run", type=int, default=100, help="Epochs per run")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--output_dir", type=str, default="grid_search_results", help="Output directory")
parser.add_argument("--score_error_weight", type=float, default=1.0, help="Weight for force tracking error in score")
parser.add_argument("--score_osc_weight", type=float, default=1.0, help="Weight for force oscillation in score")
args = parser.parse_args()

# Launch Isaac
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import numpy as np
import traceback

# ============================================================================
# GRID SEARCH PARAMETER SPACE
# ============================================================================

# Define parameter ranges for grid search
PARAM_GRID = {
    # Force tracking reward weight
    "w_force_tracking": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
    
    # Force error scale (denominator in exp)
    "force_error_scale": [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0],
    
    # Adaptive Kp reward weight
    "w_adaptive_kpz": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
    
    # Optimal Kp center (normalized 0-1)
    "optimal_kpz_center": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    
    # Kp change penalty weight
    "w_kpz_change_penalty": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0],
    
    # Contact bonus
    "w_contact_bonus": [0.0, 10.0, 20.0, 30.0, 40.0, 60.0],
}

def generate_random_params(num_samples: int) -> list:
    """Generate random parameter combinations from the grid."""
    params_list = []
    
    for _ in range(num_samples):
        params = {
            key: random.choice(values) 
            for key, values in PARAM_GRID.items()
        }
        params_list.append(params)
    
    return params_list


def setup_shared_env(args):
    """Create a single shared environment for all runs."""
    from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg
    from rl_games.common import env_configurations, vecenv
    from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = PolishEnv(cfg=env_cfg, render_mode=None)

    wrapped_env = RlGamesVecEnvWrapper(env, rl_device="cuda:0", clip_obs=10.0, clip_actions=1.0)
    env_name = "grid_search_env"
    vecenv.register(env_name, lambda config_name, num_actors, **kwargs: wrapped_env)
    env_configurations.register(env_name, {
        "vecenv_type": env_name,
        "env_creator": lambda **kwargs: wrapped_env,
    })

    return env, env_name


def patch_reward_function(params: dict):
    """
    Dynamically patch the reward function with new parameters.
    This modifies the reward computation on-the-fly.
    """
    import robo_pp_fixed.rewards.polish_kpz_control_reward as reward_module
    import robo_pp_fixed.rewards as rewards_pkg
    
    # Store original function if not already done
    if not hasattr(reward_module, '_original_compute_reward'):
        reward_module._original_compute_reward = reward_module._compute_polish_kpz_control_reward
    
    # Create patched version with custom parameters
    def patched_reward(env):
        """Patched reward function with grid search parameters."""
        import torch
        from robo_pp_fixed.cfg.config import F_TOUCH, F_LOST
        from isaaclab.utils.math import quat_apply
        from robo_pp_fixed.Trajectory_Manager import FixedPhase
        
        # Extract parameters
        w_force_tracking = params.get("w_force_tracking", 8.0)
        force_error_scale = params.get("force_error_scale", 0.7)
        w_adaptive_kpz = params.get("w_adaptive_kpz", 5.0)
        optimal_kpz_center = params.get("optimal_kpz_center", 0.6)
        w_kpz_change_penalty = params.get("w_kpz_change_penalty", 1.0)
        w_contact_bonus = params.get("w_contact_bonus", 30.0)
        
        # Phase masks
        mask_home = (env.phase_ctrl.phase == FixedPhase.HOME).float()
        mask_approach = (env.phase_ctrl.phase == FixedPhase.APPROACH).float()
        mask_descent = (env.phase_ctrl.phase == FixedPhase.DESCENT).float()
        mask_contact = (env.phase_ctrl.phase == FixedPhase.CONTACT).float()
        mask_rise = (env.phase_ctrl.phase == FixedPhase.RISE).float()
        
        # State info
        pos_err = env._last_pos_err
        ori_err = env._last_ori_err
        ee_pos = env.robot.data.body_pos_w[:, env.ee_body_idx]
        ee_vel = env.robot.data.body_lin_vel_w[:, env.ee_body_idx]
        
        # Force info
        f_ext_global = env.carpet_sensor.data.net_forces_w[:, 0]
        ee_quat = env.robot.data.body_quat_w[:, env.ee_body_idx]
        q_conj = torch.cat([ee_quat[:, 0:1], -ee_quat[:, 1:]], dim=1)
        f_ext_local = quat_apply(q_conj, f_ext_global)
        fx_local, fy_local, fz_local = f_ext_local.unbind(dim=1)
        
        current_kpz = env.dynamic_kp[:, 2]
        
        # ===== HOME PHASE =====
        r_home = mask_home * (torch.exp(-6.0 * pos_err) + torch.exp(-2.0 * ori_err))
        r_home += mask_home * ((pos_err < 0.05) & (ori_err < 0.3)).float() * 5.0
        
        # ===== APPROACH PHASE =====
        r_approach = mask_approach * (torch.exp(-10.0 * pos_err) + torch.exp(-3.0 * ori_err))
        if hasattr(env.traj_mgr, 'p_traj_env') and env.traj_mgr.p_traj_env.shape[0] > 0:
            target_z = env.traj_mgr.p_traj_env[torch.arange(env.num_envs), env.wpt_idx, 2]
        else:
            target_z = torch.full((env.num_envs,), 0.1, device=env.device)
        z_error = torch.abs(ee_pos[:, 2] - target_z)
        r_approach += mask_approach * torch.exp(-20.0 * z_error)
        
        # ===== DESCENT PHASE =====
        descent_vel = -ee_vel[:, 2]
        r_descent = mask_descent * torch.where(
            descent_vel > 0.01, descent_vel / 0.06 * 6.0, torch.tensor(-2.0, device=env.device)
        )
        
        surface_z = 0.0
        distance_to_surface = torch.clamp(ee_pos[:, 2] - surface_z, 0.0, 0.2)
        r_descent += mask_descent * torch.exp(-15.0 * distance_to_surface) * 3.0
        
        # Contact bonus (parameterized)
        contact_established = (fz_local < F_TOUCH)
        r_descent += mask_descent * contact_established.float() * w_contact_bonus
        
        # ===== CONTACT PHASE (MAIN FOCUS) =====
        r_contact = mask_contact * (torch.exp(-15.0 * pos_err) + torch.exp(-5.0 * ori_err)) * 0.5
        
        # Force tracking (parameterized)
        lateral_speed = torch.norm(ee_vel[:, :2], dim=1)
        adaptive_fz_target = env.fz_target - 1.5 * torch.clamp(lateral_speed / 0.05, 0.0, 1.0)
        fz_error = torch.abs(fz_local - adaptive_fz_target)
        r_contact += mask_contact * torch.exp(-fz_error / (env.fz_eps * force_error_scale)) * w_force_tracking
        
        # Force stability
        if hasattr(env, '_prev_fz_local'):
            fz_change = torch.abs(fz_local - env._prev_fz_local)
            r_contact += mask_contact * torch.exp(-fz_change / 2.0) * 4.0
            env._prev_fz_local = fz_local.clone()
        else:
            env._prev_fz_local = fz_local.clone()
            fz_change = torch.zeros_like(fz_local)
        
        # Optimal Kp range (parameterized center)
        kpz_normalized = (current_kpz - env.kp_lo[2]) / (env.kp_hi[2] - env.kp_lo[2])
        optimal_kpz_range = torch.exp(-((kpz_normalized - optimal_kpz_center) / 0.3)**2)
        r_contact += mask_contact * optimal_kpz_range * 3.0
        
        # Adaptive Kp (parameterized weight)
        force_error_magnitude = torch.abs(fz_error)
        desired_kpz = torch.where(
            force_error_magnitude > env.fz_eps,
            torch.full_like(kpz_normalized, 0.4),
            torch.full_like(kpz_normalized, 0.7)
        )
        kpz_adaptation_error = torch.abs(kpz_normalized - desired_kpz)
        r_contact += mask_contact * torch.exp(-kpz_adaptation_error / 0.2) * w_adaptive_kpz
        
        # Contact quality
        contact_maintained = (fz_local < F_LOST)
        force_quality = torch.exp(-fz_error / env.fz_eps)
        r_contact += mask_contact * contact_maintained.float() * force_quality * 3.0
        
        # ===== RISE PHASE =====
        rise_vel = torch.clamp(ee_vel[:, 2], 0.0, 0.08)
        r_rise = mask_rise * (rise_vel / 0.04) * 2.0
        r_rise += mask_rise * (ee_pos[:, 2] > 0.08).float() * 8.0
        
        # ===== GLOBAL TERMS =====
        # Phase transition bonus
        phase_bonus = torch.zeros(env.num_envs, device=env.device)
        if hasattr(env, '_prev_phase_for_reward'):
            phase_advanced = (env.phase_ctrl.phase > env._prev_phase_for_reward).float()
            phase_bonus = phase_advanced * 15.0
            reached_contact = ((env.phase_ctrl.phase == FixedPhase.CONTACT) & 
                              (env._prev_phase_for_reward != FixedPhase.CONTACT)).float()
            phase_bonus += reached_contact * 25.0
            env._prev_phase_for_reward = env.phase_ctrl.phase.clone()
        else:
            env._prev_phase_for_reward = env.phase_ctrl.phase.clone()
        
        # Kp change penalty (parameterized)
        if hasattr(env, '_prev_kpz'):
            kpz_change = torch.abs(current_kpz - env._prev_kpz)
            kpz_change_normalized = kpz_change / (env.kp_hi[2] - env.kp_lo[2])
            p_kpz_change = mask_contact * kpz_change_normalized * w_kpz_change_penalty
            env._prev_kpz = current_kpz.clone()
        else:
            p_kpz_change = torch.zeros(env.num_envs, device=env.device)
            env._prev_kpz = current_kpz.clone()
        
        # Time penalty
        time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.3
        
        # Update grid metrics (contact-only)
        if not hasattr(env, "_grid_metrics"):
            env._grid_metrics = {
                "contact_steps": 0.0,
                "fz_sum": 0.0,
                "fz_sq_sum": 0.0,
                "err_sum": 0.0,
                "err_sq_sum": 0.0,
                "osc_sum": 0.0,
                "osc_sq_sum": 0.0,
            }
        contact_mask = mask_contact > 0.5
        if torch.any(contact_mask):
            cm = contact_mask
            env._grid_metrics["contact_steps"] += float(cm.sum().item())
            env._grid_metrics["fz_sum"] += float(fz_local[cm].sum().item())
            env._grid_metrics["fz_sq_sum"] += float((fz_local[cm] ** 2).sum().item())
            env._grid_metrics["err_sum"] += float(fz_error[cm].sum().item())
            env._grid_metrics["err_sq_sum"] += float((fz_error[cm] ** 2).sum().item())
            env._grid_metrics["osc_sum"] += float(fz_change[cm].sum().item())
            env._grid_metrics["osc_sq_sum"] += float((fz_change[cm] ** 2).sum().item())

        # Total reward
        reward = r_home + r_approach + r_descent + r_contact + r_rise + phase_bonus - p_kpz_change - time_penalty
        
        # Contact loss penalty
        contact_lost = getattr(env, '_just_lost_contact', 
                              torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        reward[contact_lost] -= 20.0
        
        terms = {
            'r_home': r_home,
            'r_approach': r_approach,
            'r_descent': r_descent,
            'r_contact': r_contact,
            'r_rise': r_rise,
            'phase_bonus': phase_bonus,
            'p_kpz_change': p_kpz_change,
            'fz_error': fz_error,
            'kpz_normalized': kpz_normalized,
        }
        
        return reward, terms
    
    # Apply patch
    reward_module._compute_polish_kpz_control_reward = patched_reward
    reward_module.compute_polish_kpz_control_reward = patched_reward
    rewards_pkg.REWARD_FUNCS["kpz"] = patched_reward
    rewards_pkg.compute_reward = patched_reward

    return patched_reward


def run_single_training(
    run_id: int,
    params: dict,
    args,
    output_dir: str,
    env,
    env_name: str,
    horizon_length: int,
    minibatch_size: int,
) -> dict:
    """Run a single training with given parameters."""
    
    print(f"\n{'='*60}")
    print(f"RUN {run_id + 1}/{args.num_runs}")
    print(f"Parameters: {params}")
    print(f"{'='*60}\n")

    metrics = {
        "run_id": run_id,
        "params": params,
        "epoch_rewards": [],
        "best_reward": float("-inf"),
        "final_reward": 0.0,
    }
    
    try:
        # Patch reward function
        patched_reward = patch_reward_function(params)

        # Import after patching
        from rl_games.torch_runner import Runner

        if env is not None:
            env._reward_fn = patched_reward
            env._grid_metrics = {
                "contact_steps": 0.0,
                "fz_sum": 0.0,
                "fz_sq_sum": 0.0,
                "err_sum": 0.0,
                "err_sq_sum": 0.0,
                "osc_sum": 0.0,
                "osc_sq_sum": 0.0,
            }

        # RL-Games config
        rl_config = {
            "params": {
                "seed": 42 + run_id,
                "algo": {
                    "name": "a2c_continuous"
                },
                "model": {
                    "name": "continuous_a2c_logstd"
                },
                "network": {
                    "name": "actor_critic",
                    "separate": False,
                    "space": {
                        "continuous": {
                            "mu_activation": "None",
                            "sigma_activation": "None",
                            "mu_init": {"name": "default"},
                            "sigma_init": {"name": "const_initializer", "val": 0.0},
                            "fixed_sigma": True,
                        }
                    },
                    "mlp": {
                        "units": [256, 128, 64],
                        "activation": "elu",
                        "initializer": {"name": "default"},
                    }
                },
                "config": {
                    "name": env_name,
                    "env_name": env_name,
                    "device": "cuda:0",
                    "device_name": "cuda:0",
                    "multi_gpu": False,
                    "ppo": True,
                    "mixed_precision": False,
                    "normalize_input": True,
                    "normalize_value": True,
                    "value_bootstrap": True,
                    "num_actors": args.num_envs,
                    "reward_shaper": {"scale_value": 1.0},
                    "normalize_advantage": True,
                    "gamma": 0.99,
                    "tau": 0.95,
                    "e_clip": 0.2,
                    "clip_value": True,
                    "entropy_coef": 0.0,
                    "critic_coef": 2.0,
                    "bounds_loss_coef": 0.0,
                    "learning_rate": 3e-4,
                    "lr_schedule": "linear",
                    "schedule_type": "legacy",
                    "kl_threshold": 0.016,
                    "mini_epochs": 4,
                    "minibatch_size": minibatch_size,
                    "horizon_length": horizon_length,
                    "max_epochs": args.epochs_per_run,
                    "score_to_win": 100000,
                    "save_best_after": 50,
                    "save_frequency": 1000,  # Don't save intermediate
                    "print_stats": True,
                    "grad_norm": 1.0,
                    "truncate_grads": True,
                    "player": {"render": False},
                }
            }
        }

        # Create runner
        runner = Runner()
        runner.load(rl_config)
        runner.reset()

        agent = runner.algo_factory.create(
            runner.algo_name,
            base_name=env_name,
            params=runner.params
        )
        agent.init_tensors()
        agent.obs = agent.env_reset()
        agent.curr_frames = agent.batch_size_envs
        agent.set_eval()
        
        for epoch in range(args.epochs_per_run):
            # Train one epoch
            agent.set_train()
            agent.train_epoch()
            
            # Get mean reward from RL-Games meters.
            if hasattr(agent, "game_rewards") and agent.game_rewards.current_size > 0:
                mean_reward = float(agent.game_rewards.get_mean()[0].item())
            else:
                mean_reward = 0.0
            
            metrics["epoch_rewards"].append(mean_reward)
            
            if mean_reward > metrics["best_reward"]:
                metrics["best_reward"] = mean_reward
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs_per_run}: reward={mean_reward:.1f}, best={metrics['best_reward']:.1f}")
        
        metrics["final_reward"] = metrics["epoch_rewards"][-1] if metrics["epoch_rewards"] else 0.0
        metrics["success"] = True
        
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  [ERROR] Run failed: {e}\n{tb}")
        metrics["success"] = False
        metrics["error"] = str(e)
        metrics["traceback"] = tb
    
    finally:
        # Extract force/oscillation metrics from training rollouts (contact-only)
        if env is not None and hasattr(env, "_grid_metrics") and env._grid_metrics["contact_steps"] > 0:
            m = env._grid_metrics
            steps = m["contact_steps"]
            fz_mean = m["fz_sum"] / steps
            fz_std = float(np.sqrt(max(m["fz_sq_sum"] / steps - fz_mean ** 2, 0.0)))
            err_mean = m["err_sum"] / steps
            err_rmse = float(np.sqrt(m["err_sq_sum"] / steps))
            osc_mean = m["osc_sum"] / steps
            osc_rms = float(np.sqrt(m["osc_sq_sum"] / steps))
            score = -args.score_error_weight * err_mean - args.score_osc_weight * osc_rms
            metrics.update({
                "contact_steps": steps,
                "fz_mean": fz_mean,
                "fz_std": fz_std,
                "force_err_mean": err_mean,
                "force_err_rmse": err_rmse,
                "fz_osc_mean": osc_mean,
                "fz_osc_rms": osc_rms,
                "score": score,
            })
        else:
            metrics.update({
                "contact_steps": 0.0,
                "score": float("-inf"),
            })
    
    return metrics


def main():
    """Main grid search loop."""
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"grid_search_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GRID SEARCH FOR REWARD HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Total runs: {args.num_runs}")
    print(f"Epochs per run: {args.epochs_per_run}")
    print(f"Environments: {args.num_envs}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    env, env_name = setup_shared_env(args)
    horizon_length = 32
    batch_size = args.num_envs * horizon_length
    minibatch_size = min(512, batch_size)
    if batch_size % minibatch_size != 0:
        # Ensure RL-Games minibatch divides batch size.
        for candidate in range(minibatch_size, 0, -1):
            if batch_size % candidate == 0:
                minibatch_size = candidate
                break
    
    # Generate parameter combinations
    all_params = generate_random_params(args.num_runs)
    
    # Save parameter grid
    with open(os.path.join(output_dir, "param_grid.json"), "w") as f:
        json.dump({"grid": PARAM_GRID, "samples": all_params}, f, indent=2)
    
    # Run all experiments
    all_results = []
    
    for run_id, params in enumerate(all_params):
        result = run_single_training(
            run_id,
            params,
            args,
            output_dir,
            env,
            env_name,
            horizon_length,
            minibatch_size,
        )
        all_results.append(result)
        
        # Save intermediate results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    
    # Analyze results
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE - ANALYSIS")
    print(f"{'='*60}\n")
    
    # Sort by score if available, otherwise best reward
    successful_runs = [r for r in all_results if r.get("success", False)]
    if successful_runs and "score" in successful_runs[0]:
        sorted_runs = sorted(successful_runs, key=lambda x: x.get("score", float("-inf")), reverse=True)
    else:
        sorted_runs = sorted(successful_runs, key=lambda x: x["best_reward"], reverse=True)
    
    print("TOP 10 PARAMETER COMBINATIONS:")
    print("-" * 60)
    for i, run in enumerate(sorted_runs[:10]):
        if "score" in run:
            print(f"\n{i+1}. Score: {run['score']:.3f}")
            print(f"   Force err mean: {run.get('force_err_mean', float('nan')):.3f}")
            print(f"   Force osc rms:  {run.get('fz_osc_rms', float('nan')):.3f}")
        print(f"   Best Reward: {run['best_reward']:.2f}")
        print(f"   Final Reward: {run['final_reward']:.2f}")
        for key, val in run["params"].items():
            print(f"   {key}: {val}")
    
    # Save summary
    summary = {
        "total_runs": args.num_runs,
        "successful_runs": len(successful_runs),
        "best_params": sorted_runs[0]["params"] if sorted_runs else None,
        "best_reward": sorted_runs[0]["best_reward"] if sorted_runs else None,
        "top_10": sorted_runs[:10],
    }
    
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n\nResults saved to: {output_dir}")
    
    if env is not None:
        env.close()

    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
