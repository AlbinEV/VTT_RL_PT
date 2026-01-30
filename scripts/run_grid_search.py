#!/usr/bin/env python3
"""
Simplified Grid Search for Reward Hyperparameters.
Runs 50 short training experiments with different reward parameter combinations.

This script modifies reward parameters via environment variables and runs
separate training processes for each configuration.

Usage:
    python scripts/run_grid_search.py --num_runs 50 --epochs_per_run 100
"""
import _path_setup

import argparse
import os
import sys
import json
import random
import subprocess
import time
import glob
import numpy as np
from datetime import datetime
from pathlib import Path

# External drive for data storage
EXTERNAL_DRIVE = str(_path_setup.DATA_ROOT)
REPO_DIR = Path(__file__).resolve().parents[1]

# Base parameters (fixed during contact-focused grid search)
BASE_PARAMS = {
    "W_FORCE_TRACKING": 6.0,
    "FORCE_ERROR_SCALE": 2.0,
    "W_ADAPTIVE_KPZ": 10.0,
    "OPTIMAL_KPZ_CENTER": 0.3,
    "W_KPZ_CHANGE_PENALTY": 6.0,
    "W_CONTACT_BONUS": 30.0,
}

# Parameter grid (contact-focused)
PARAM_GRID = {
    "W_CONTACT_IMPACT_PENALTY": [4.0, 6.0, 8.0, 10.0],
    "CONTACT_RAMP_STEPS": [20, 30, 40, 50],
    "IMPACT_FORCE_MARGIN": [1.0, 2.0],
    "IMPACT_FORCE_SCALE": [3.0, 5.0],
    "IMPACT_DF_SCALE": [3.0, 5.0],
    "W_CONTACT_ZVEL_PENALTY": [2.0, 4.0, 6.0],
    "CONTACT_ZVEL_LIMIT": [0.01, 0.02, 0.03],
    "CONTACT_ZVEL_SCALE": [0.03, 0.05],
    "W_CONTACT_SOFT_BONUS": [4.0, 6.0, 8.0],
    "SOFT_FORCE_BAND": [3.0, 4.0, 5.0],
    "SOFT_DF_BAND": [4.0, 5.0, 6.0],
    "W_SOFT_KPZ_BONUS": [1.0, 2.0, 3.0],
    "SOFT_KPZ_TARGET": [0.30, 0.35, 0.40],
    "SOFT_KPZ_SCALE": [0.15, 0.20, 0.25],
    # Damping-specific terms (KZ+DZ reward)
    "W_SOFT_DZ_BONUS": [1.0, 2.0, 3.0],
    "SOFT_DZ_TARGET": [0.8, 0.9, 1.0],
    "SOFT_DZ_SCALE": [0.2, 0.25, 0.3],
    "W_DZ_DF_PENALTY": [1.0, 2.0, 3.0],
    "DZ_DF_SCALE": [4.0, 6.0, 8.0],
    "DZ_OSC_TARGET": [0.8, 0.9, 1.0],
    "DZ_OSC_GAIN": [1.0, 2.0, 3.0],
    "W_DZ_STABLE_PENALTY": [1.0, 2.0, 3.0],
    "DZ_STABLE_MIN": [0.5, 0.6, 0.7],
    "DZ_STABLE_MAX": [1.2, 1.4, 1.6],
    "DZ_STABLE_SCALE": [0.15, 0.2, 0.3],
}

def generate_random_params(num_samples: int) -> list:
    """Generate random parameter combinations."""
    params_list = []
    for _ in range(num_samples):
        params = {key: random.choice(values) for key, values in PARAM_GRID.items()}
        params_list.append(params)
    return params_list


def _find_latest_run_dir(exp_name: str) -> str | None:
    runs_root = os.path.join(EXTERNAL_DRIVE, "runs")
    candidates = [p for p in glob.glob(os.path.join(runs_root, f"{exp_name}_*")) if os.path.isdir(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _find_latest_checkpoint(run_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(run_dir, "**", "nn", "*.pth"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _find_latest_traj_dir(output_dir: str) -> str | None:
    candidates = [p for p in glob.glob(os.path.join(output_dir, "traj_*")) if os.path.isdir(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _compute_contact_metrics(
    traj_dir: str,
    contact_threshold: float,
    contact_window: int,
    contact_ratio_weight: float,
) -> dict:
    traj_path = os.path.join(traj_dir, "trajectories.json")
    if not os.path.isfile(traj_path):
        return {}
    with open(traj_path, "r") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    min_fz_list = []
    max_df_list = []
    mean_df_list = []
    contact_ratio_list = []
    for ep in episodes:
        fz = np.array(ep.get("contact_forces", []), dtype=float)
        if fz.size == 0:
            continue
        contact_ratio = float(np.mean(fz < contact_threshold))
        idxs = np.where(fz < contact_threshold)[0]
        if idxs.size == 0:
            continue
        start = int(idxs[0])
        seg = fz[start:start + contact_window]
        if seg.size == 0:
            continue
        df = np.abs(np.diff(seg, prepend=seg[0]))
        min_fz_list.append(float(seg.min()))
        max_df_list.append(float(df.max()))
        mean_df_list.append(float(df.mean()))
        contact_ratio_list.append(contact_ratio)
    if not min_fz_list or not max_df_list:
        return {}
    min_fz_mean = float(np.mean(min_fz_list))
    max_df_mean = float(np.mean(max_df_list))
    mean_df_mean = float(np.mean(mean_df_list)) if mean_df_list else 0.0
    contact_ratio_mean = float(np.mean(contact_ratio_list)) if contact_ratio_list else 0.0
    contact_score = min_fz_mean - 0.5 * max_df_mean + contact_ratio_weight * contact_ratio_mean
    return {
        "contact_min_fz_mean": min_fz_mean,
        "contact_max_df_mean": max_df_mean,
        "contact_mean_df_mean": mean_df_mean,
        "contact_ratio_mean": contact_ratio_mean,
        "contact_score": contact_score,
        "contact_episodes": len(min_fz_list),
    }


def run_training(
    run_id: int,
    params: dict,
    epochs: int,
    num_envs: int,
    output_dir: str,
    eval_episodes: int,
    contact_threshold: float,
    contact_window: int,
    contact_ratio_weight: float,
    use_kz_dz: bool,
) -> dict:
    """Run a single training with specified parameters."""
    
    print(f"\n{'='*60}")
    print(f"RUN {run_id + 1}: {params}")
    print(f"{'='*60}")
    
    # Set environment variables for reward parameters
    env = os.environ.copy()
    for key, value in BASE_PARAMS.items():
        env[key] = str(value)
    for key, value in params.items():
        env[key] = str(value)
    
    # Experiment name
    exp_name = f"grid_{run_id:03d}"
    
    # Build command - use Python directly instead of isaaclab.sh
    if use_kz_dz:
        script_path = str(REPO_DIR / "scripts" / "train_kz_dz.py")
    else:
        script_path = str(REPO_DIR / "scripts" / "train_kpz_only.py")
    cmd = [
        sys.executable,  # Use current Python interpreter
        script_path,
        "--num_envs", str(num_envs),
        "--max_epochs", str(epochs),
        "--experiment_name", exp_name,
    ]
    
    log_file = os.path.join(output_dir, f"run_{run_id:03d}.log")
    
    start_time = time.time()
    
    try:
        with open(log_file, "w") as f:
            f.write(f"Parameters: {json.dumps(params)}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()
            
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=3600,  # 1 hour timeout
                cwd=str(REPO_DIR)
            )
        
        elapsed = time.time() - start_time
        
        # Try to extract final reward from log
        final_reward = extract_final_reward(log_file)
        
        result_payload = {
            "run_id": run_id,
            "params": params,
            "success": result.returncode == 0,
            "elapsed_time": elapsed,
            "final_reward": final_reward,
            "log_file": log_file,
        }

        if result_payload["success"]:
            run_dir = _find_latest_run_dir(exp_name)
            checkpoint = _find_latest_checkpoint(run_dir) if run_dir else None
            if checkpoint:
                eval_log = os.path.join(output_dir, f"eval_{run_id:03d}.log")
                eval_cmd = [
                    sys.executable,
                    str(REPO_DIR / "scripts" / "extract_trajectories.py"),
                    "--checkpoint", checkpoint,
                    "--num_episodes", str(eval_episodes),
                    "--num_envs", "1",
                    "--output", output_dir,
                    "--headless",
                ]
                with open(eval_log, "w") as ef:
                    subprocess.run(
                        eval_cmd,
                        env=env,
                        stdout=ef,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                        cwd=str(REPO_DIR),
                    )
                traj_dir = _find_latest_traj_dir(output_dir)
                metrics = _compute_contact_metrics(
                    traj_dir,
                    contact_threshold,
                    contact_window,
                    contact_ratio_weight,
                ) if traj_dir else {}
                result_payload.update({
                    "checkpoint": checkpoint,
                    "traj_dir": traj_dir,
                })
                result_payload.update(metrics)

        return result_payload
        
    except subprocess.TimeoutExpired:
        return {
            "run_id": run_id,
            "params": params,
            "success": False,
            "error": "timeout",
            "log_file": log_file,
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "params": params,
            "success": False,
            "error": str(e),
        }


def extract_final_reward(log_file: str) -> float:
    """Extract final reward from training log."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Look for reward lines from the end
        for line in reversed(lines):
            if "mean_reward" in line.lower() or "reward" in line.lower():
                # Try to extract number
                import re
                numbers = re.findall(r"[-+]?\d*\.?\d+", line)
                if numbers:
                    return float(numbers[-1])
        
        return None
    except:
        return None


def main():
    parser = argparse.ArgumentParser(description="Grid Search for Reward Parameters")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of runs")
    parser.add_argument("--epochs_per_run", type=int, default=100, help="Epochs per run")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of envs")
    parser.add_argument("--output_dir", type=str, default=os.path.join(EXTERNAL_DRIVE, "grid_search"))
    parser.add_argument("--eval_episodes", type=int, default=2, help="Episodes to evaluate per run")
    parser.add_argument("--contact_threshold", type=float, default=-2.0, help="Contact detection threshold")
    parser.add_argument("--contact_window", type=int, default=20, help="Contact window length")
    parser.add_argument("--contact_ratio_weight", type=float, default=20.0, help="Weight for contact ratio term")
    parser.add_argument("--use_kz_dz", action="store_true", help="Use KZ+DZ training script")
    args = parser.parse_args()
    
    # Create output directory on external drive
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"grid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("GRID SEARCH FOR REWARD HYPERPARAMETERS")
    print(f"{'='*60}")
    print(f"Runs: {args.num_runs}")
    print(f"Epochs/run: {args.epochs_per_run}")
    print(f"Output: {output_dir}")
    print(f"KZ+DZ mode: {args.use_kz_dz}")
    
    # Generate parameters
    all_params = generate_random_params(args.num_runs)
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "grid": PARAM_GRID,
            "base_params": BASE_PARAMS,
            "samples": all_params,
            "epochs_per_run": args.epochs_per_run,
            "num_envs": args.num_envs,
            "eval_episodes": args.eval_episodes,
            "contact_threshold": args.contact_threshold,
            "contact_window": args.contact_window,
            "contact_ratio_weight": args.contact_ratio_weight,
            "use_kz_dz": args.use_kz_dz,
        }, f, indent=2)
    
    # Run experiments
    results = []
    for i, params in enumerate(all_params):
        result = run_training(
            i,
            params,
            args.epochs_per_run,
            args.num_envs,
            output_dir,
            args.eval_episodes,
            args.contact_threshold,
            args.contact_window,
            args.contact_ratio_weight,
            args.use_kz_dz,
        )
        results.append(result)
        
        # Save intermediate results
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Print progress
        successful = sum(1 for r in results if r.get("success"))
        print(f"\nProgress: {i+1}/{args.num_runs} ({successful} successful)")
    
    # Final analysis
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    
    successful_results = [r for r in results if r.get("success")]
    with_contact = [r for r in successful_results if r.get("contact_score") is not None]

    if with_contact:
        sorted_results = sorted(with_contact, key=lambda x: x.get("contact_score", -1e9), reverse=True)
        print("\nTOP 10 CONFIGURATIONS (CONTACT SCORE):")
        for i, r in enumerate(sorted_results[:10]):
            print(f"\n{i+1}. Contact score: {r.get('contact_score'):.3f}")
            print(
                f"   min_fz_mean: {r.get('contact_min_fz_mean'):.2f}  "
                f"max_df_mean: {r.get('contact_max_df_mean'):.2f}  "
                f"mean_df: {r.get('contact_mean_df_mean'):.2f}  "
                f"contact_ratio: {r.get('contact_ratio_mean'):.2f}"
            )
            for k, v in r["params"].items():
                print(f"   {k}: {v}")
    else:
        successful_results = [r for r in results if r.get("success") and r.get("final_reward") is not None]
        if successful_results:
            sorted_results = sorted(successful_results, key=lambda x: x.get("final_reward", 0), reverse=True)
            print("\nTOP 10 CONFIGURATIONS (FINAL REWARD):")
            for i, r in enumerate(sorted_results[:10]):
                print(f"\n{i+1}. Reward: {r.get('final_reward', 'N/A')}")
                for k, v in r["params"].items():
                    print(f"   {k}: {v}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
