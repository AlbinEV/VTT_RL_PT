#!/usr/bin/env python3
"""Debug force control - check actual OSC settings during contact"""
import _path_setup

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug Force Control")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np


from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg


def main():
    env_cfg = PolishEnvCfg()
    env_cfg.scene.num_envs = 1
    env = PolishEnv(cfg=env_cfg, render_mode=None)
    
    print(f"\n{'='*70}")
    print(f"DEBUG FORCE CONTROL - DETAILED")
    print(f"{'='*70}")
    print(f"fz_target: {env.fz_target} N")
    print(f"\nInitial OSC Config:")
    print(f"  motion_control_axes_task:         {env.osc.impl.cfg.motion_control_axes_task}")
    print(f"  contact_wrench_control_axes_task: {env.osc.impl.cfg.contact_wrench_control_axes_task}")
    print(f"  contact_wrench_stiffness_task:    {env.osc.impl.cfg.contact_wrench_stiffness_task}")
    print(f"{'='*70}\n")
    
    obs, info = env.reset()
    
    step = 0
    printed_contact = False
    
    print("Running episode...")
    print(f"{'Step':>5} {'Ph':>3} {'Fz':>8} {'mot_z':>5} {'frc_z':>5} {'Kf_z':>8} {'Kp_z':>8}")
    print("-" * 60)
    
    done = False
    while not done and step < 150:
        action = torch.zeros((1, env.action_space.shape[0]), device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step += 1
        phase = env.phase_ctrl.phase[0].item() if hasattr(env, 'phase_ctrl') else 0
        
        fz_ema = env._fz_ema[0].item()
        
        # Get current OSC config
        mc = env.osc.impl.cfg.motion_control_axes_task
        fc = env.osc.impl.cfg.contact_wrench_control_axes_task
        kf = env.osc.impl.cfg.contact_wrench_stiffness_task
        kp_z = env.dynamic_kp[0, 2].item()
        
        if phase >= 1 and not printed_contact:
            print(f"\n*** CONTACT DETECTED at step {step} ***")
            print(f"  motion_control_axes_task:         {mc}")
            print(f"  contact_wrench_control_axes_task: {fc}")
            print(f"  contact_wrench_stiffness_task:    {kf}")
            print(f"  dynamic_kp[z]:                    {kp_z}")
            print()
            printed_contact = True
        
        if step % 10 == 0:
            print(f"{step:5d} {phase:3d} {fz_ema:8.2f} {mc[2]:5d} {fc[2]:5d} {kf[2]:8.1f} {kp_z:8.1f}")
        
        done = terminated.any().item() or truncated.any().item()
    
    print(f"\n{'='*70}")
    print(f"ANALYSIS:")
    print(f"{'='*70}")
    print(f"When in contact (phase >= 1):")
    print(f"  motion_control_axes_task[2] (Z) = {mc[2]}  (should be 0 for force control)")
    print(f"  contact_wrench_control_axes_task[2] (Z) = {fc[2]}  (should be 1 for force control)")
    print(f"  contact_wrench_stiffness_task[2] = {kf[2]} N/m (force gain)")
    print(f"\nIf motion_z=0 and force_z=1, then Z is in PURE force control.")
    print(f"The force gain {kf[2]} determines how aggressively it tracks {env.fz_target}N")
    print(f"{'='*70}\n")
    
    env.close()


if __name__ == "__main__":
    main()
