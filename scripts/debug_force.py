#!/usr/bin/env python3
"""Debug force control to understand why Fz doesn't reach -20N"""
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
    
    print(f"\n{'='*60}")
    print(f"DEBUG FORCE CONTROL")
    print(f"{'='*60}")
    print(f"fz_target: {env.fz_target} N")
    print(f"kp_hi: {env.kp_hi.cpu().numpy()}")
    print(f"kp_lo: {env.kp_lo.cpu().numpy()}")
    print(f"init_kp: {env.init_kp.cpu().numpy()}")
    print(f"\nOSC Config:")
    print(f"  motion_stiffness_task: {env.cfg.ctrl.motion_stiffness_task}")
    print(f"  contact_wrench_stiffness_task: {env.cfg.ctrl.contact_wrench_stiffness_task}")
    print(f"  motion_damping_ratio_task: {env.cfg.ctrl.motion_damping_ratio_task}")
    print(f"  impedance_mode: {env.cfg.ctrl.impedance_mode}")
    print(f"{'='*60}\n")
    
    obs, info = env.reset()
    
    step = 0
    contact_forces = []
    
    print("Running episode to analyze force behavior...")
    print(f"{'Step':>5} {'Phase':>5} {'Fz':>8} {'Fz_EMA':>8} {'Kp_z':>8} {'EE_z':>8} {'Wpt':>4}")
    print("-" * 60)
    
    done = False
    while not done:
        action = torch.zeros((1, env.action_space.shape[0]), device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        
        step += 1
        phase = env.phase_ctrl.phase[0].item() if hasattr(env, 'phase_ctrl') else 0
        
        fz_raw = env.cube_sensor.data.net_forces_w[0, 0, 2].item()
        fz_ema = env._fz_ema[0].item()
        kp_z = env.dynamic_kp[0, 2].item()
        ee_z = env.robot.data.body_pos_w[0, env.ee_body_idx, 2].item()
        wpt = env.wpt_idx[0].item()
        
        if phase >= 3:
            contact_forces.append(fz_ema)
        
        if step % 50 == 0 or (phase >= 3 and step % 10 == 0):
            print(f"{step:5d} {phase:5d} {fz_raw:8.2f} {fz_ema:8.2f} {kp_z:8.1f} {ee_z:8.4f} {wpt:4d}")
        
        done = terminated.any().item() or truncated.any().item()
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    if contact_forces:
        print(f"Contact force stats (phase >= 3):")
        print(f"  Mean: {np.mean(contact_forces):.2f} N")
        print(f"  Std:  {np.std(contact_forces):.2f} N")
        print(f"  Min:  {np.min(contact_forces):.2f} N")
        print(f"  Max:  {np.max(contact_forces):.2f} N")
        print(f"  Target: {env.fz_target} N")
        print(f"  Error: {abs(np.mean(contact_forces) - env.fz_target):.2f} N")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == "__main__":
    main()
