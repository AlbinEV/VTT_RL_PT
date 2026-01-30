#!/usr/bin/env python3
import _path_setup
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from robo_pp_fixed.Polish_Env_OSC import PolishEnv, PolishEnvCfg

env_cfg = PolishEnvCfg()
env_cfg.scene.num_envs = 1
env = PolishEnv(cfg=env_cfg, render_mode=None)

with open("/tmp/osc_check.txt", "w") as f:
    f.write(f"fz_target: {env.fz_target}\n")
    f.write(f"_mask_free: {env._mask_free}\n")
    f.write(f"_mask_touch: {env._mask_touch}\n")
    f.write(f"contact_wrench_stiffness_task: {env.osc.impl.cfg.contact_wrench_stiffness_task}\n")
    f.write(f"motion_stiffness_task: {env.osc.impl.cfg.motion_stiffness_task}\n")
    
    obs, _ = env.reset()
    for step in range(100):
        action = torch.zeros((1, env.action_space.shape[0]), device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        phase = env.phase_ctrl.phase[0].item()
        fz = env._fz_ema[0].item()
        mc = env.osc.impl.cfg.motion_control_axes_task
        fc = env.osc.impl.cfg.contact_wrench_control_axes_task
        
        if step % 10 == 0:
            f.write(f"Step {step}: phase={phase}, Fz={fz:.2f}, motion_z={mc[2]}, force_z={fc[2]}\n")
        
        if terminated.any() or truncated.any():
            break

env.close()
print("Done - check /tmp/osc_check.txt")
