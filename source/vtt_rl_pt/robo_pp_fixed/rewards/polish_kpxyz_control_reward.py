"""Reward for simultaneous control of Kp_X, Kp_Y, Kp_Z.

Design:
- XY: track trajectory, minimize pos_err, stabilize lateral forces; prefer mid-high Kp when smooth, lower with disturbances.
- Z: track vertical contact force; prefer mid Kpz, adaptive lower/higher depending on force error and oscillation.
- Smoothness: penalize large changes in Kp across steps.
"""
from __future__ import annotations
import torch
from ..Trajectory_Manager import FixedPhase
from isaaclab.utils.math import quat_apply


def _compute_polish_kpxyz_control_reward(env):
    device = env.device
    phase = env.phase_ctrl.phase
    mask_home     = (phase == FixedPhase.HOME).float()
    mask_approach = (phase == FixedPhase.APPROACH).float()
    mask_descent  = (phase == FixedPhase.DESCENT).float()
    mask_contact  = (phase == FixedPhase.CONTACT).float()
    mask_rise     = (phase == FixedPhase.RISE).float()

    pos_err = env._last_pos_err
    ori_err = env._last_ori_err
    ee_vel  = env.robot.data.body_lin_vel_w[:, env.ee_body_idx]

    # Forces in ee-local
    f_ext_w = env.carpet_sensor.data.net_forces_w[:, 0]
    ee_quat = env.robot.data.body_quat_w[:, env.ee_body_idx]
    q_conj = torch.cat([ee_quat[:, :1], -ee_quat[:, 1:]], dim=1)
    f_ext_b = quat_apply(q_conj, f_ext_w)
    fx, fy, fz = f_ext_b.unbind(dim=1)

    # Current Kp
    kpx, kpy, kpz = env.dynamic_kp[:, 0], env.dynamic_kp[:, 1], env.dynamic_kp[:, 2]
    kpx_n = (kpx - env.kp_lo[0]) / (env.kp_hi[0] - env.kp_lo[0] + 1e-8)
    kpy_n = (kpy - env.kp_lo[1]) / (env.kp_hi[1] - env.kp_lo[1] + 1e-8)
    kpz_n = (kpz - env.kp_lo[2]) / (env.kp_hi[2] - env.kp_lo[2] + 1e-8)

    # Navigation phases
    r_home = mask_home * (torch.exp(-6.0 * pos_err) + torch.exp(-2.0 * ori_err))
    r_approach = mask_approach * (torch.exp(-10.0 * pos_err) + torch.exp(-3.0 * ori_err))
    r_descent_xy = mask_descent * torch.exp(-20.0 * pos_err) * 1.5
    # Encourage moving downward during DESCENT (vz < 0), capped for safety
    vz = env.robot.data.body_lin_vel_w[:, env.ee_body_idx, 2]
    r_descent_z = mask_descent * torch.clamp(torch.relu(-vz), 0.0, 0.2) * 5.0
    # Give a mild pre-contact force shaping to approach touch smoothly (target around -1N)
    precontact_target = -1.0
    r_precontact_force = mask_descent * torch.exp(-torch.abs(fz - precontact_target) / 2.0) * 2.0

    # Contact: XY tracking
    r_xy_track = mask_contact * torch.exp(-25.0 * pos_err) * 3.0
    # Lateral stability
    if hasattr(env, '_prev_fx_fy'):
        dfx = torch.abs(fx - env._prev_fx_fy[:, 0])
        dfy = torch.abs(fy - env._prev_fx_fy[:, 1])
        env._prev_fx_fy = torch.stack([fx, fy], dim=1)
        r_lat_stability = mask_contact * torch.exp(-(dfx + dfy) / 3.0) * 2.5
    else:
        env._prev_fx_fy = torch.stack([fx, fy], dim=1)
        r_lat_stability = torch.zeros_like(fx)

    # XY Kp preference (mid-high, not max)
    opt_xy = 0.6
    width_xy = 0.25
    r_kp_xy_opt = mask_contact * (
        torch.exp(-((kpx_n - opt_xy) / width_xy) ** 2) +
        torch.exp(-((kpy_n - opt_xy) / width_xy) ** 2)
    ) * 1.5

    # XY adaptive rule by disturbance
    if hasattr(env, '_ema_df_lat'):
        decay = 0.9
        env._ema_df_lat = decay * env._ema_df_lat + (1 - decay) * (torch.abs(fx) + torch.abs(fy))
    else:
        env._ema_df_lat = torch.abs(fx) + torch.abs(fy)
    df_lat = env._ema_df_lat
    desired_kp_xy = torch.where(df_lat > 5.0, 0.35, 0.7)
    r_kp_xy_adapt = mask_contact * (
        torch.exp(-torch.abs(kpx_n - desired_kp_xy) / 0.2) +
        torch.exp(-torch.abs(kpy_n - desired_kp_xy) / 0.2)
    ) * 3.0

    # Z: force tracking and stability
    if hasattr(env, 'fz_target') and hasattr(env, 'fz_eps'):
        fz_err = torch.abs(fz - env.fz_target)
        r_fz_track = mask_contact * torch.exp(-fz_err / (env.fz_eps * 3.0 + 1e-6)) * 6.0
    else:
        fz_err = torch.zeros_like(fz)
        r_fz_track = torch.zeros_like(fz)

    if hasattr(env, '_prev_fz_local'):
        dfz = torch.abs(fz - env._prev_fz_local)
        env._prev_fz_local = fz.clone()
        r_fz_stability = mask_contact * torch.exp(-dfz / 2.0) * 3.0
    else:
        env._prev_fz_local = fz.clone()
        r_fz_stability = torch.zeros_like(fz)

    # KpZ preference & adaptation
    opt_z = 0.55
    width_z = 0.3
    r_kp_z_opt = mask_contact * torch.exp(-((kpz_n - opt_z) / width_z) ** 2) * 2.0
    desired_kpz = torch.where(fz_err > (getattr(env, 'fz_eps', 1.0)), 0.4, 0.7)
    r_kp_z_adapt = mask_contact * torch.exp(-torch.abs(kpz_n - desired_kpz) / 0.2) * 3.0

    # Smoothness penalties on Kp changes
    if hasattr(env, '_prev_kpxyz'):
        dkpx = torch.abs(kpx - env._prev_kpxyz[:, 0]) / (env.kp_hi[0] - env.kp_lo[0] + 1e-8)
        dkpy = torch.abs(kpy - env._prev_kpxyz[:, 1]) / (env.kp_hi[1] - env.kp_lo[1] + 1e-8)
        dkpz = torch.abs(kpz - env._prev_kpxyz[:, 2]) / (env.kp_hi[2] - env.kp_lo[2] + 1e-8)
        p_kp_change = mask_contact * (dkpx + dkpy + dkpz) * 1.0
        env._prev_kpxyz = torch.stack([kpx, kpy, kpz], dim=1)
    else:
        p_kp_change = torch.zeros_like(kpx)
        env._prev_kpxyz = torch.stack([kpx, kpy, kpz], dim=1)

    # Rise phase: encourage detachment and moving up to finish
    r_rise = mask_rise * 2.0

    # Global penalties
    time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.05
    if hasattr(env, 'tau2'):
        tau_norm = torch.linalg.vector_norm(env.tau2, dim=1)
        p_energy = tau_norm / (env.torque_limit.max() + 1e-6) * 0.1
    else:
        p_energy = torch.zeros_like(time_penalty)

    reward = (
    r_home + r_approach + r_descent_xy + r_descent_z + r_precontact_force +
        r_xy_track + r_lat_stability + r_kp_xy_opt + r_kp_xy_adapt +
        r_fz_track + r_fz_stability + r_kp_z_opt + r_kp_z_adapt +
        r_rise - p_kp_change - time_penalty - p_energy
    )

    terms = {
        'r_home': r_home,
        'r_approach': r_approach,
    'r_descent_xy': r_descent_xy,
    'r_descent_z': r_descent_z,
    'r_precontact_force': r_precontact_force,
        'r_xy_track': r_xy_track,
        'r_lat_stability': r_lat_stability,
        'r_kp_xy_opt': r_kp_xy_opt,
        'r_kp_xy_adapt': r_kp_xy_adapt,
        'r_fz_track': r_fz_track,
        'r_fz_stability': r_fz_stability,
        'r_kp_z_opt': r_kp_z_opt,
        'r_kp_z_adapt': r_kp_z_adapt,
        'p_kp_change': p_kp_change,
        'time_penalty': time_penalty,
        'p_energy': p_energy,
        'kpx_n': kpx_n,
        'kpy_n': kpy_n,
        'kpz_n': kpz_n,
    }
    return reward, terms

# Public alias without underscore for external imports
compute_polish_kpxyz_control_reward = _compute_polish_kpxyz_control_reward

__all__ = ['_compute_polish_kpxyz_control_reward', 'compute_polish_kpxyz_control_reward']
