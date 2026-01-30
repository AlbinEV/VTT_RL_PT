"""Reward function for planar stiffness control (Kp_X, Kp_Y).

Goals during CONTACT:
- Track XY trajectory (low pos_err) while maintaining vertical force control (fz near target handled elsewhere).
- Reduce lateral force spikes/oscillations (fx, fy stability).
- Adapt Kp_X/Kp_Y: lower when lateral disturbances are high, moderate-high when motion is smooth, to balance compliance and accuracy.
- Penalize excessive Kp changes to avoid chattering.

Weights are conservative and can be tuned.
"""
from __future__ import annotations
import torch
from ..Trajectory_Manager import FixedPhase
from isaaclab.utils.math import quat_apply


def _compute_polish_kpxy_control_reward(env):
    device = env.device
    phase = env.phase_ctrl.phase
    mask_home     = (phase == FixedPhase.HOME).float()
    mask_approach = (phase == FixedPhase.APPROACH).float()
    mask_descent  = (phase == FixedPhase.DESCENT).float()
    mask_contact  = (phase == FixedPhase.CONTACT).float()
    mask_rise     = (phase == FixedPhase.RISE).float()

    # Basic signals
    pos_err = env._last_pos_err  # scalar XY position error
    ori_err = env._last_ori_err  # scalar orientation error
    ee_vel  = env.robot.data.body_lin_vel_w[:, env.ee_body_idx]

    # Forces in end-effector local frame
    f_ext_global = env.carpet_sensor.data.net_forces_w[:, 0]  # (N,3)
    ee_quat = env.robot.data.body_quat_w[:, env.ee_body_idx]  # (N,4)
    q_conj = torch.cat([ee_quat[:, 0:1], -ee_quat[:, 1:]], dim=1)
    f_ext_local = quat_apply(q_conj, f_ext_global)
    fx, fy, fz = f_ext_local.unbind(dim=1)

    # Current planar Kp
    kpx = env.dynamic_kp[:, 0]
    kpy = env.dynamic_kp[:, 1]
    # Normalize to [0,1] within allowed range
    kpx_n = (kpx - env.kp_lo[0]) / (env.kp_hi[0] - env.kp_lo[0] + 1e-8)
    kpy_n = (kpy - env.kp_lo[1]) / (env.kp_hi[1] - env.kp_lo[1] + 1e-8)

    # One-time notice if X/Y not effectively trainable
    if not hasattr(env, '_warned_kpxy_train'):
        if (env.kp_hi[0] - env.kp_lo[0]) < 1e-6 or (env.kp_hi[1] - env.kp_lo[1]) < 1e-6:
            print('[KPXY Reward] Warning: kp_x/ kp_y ranges are degenerate; check TRAIN_AXES or kp bounds')
        env._warned_kpxy_train = True

    # -------------------------
    # Navigation phases (light)
    # -------------------------
    r_home = mask_home * (torch.exp(-6.0 * pos_err) + torch.exp(-2.0 * ori_err))
    r_approach = mask_approach * (torch.exp(-10.0 * pos_err) + torch.exp(-3.0 * ori_err))

    # Descent support (XY stability)
    r_descent_xy = mask_descent * torch.exp(-20.0 * pos_err) * 1.5

    # ----------------------------------------
    # Contact phase: main planar control focus
    # ----------------------------------------
    # 1) XY trajectory following (higher weight here)
    r_xy_track = mask_contact * torch.exp(-25.0 * pos_err) * 3.0

    # 2) Lateral force stability: penalize |Δfx|, |Δfy|
    if hasattr(env, '_prev_fx_fy'):
        dfx = torch.abs(fx - env._prev_fx_fy[:, 0])
        dfy = torch.abs(fy - env._prev_fx_fy[:, 1])
        env._prev_fx_fy = torch.stack([fx, fy], dim=1)
        r_lat_stability = mask_contact * (torch.exp(-(dfx + dfy) / 3.0)) * 2.5
    else:
        env._prev_fx_fy = torch.stack([fx, fy], dim=1)
        r_lat_stability = torch.zeros_like(fx)

    # 3) Optimal Kp range in contact: prefer mid-high stiffness for accuracy but not max
    opt_center = 0.6
    opt_width = 0.25
    r_kp_opt = mask_contact * (
        torch.exp(-((kpx_n - opt_center) / opt_width) ** 2) +
        torch.exp(-((kpy_n - opt_center) / opt_width) ** 2)
    ) * 1.5

    # 4) Adaptive kp rule: if lateral disturbance is large -> prefer lower stiffness; if small -> higher
    lat_force = torch.sqrt(fx * fx + fy * fy)
    # Estimate disturbance via short EMA on |Δf|
    if hasattr(env, '_ema_df_lat'):
        ema = env._ema_df_lat
        decay = 0.9
        env._ema_df_lat = decay * ema + (1 - decay) * (torch.abs(fx) + torch.abs(fy))
    else:
        env._ema_df_lat = torch.abs(fx) + torch.abs(fy)
    df_est = env._ema_df_lat
    desired_kp = torch.where(df_est > 5.0, 0.35, 0.7)  # heuristic thresholds in N
    r_kp_adapt = mask_contact * (
        torch.exp(-torch.abs(kpx_n - desired_kp) / 0.2) +
        torch.exp(-torch.abs(kpy_n - desired_kp) / 0.2)
    ) * 3.0

    # 5) Smoothness: penalize large kp changes in contact
    if hasattr(env, '_prev_kpxy'):
        dkpx = torch.abs(kpx - env._prev_kpxy[:, 0]) / (env.kp_hi[0] - env.kp_lo[0] + 1e-8)
        dkpy = torch.abs(kpy - env._prev_kpxy[:, 1]) / (env.kp_hi[1] - env.kp_lo[1] + 1e-8)
        p_kp_change = mask_contact * (dkpx + dkpy) * 1.0
        env._prev_kpxy = torch.stack([kpx, kpy], dim=1)
    else:
        p_kp_change = torch.zeros_like(kpx)
        env._prev_kpxy = torch.stack([kpx, kpy], dim=1)

    # Retain small term for vertical contact quality to prevent degenerate lateral behavior
    # Use fz deviation from target if available
    if hasattr(env, 'fz_target') and hasattr(env, 'fz_eps'):
        fz_error = torch.abs(fz - env.fz_target)
        r_fz_quality = mask_contact * torch.exp(-fz_error / (env.fz_eps + 1e-6)) * 1.5
    else:
        r_fz_quality = torch.zeros_like(kpx)

    # Rise phase light reward
    r_rise = mask_rise * 1.0

    # Time/energy penalties (light)
    time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.2
    if hasattr(env, 'tau2'):
        tau_norm = torch.linalg.vector_norm(env.tau2, dim=1)
        p_energy = tau_norm / (env.torque_limit.max() + 1e-6) * 0.1
    else:
        p_energy = torch.zeros_like(time_penalty)

    reward = (
        r_home + r_approach + r_descent_xy +
        r_xy_track + r_lat_stability + r_kp_opt + r_kp_adapt + r_fz_quality +
        r_rise - p_kp_change - time_penalty - p_energy
    )

    terms = {
        'r_home': r_home,
        'r_approach': r_approach,
        'r_descent_xy': r_descent_xy,
        'r_xy_track': r_xy_track,
        'r_lat_stability': r_lat_stability,
        'r_kp_opt': r_kp_opt,
        'r_kp_adapt': r_kp_adapt,
        'p_kp_change': p_kp_change,
        'r_fz_quality': r_fz_quality,
        'time_penalty': time_penalty,
        'p_energy': p_energy,
        'kpx_n': kpx_n,
        'kpy_n': kpy_n,
        'df_est': df_est,
        'lat_force': lat_force,
        'phase': phase.float(),
    }
    return reward, terms

# Public alias without underscore for external imports
compute_polish_kpxy_control_reward = _compute_polish_kpxy_control_reward

__all__ = ['_compute_polish_kpxy_control_reward', 'compute_polish_kpxy_control_reward']
