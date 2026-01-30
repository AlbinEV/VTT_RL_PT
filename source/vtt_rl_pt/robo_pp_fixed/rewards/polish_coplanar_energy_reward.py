# File: polish_coplanar_energy_reward.py
"""
Module: polish_coplanar_energy_reward.py
Brief: Compute the reinforcement learning reward for the polishing task.
This reward integrates multiple components:
 1. Contact establishment and descent speed bonus
 2. Trajectory following in position and orientation
 3. Force control adherence
 4. Deformation/work done reward
 5. Impedance optimization for optimal stiffness
 6. Coplanarity of the end-effector relative to the trajectory (roll/pitch)
 7. Energy penalty based on torque usage

This file should be imported and called from the main PolishEnv implementation.
"""

import torch
from ..cfg.config import F_TOUCH, F_LOST
from isaaclab.utils.math import quat_apply  # efficient quaternion-vector rotate
from ..Trajectory_Manager import FixedPhase


def _compute_polish_coplanar_energy_reward(env) -> torch.Tensor:
    """
    Compute the polishing reward for all environments in a batch.
    :param env: instance of PolishEnv
    :return: reward tensor of shape (num_envs,)
    """
    # Phase masks
    mask_descent = (env.phase_ctrl.phase == FixedPhase.DESCENT).float()
    mask_follow  = (env.phase_ctrl.phase == FixedPhase.CONTACT).float()

    # External forces in world frame
    f_ext_global = env.carpet_sensor.data.net_forces_w[:, 0]  # (N,3)
    ee_quat      = env.robot.data.body_quat_w[:, env.ee_body_idx]  # (N,4)

    # Rotate world force into local EE frame
    # Invert quaternion for world->local
    q_conj       = torch.cat([ee_quat[:, 0:1], -ee_quat[:, 1:]], dim=1)
    f_ext_local  = quat_apply(q_conj, f_ext_global)  # (N,3)
    fx_local, fy_local, fz_local = f_ext_local.unbind(dim=1)

    # Position/orientation errors from env (already stored)
    pos_err = env._last_pos_err  # XY-plane position error
    ori_err = env._last_ori_err  # orientation error

    # 1. CONTACT ESTABLISHMENT REWARD
    descent_speed              = torch.clamp(-env.robot.data.body_lin_vel_w[:, env.ee_body_idx, 2], 0.0, 0.1)
    r_contact_speed           = mask_descent * (descent_speed / 0.05)
    contact_established_bonus = mask_descent * (fz_local < F_TOUCH).float() * 10.0

    # 2. TRAJECTORY FOLLOWING (Phase 1)
    r_xy  = mask_follow * torch.exp(-15.0 * pos_err)
    r_ori = mask_follow * torch.exp(-3.0  * ori_err)

    # 3. FORCE CONTROL (Phase 1)
    ee_vel_xy_norm    = torch.norm(env.robot.data.body_lin_vel_w[:, env.ee_body_idx, :2], dim=1)
    adaptive_fz_target = env.fz_target - 2.0 * torch.clamp(ee_vel_xy_norm / 0.05, 0.0, 1.0)
    fz_error          = torch.abs(fz_local - adaptive_fz_target)
    r_fz_control      = mask_follow * torch.exp(-fz_error / env.fz_eps)

    # 4. DEFORMATION/WORK DONE REWARD
    lateral_force_norm      = torch.sqrt(fx_local**2 + fy_local**2)
    r_deformation           = mask_follow * torch.clamp(lateral_force_norm / 2.0, 0.0, 1.0)
    in_contact_continuity   = (fz_local < F_LOST).float()
    r_contact_continuity    = mask_follow * in_contact_continuity

    # 5. IMPEDANCE CONTROL OPTIMIZATION
    kp_z_axis       = env.dynamic_kp[:, 2]
    kp_xy_axes_mean = env.dynamic_kp[:, :2].mean(dim=1)
    optimal_kp_z_contact  = env.kp_lo[2] + 0.2 * (env.kp_hi[2] - env.kp_lo[2])
    optimal_kp_xy_contact = env.kp_lo[0] + 0.8 * (env.kp_hi[0] - env.kp_lo[0])

    # Safe denominators
    kp_hi_z_safe  = torch.where(env.kp_hi[2] > 1e-6, env.kp_hi[2], torch.tensor(1.0, device=env.device))
    kp_hi_xy_safe = torch.where(env.kp_hi[0] > 1e-6, env.kp_hi[0], torch.tensor(1.0, device=env.device))

    r_impedance_z  = mask_follow * torch.exp(-((kp_z_axis - optimal_kp_z_contact)**2) / (0.2 * kp_hi_z_safe)**2)
    r_impedance_xy = mask_follow * torch.exp(-((kp_xy_axes_mean - optimal_kp_xy_contact)**2) / (0.2 * kp_hi_xy_safe)**2)

    # 6. EFFICIENCY AND PROGRESS REWARDS
    progress_reward = mask_follow * (env.wpt_idx.float() / max(1, env.traj_mgr.T - 1))
    time_penalty    = (env.episode_length_buf.float() / env.max_episode_length)

    # ---------- NEW 1: Coplanarity Error Reward ----------
    # Compute local Z-axis in world => measure roll/pitch error
    z_local         = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3)
    ee_z_w          = quat_apply(ee_quat, z_local)
    cos_theta       = ee_z_w[:, 2].clamp(-1.0, 1.0)
    roll_pitch_err  = torch.acos(cos_theta)
    r_coplanar      = mask_follow * torch.exp(-10.0 * roll_pitch_err**2)

    # ---------- NEW 2: Energy/Torque Penalty ----------
    tau_norm        = torch.linalg.vector_norm(env.tau2, dim=1)
    tau_norm_scaled = tau_norm / (env.torque_limit.max() + 1e-6)
    p_energy        = tau_norm_scaled

    # 7. FINAL REWARD COMPOSITION
    reward = (
        1.0 * r_contact_speed +
        5.0 * contact_established_bonus +
        2.0 * r_xy +
        1.0 * r_ori +
        3.0 * r_fz_control +
        4.0 * r_deformation +
        2.0 * r_contact_continuity +
        1.0 * r_impedance_z +
        1.0 * r_impedance_xy +
        1.0 * r_coplanar +  # NEW term
        2.0 * progress_reward -
        0.5 * time_penalty -
        0.5 * p_energy           # NEW penalty
    )


    terms = dict(
        r_contact_speed=r_contact_speed,
        contact_bonus=contact_established_bonus,
        r_xy=r_xy,
        r_ori=r_ori,
        r_fz=r_fz_control,
        r_deformation=r_deformation,
        r_continuity=r_contact_continuity,
        r_imp_z=r_impedance_z,
        r_imp_xy=r_impedance_xy,
        r_coplanar=r_coplanar,
        p_energy=p_energy,
        progress=progress_reward,
        time_penalty=time_penalty,
    )
    

    # Apply lost-contact penalty
    lost_mask = getattr(env, '_just_lost_contact', torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    reward[lost_mask] -= 1000.0

    # Bonus for RISE phase entry
    rise_bonus = ((env.phase_ctrl.phase == FixedPhase.RISE).float() *
                  (env._prev_phase != FixedPhase.RISE).float()) * 10.0
    env._prev_phase = env.phase_ctrl.phase.clone()
    reward += rise_bonus

    return reward, terms

# Public alias without underscore for external imports
compute_polish_coplanar_energy_reward = _compute_polish_coplanar_energy_reward

__all__ = [
    "_compute_polish_coplanar_energy_reward",
    "compute_polish_coplanar_energy_reward",
]
