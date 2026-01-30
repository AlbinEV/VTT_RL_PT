# File: default_reward.py
# Description: Implements a comprehensive reward for the polishing task in PolishEnv.
#              Combines:
#                1) Contact establishment speed bonus and contact initiation reward,
#                2) Trajectory following rewards (position and orientation alignment),
#                3) Force control reward toward adaptive target force,
#                4) Deformation/work continuity reward,
#                5) Coplanarity bonus penalizing roll and pitch deviation of the end-effector relative to the world Z-axis,
#                6) Light energy penalty proportional to control effort,
#                7) Large penalty on contact loss and one-time bonus on entering the RISE phase.

import torch
from ..Trajectory_Manager import FixedPhase


def _get_rewards(env) -> torch.Tensor:
    """
    Advanced reward function for polishing task with dynamic impedance control.
    Returns a tensor of shape (num_envs,).
    """
    if not self._trajectory_initialized: # Guard against missing initialization
        return torch.zeros(self.num_envs, device=self.device)

    # Phase masks from PhaseController
    mask_descent = (self.phase_ctrl.phase == FixedPhase.DESCENT).float()
    mask_follow = (self.phase_ctrl.phase == FixedPhase.CONTACT).float()   # In contact, following trajectory

    f_ext_global = self.carpet_sensor.data.net_forces_w[:, 0] # (N,3)
    ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx] # (N,4)

    f_ext_local = torch.zeros_like(f_ext_global)
    for i in range(self.num_envs): # This loop is slow for many envs, consider batched quat_apply
        rot_matrix = quat_to_rot_matrix(ee_quat[i]) # quat_to_rot_matrix needs to handle batch if ee_quat is batched
        f_ext_local[i] = rot_matrix.T @ f_ext_global[i]
    # TODO: Implement batched quaternion rotation for f_ext_global to f_ext_local

    fz_local = f_ext_local[:, 2]
    fx_local = f_ext_local[:, 0]
    fy_local = f_ext_local[:, 1]

    # 1. CONTACT ESTABLISHMENT REWARD (Phase 0)
    descent_speed = torch.clamp(-self.robot.data.body_lin_vel_w[:, self.ee_body_idx, 2], 0.0, 0.1)
    r_contact_speed = mask_descent * (descent_speed / 0.05)
    contact_established_bonus = mask_descent * (fz_local < F_TOUCH).float() * 10.0 # Use F_TOUCH

    # 2. TRAJECTORY FOLLOWING (Phase 1)
    r_xy = mask_follow * torch.exp(-15.0 * self._last_pos_err)
    r_ori = mask_follow * torch.exp(-3.0 * self._last_ori_err)

    # 3. FORCE CONTROL (Phase 1)
    ee_vel_xy_norm = torch.norm(self.robot.data.body_lin_vel_w[:, self.ee_body_idx, :2], dim=1)
    adaptive_fz_target = self.fz_target - 2.0 * torch.clamp(ee_vel_xy_norm / 0.05, 0.0, 1.0)
    fz_error = torch.abs(fz_local - adaptive_fz_target)
    r_fz_control = mask_follow * torch.exp(-fz_error / self.fz_eps) # Use fz_eps for scaling

    # 4. DEFORMATION/WORK DONE REWARD
    lateral_force_norm = torch.sqrt(fx_local**2 + fy_local**2)
    r_deformation = mask_follow * torch.clamp(lateral_force_norm / 2.0, 0.0, 1.0)
    in_contact_during_follow = (fz_local < F_LOST).float() # Use F_LOST for maintaining contact
    r_contact_continuity = mask_follow * in_contact_during_follow

    # 5. IMPEDANCE CONTROL OPTIMIZATION
    kp_z_axis = self.dynamic_kp[:, 2]
    kp_xy_axes_mean = self.dynamic_kp[:, :2].mean(dim=1)
    
    # Define optimal Kp values (example, can be tuned)
    # kp_lo/hi are (6,) tensors. Need to access specific components.
    optimal_kp_z_contact = self.kp_lo[2] + 0.2 * (self.kp_hi[2] - self.kp_lo[2]) # Softer Z in contact
    optimal_kp_xy_contact = self.kp_lo[0] + 0.8 * (self.kp_hi[0] - self.kp_lo[0]) # Stiffer XY in contact (assuming X,Y similar limits)

    r_impedance_z = torch.zeros_like(mask_follow)
    r_impedance_xy = torch.zeros_like(mask_follow)

    if mask_follow.any(): # Avoid division by zero if kp_hi can be zero
        # Ensure kp_hi components are not zero before using as divisor
        kp_hi_z_safe = torch.where(self.kp_hi[2] > 1e-6, self.kp_hi[2], torch.tensor(1.0, device=self.device))
        kp_hi_xy_safe = torch.where(self.kp_hi[0] > 1e-6, self.kp_hi[0], torch.tensor(1.0, device=self.device))

        r_impedance_z = mask_follow * torch.exp(-((kp_z_axis - optimal_kp_z_contact)**2) / (0.2 * kp_hi_z_safe)**2)
        r_impedance_xy = mask_follow * torch.exp(-((kp_xy_axes_mean - optimal_kp_xy_contact)**2) / (0.2 * kp_hi_xy_safe)**2)

    # 6. EFFICIENCY AND PROGRESS REWARDS
    progress_reward = mask_follow * (self.wpt_idx.float() / max(1, self.traj_mgr.T - 1))
    
    # episode_length_buf is from DirectRLEnv, max_episode_length too
    time_penalty = (self.episode_length_buf.float() / self.max_episode_length) * 0.5 # Small penalty for time

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
        2.0 * progress_reward - # Make time a penalty
        time_penalty
    )

    lost_mask = getattr(self, "_just_lost_contact",
                        torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
    reward[lost_mask] -= 1000.0
    
    # Add bonus when entering RISE phase
    rise_bonus = ((self.phase_ctrl.phase == FixedPhase.RISE).float() *
              (self._prev_phase != FixedPhase.RISE).float()) * 10.0
    self._prev_phase = self.phase_ctrl.phase.clone()
    reward += rise_bonus

    return reward
