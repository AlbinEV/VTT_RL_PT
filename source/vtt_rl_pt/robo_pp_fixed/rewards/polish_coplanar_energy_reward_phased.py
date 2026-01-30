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


def _compute_polish_coplanar_energy_reward_phased(env) -> torch.Tensor:
    """
    Compute phase-based polishing reward for all environments.
    
    Args:
        env: instance of PolishEnv
        
    Returns:
        tuple: (reward tensor of shape (num_envs,), reward terms dict)
    """
    
    # =================================================================
    # SETUP: Phase masks and basic state information
    # =================================================================
    mask_home     = (env.phase_ctrl.phase == FixedPhase.HOME).float()
    mask_approach = (env.phase_ctrl.phase == FixedPhase.APPROACH).float()  
    mask_descent  = (env.phase_ctrl.phase == FixedPhase.DESCENT).float()
    mask_contact  = (env.phase_ctrl.phase == FixedPhase.CONTACT).float()
    mask_rise     = (env.phase_ctrl.phase == FixedPhase.RISE).float()

    # Current state information
    pos_err = env._last_pos_err  # XY-plane position error
    ori_err = env._last_ori_err  # orientation error
    ee_pos = env.robot.data.body_pos_w[:, env.ee_body_idx]  # End-effector position
    ee_vel = env.robot.data.body_lin_vel_w[:, env.ee_body_idx]  # End-effector velocity
    
    # Force information
    f_ext_global = env.carpet_sensor.data.net_forces_w[:, 0]  # (N,3)
    ee_quat = env.robot.data.body_quat_w[:, env.ee_body_idx]  # (N,4)
    q_conj = torch.cat([ee_quat[:, 0:1], -ee_quat[:, 1:]], dim=1)
    f_ext_local = quat_apply(q_conj, f_ext_global)  # (N,3)
    fx_local, fy_local, fz_local = f_ext_local.unbind(dim=1)

    # =================================================================
    # PHASE 0: HOME - Navigate to starting position
    # =================================================================
    # Reward for moving towards home position
    r_home_positioning = mask_home * torch.exp(-8.0 * pos_err)
    r_home_orientation = mask_home * torch.exp(-3.0 * ori_err)
    
    # Bonus for reaching home position (enables transition to next phase)
    home_reached = (pos_err < 0.05) & (ori_err < 0.3)
    r_home_completion = mask_home * home_reached.float() * 8.0

    # =================================================================
    # PHASE 1: APPROACH - Move to approach position above surface
    # =================================================================
    # Strong reward for accurate positioning during approach
    r_approach_positioning = mask_approach * torch.exp(-12.0 * pos_err)
    r_approach_orientation = mask_approach * torch.exp(-4.0 * ori_err)
    
    # Critical: Reward for being at correct approach height
    # Get target Z position from current waypoint
    if hasattr(env.traj_mgr, 'p_traj_env') and env.traj_mgr.p_traj_env.shape[0] > 0:
        target_z = env.traj_mgr.p_traj_env[torch.arange(env.num_envs), env.wpt_idx, 2]
    else:
        target_z = torch.full((env.num_envs,), 0.1, device=env.device)  # fallback
    
    z_error = torch.abs(ee_pos[:, 2] - target_z)
    r_approach_height = mask_approach * torch.exp(-25.0 * z_error)
    
    # Major bonus for reaching approach position accurately
    approach_reached = (pos_err < 0.03) & (z_error < 0.02) & (ori_err < 0.2)
    r_approach_completion = mask_approach * approach_reached.float() * 15.0

    # =================================================================
    # PHASE 2: DESCENT - Descend until contact is established
    # =================================================================
    # Reward controlled downward movement
    descent_vel = torch.clamp(-ee_vel[:, 2], 0.0, 0.08)  # Controlled descent speed
    r_descent_speed = mask_descent * (descent_vel / 0.04) * 4.0  # Normalize to 0-4
    
    # Critical: Maintain XY position during descent
    r_descent_xy_stability = mask_descent * torch.exp(-25.0 * pos_err)
    
    # Progressive reward as we approach the surface
    surface_z = 0.0  # Assuming contact surface at Z=0
    distance_to_surface = torch.clamp(ee_pos[:, 2] - surface_z, 0.0, 0.2)
    r_descent_progress = mask_descent * torch.exp(-10.0 * distance_to_surface) * 2.0
    
    # Major bonus for establishing contact
    contact_established = (fz_local < F_TOUCH)
    r_descent_contact_bonus = mask_descent * contact_established.float() * 25.0

    # =================================================================
    # PHASE 3: CONTACT - Lateral movement while maintaining contact
    # =================================================================
    # Precise trajectory following during contact phase
    r_contact_positioning = mask_contact * torch.exp(-18.0 * pos_err)
    r_contact_orientation = mask_contact * torch.exp(-6.0 * ori_err)
    
    # Force control reward - maintaining target contact force
    adaptive_fz_target = env.fz_target - 2.0 * torch.clamp(torch.norm(ee_vel[:, :2], dim=1) / 0.05, 0.0, 1.0)
    fz_error = torch.abs(fz_local - adaptive_fz_target)
    r_contact_force = mask_contact * torch.exp(-fz_error / env.fz_eps) * 3.0
    
    # Contact maintenance reward
    contact_maintained = (fz_local < F_LOST)
    r_contact_maintenance = mask_contact * contact_maintained.float() * 3.0
    
    # Progress through trajectory waypoints
    if env.traj_mgr.T > 1:
        waypoint_progress = env.wpt_idx.float() / (env.traj_mgr.T - 1)
    else:
        waypoint_progress = torch.ones_like(env.wpt_idx.float())
    r_contact_progress = mask_contact * waypoint_progress * 4.0
    
    # Deformation/work reward (polishing effectiveness)
    lateral_force_norm = torch.sqrt(fx_local**2 + fy_local**2)
    r_contact_work = mask_contact * torch.clamp(lateral_force_norm / 3.0, 0.0, 1.0) * 2.0

    # =================================================================
    # PHASE 4: RISE - Lift up after task completion
    # =================================================================
    # Reward controlled upward movement
    rise_vel = torch.clamp(ee_vel[:, 2], 0.0, 0.08)
    r_rise_speed = mask_rise * (rise_vel / 0.04) * 3.0
    
    # Reward for reaching target rise height
    rise_target_z = surface_z + 0.1  # 10cm above surface
    rise_z_error = torch.abs(ee_pos[:, 2] - rise_target_z)
    r_rise_height = mask_rise * torch.exp(-15.0 * rise_z_error) * 2.0
    
    # Completion bonus
    rise_completed = (ee_pos[:, 2] > rise_target_z - 0.02)
    r_rise_completion = mask_rise * rise_completed.float() * 10.0

    # =================================================================
    # GLOBAL REWARDS AND PENALTIES
    # =================================================================
    
    # Phase transition bonuses (reward progression through phases)
    phase_transition_bonus = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, '_prev_phase_for_reward'):
        # Detect phase advancement
        phase_advanced = (env.phase_ctrl.phase > env._prev_phase_for_reward).float()
        phase_transition_bonus = phase_advanced * 20.0  # Major bonus for phase progression
        
        # Special bonus for reaching CONTACT phase (critical milestone)
        reached_contact = ((env.phase_ctrl.phase == FixedPhase.CONTACT) & 
                          (env._prev_phase_for_reward != FixedPhase.CONTACT)).float()
        phase_transition_bonus += reached_contact * 30.0
        
        # Update previous phase
        env._prev_phase_for_reward = env.phase_ctrl.phase.clone()
    else:
        env._prev_phase_for_reward = env.phase_ctrl.phase.clone()
    
    # Energy penalty (encourage efficiency across all phases)
    tau_norm = torch.linalg.vector_norm(env.tau2, dim=1)
    tau_norm_scaled = tau_norm / (env.torque_limit.max() + 1e-6)
    p_energy = tau_norm_scaled * 0.3
    
    # Gentle time penalty to encourage task completion
    time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.5
    
    # Coplanarity reward (only during contact)
    z_local = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3)
    ee_z_w = quat_apply(ee_quat, z_local)
    cos_theta = ee_z_w[:, 2].clamp(-1.0, 1.0)
    roll_pitch_err = torch.acos(cos_theta)
    r_coplanar = mask_contact * torch.exp(-8.0 * roll_pitch_err**2) * 1.5

    # =================================================================
    # FINAL REWARD COMPOSITION
    # =================================================================
    reward = (
        # HOME phase rewards (8 + 3 + 8 = 19 max)
        r_home_positioning +
        r_home_orientation +
        r_home_completion +
        
        # APPROACH phase rewards (12 + 4 + exp(-25*z_err) + 15 = ~31 max)
        r_approach_positioning +
        r_approach_orientation +
        r_approach_height +
        r_approach_completion +
        
        # DESCENT phase rewards (4 + exp(-25*pos) + 2 + 25 = ~31 max)
        r_descent_speed +
        r_descent_xy_stability +
        r_descent_progress +
        r_descent_contact_bonus +
        
        # CONTACT phase rewards (exp(-18*pos) + exp(-6*ori) + 3 + 3 + 4 + 2 + 1.5 = ~13.5 + exp terms)
        r_contact_positioning +
        r_contact_orientation +
        r_contact_force +
        r_contact_maintenance +
        r_contact_progress +
        r_contact_work +
        r_coplanar +
        
        # RISE phase rewards (3 + 2 + 10 = 15 max)
        r_rise_speed +
        r_rise_height +
        r_rise_completion +
        
        # Global bonuses and penalties
        phase_transition_bonus -
        p_energy -
        time_penalty
    )

    # =================================================================
    # FAILURE PENALTIES (reduced severity for better learning)
    # =================================================================
    
    # Contact loss penalty (much less severe than original)
    contact_lost = getattr(env, '_just_lost_contact', 
                          torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    reward[contact_lost] -= 250.0  # Reduced from 1000.0 to 25.0
    
    # Safety bounds penalty
    out_of_bounds = _check_safety_bounds(env, ee_pos)
    reward[out_of_bounds] -= 15.0

    # =================================================================
    # REWARD TERMS FOR LOGGING
    # =================================================================
    terms = {
        # Home phase
        'r_home_pos': r_home_positioning,
        'r_home_ori': r_home_orientation, 
        'r_home_complete': r_home_completion,
        
        # Approach phase
        'r_approach_pos': r_approach_positioning,
        'r_approach_ori': r_approach_orientation,
        'r_approach_height': r_approach_height,
        'r_approach_complete': r_approach_completion,
        
        # Descent phase
        'r_descent_speed': r_descent_speed,
        'r_descent_xy': r_descent_xy_stability,
        'r_descent_progress': r_descent_progress,
        'r_descent_contact': r_descent_contact_bonus,
        
        # Contact phase
        'r_contact_pos': r_contact_positioning,
        'r_contact_ori': r_contact_orientation,
        'r_contact_force': r_contact_force,
        'r_contact_maintain': r_contact_maintenance,
        'r_contact_progress': r_contact_progress,
        'r_contact_work': r_contact_work,
        'r_coplanar': r_coplanar,
        
        # Rise phase
        'r_rise_speed': r_rise_speed,
        'r_rise_height': r_rise_height,
        'r_rise_complete': r_rise_completion,
        
        # Global terms
        'phase_bonus': phase_transition_bonus,
        'p_energy': p_energy,
        'time_penalty': time_penalty,
        
        # Current phase info (for debugging)
        'current_phase': env.phase_ctrl.phase.float(),
        'pos_error': pos_err,
        'ori_error': ori_err,
    }

    return reward, terms

# Public alias without underscore for external imports
compute_polish_coplanar_energy_reward_phased = _compute_polish_coplanar_energy_reward_phased

__all__ = [
    "_compute_polish_coplanar_energy_reward_phased",
    "compute_polish_coplanar_energy_reward_phased",
]


def _check_safety_bounds(env, ee_pos: torch.Tensor) -> torch.Tensor:
    """Check if end-effector is within safe workspace bounds."""
    # Define safe workspace bounds based on robot setup
    x_min, x_max = 0.1, 0.9   # Reasonable X range
    y_min, y_max = -0.6, 0.6  # Reasonable Y range  
    z_min, z_max = -0.15, 1.2 # Z range (slightly below surface to above reach)
    
    out_of_bounds = (
        (ee_pos[:, 0] < x_min) | (ee_pos[:, 0] > x_max) |
        (ee_pos[:, 1] < y_min) | (ee_pos[:, 1] > y_max) |
        (ee_pos[:, 2] < z_min) | (ee_pos[:, 2] > z_max)
    )
    
    return out_of_bounds