# File: polish_kpz_control_reward.py
"""
Module: polish_kpz_control_reward.py
Brief: Compute the reinforcement learning reward for the polishing task with Kp_Z control.
This reward is specifically designed for RL training where only the Z-axis stiffness (Kp_Z) 
is controlled by the agent. The reward focuses on:
 1. Optimal force control during contact phase
 2. Force stability and consistency
 3. Adaptive stiffness for different contact conditions
 4. Efficient trajectory execution
 5. Minimal force oscillations

This file should be imported and called from the main PolishEnv implementation.
"""

import torch
from ..cfg.config import F_TOUCH, F_LOST
from isaaclab.utils.math import quat_apply  # efficient quaternion-vector rotate
from ..Trajectory_Manager import FixedPhase


def _compute_polish_kpz_control_reward(env) -> torch.Tensor:
    """
    Compute Kp_Z-focused polishing reward for all environments.
    
    Args:
        env: instance of PolishEnv
        
    Returns:
        tuple: (reward tensor of shape (num_envs,), reward terms dict)
    """
    import os
    
    # =================================================================
    # CONFIGURABLE HYPERPARAMETERS (can be overridden via env vars)
    # =================================================================
    W_FORCE_TRACKING = float(os.environ.get("W_FORCE_TRACKING", "8.0"))
    FORCE_ERROR_SCALE = float(os.environ.get("FORCE_ERROR_SCALE", "0.7"))
    W_ADAPTIVE_KPZ = float(os.environ.get("W_ADAPTIVE_KPZ", "5.0"))
    OPTIMAL_KPZ_CENTER = float(os.environ.get("OPTIMAL_KPZ_CENTER", "0.6"))
    W_KPZ_CHANGE_PENALTY = float(os.environ.get("W_KPZ_CHANGE_PENALTY", "1.0"))
    W_CONTACT_BONUS = float(os.environ.get("W_CONTACT_BONUS", "10.0"))
    W_CONTACT_IMPACT_PENALTY = float(os.environ.get("W_CONTACT_IMPACT_PENALTY", "6.0"))
    CONTACT_RAMP_STEPS = int(os.environ.get("CONTACT_RAMP_STEPS", "20"))
    IMPACT_FORCE_MARGIN = float(os.environ.get("IMPACT_FORCE_MARGIN", "2.0"))
    IMPACT_FORCE_SCALE = float(os.environ.get("IMPACT_FORCE_SCALE", "5.0"))
    IMPACT_DF_SCALE = float(os.environ.get("IMPACT_DF_SCALE", "5.0"))
    W_CONTACT_ZVEL_PENALTY = float(os.environ.get("W_CONTACT_ZVEL_PENALTY", "4.0"))
    CONTACT_ZVEL_LIMIT = float(os.environ.get("CONTACT_ZVEL_LIMIT", "0.02"))
    CONTACT_ZVEL_SCALE = float(os.environ.get("CONTACT_ZVEL_SCALE", "0.05"))
    W_CONTACT_SOFT_BONUS = float(os.environ.get("W_CONTACT_SOFT_BONUS", "4.0"))
    SOFT_FORCE_BAND = float(os.environ.get("SOFT_FORCE_BAND", "4.0"))
    SOFT_DF_BAND = float(os.environ.get("SOFT_DF_BAND", "4.0"))
    W_SOFT_KPZ_BONUS = float(os.environ.get("W_SOFT_KPZ_BONUS", "2.0"))
    SOFT_KPZ_TARGET = float(os.environ.get("SOFT_KPZ_TARGET", "0.35"))
    SOFT_KPZ_SCALE = float(os.environ.get("SOFT_KPZ_SCALE", "0.2"))
    W_CONTACT_HARD_PENALTY = float(os.environ.get("W_CONTACT_HARD_PENALTY", "10.0"))
    CONTACT_HARD_LIMIT = float(os.environ.get("CONTACT_HARD_LIMIT", "-35.0"))
    CONTACT_HARD_WINDOW = int(os.environ.get("CONTACT_HARD_WINDOW", "25"))
    CONTACT_HARD_SCALE = float(os.environ.get("CONTACT_HARD_SCALE", "10.0"))
    
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

    # Current Kp_Z values (the parameter being controlled by RL)
    current_kpz = env.dynamic_kp[:, 2]  # Z-axis stiffness

    # =================================================================
    # PHASE 0-2: HOME, APPROACH, DESCENT - Standard navigation rewards
    # =================================================================
    
    # HOME phase - Quick navigation to start
    r_home_positioning = mask_home * torch.exp(-6.0 * pos_err)
    r_home_orientation = mask_home * torch.exp(-2.0 * ori_err)
    r_home_completion = mask_home * ((pos_err < 0.05) & (ori_err < 0.3)).float() * 5.0

    # APPROACH phase - Precise positioning
    r_approach_positioning = mask_approach * torch.exp(-10.0 * pos_err)
    r_approach_orientation = mask_approach * torch.exp(-3.0 * ori_err)
    
    # Get target Z position from current waypoint
    if hasattr(env.traj_mgr, 'p_traj_env') and env.traj_mgr.p_traj_env.shape[0] > 0:
        target_z = env.traj_mgr.p_traj_env[torch.arange(env.num_envs), env.wpt_idx, 2]
    else:
        target_z = torch.full((env.num_envs,), 0.1, device=env.device)
    
    z_error = torch.abs(ee_pos[:, 2] - target_z)
    r_approach_height = mask_approach * torch.exp(-20.0 * z_error)
    r_approach_completion = mask_approach * ((pos_err < 0.03) & (z_error < 0.02) & (ori_err < 0.2)).float() * 10.0

    # DESCENT phase - Fast and decisive approach to surface
    # 1. High reward for fast descent (encourage speed)
    descent_vel = -ee_vel[:, 2]  # Positive when moving down
    target_descent_speed = 0.06  # 6cm/s target speed (was 4cm/s)
    max_descent_speed = 0.12     # Allow up to 12cm/s (was 8cm/s)
    
    # Reward fast descent with exponential bonus for target speed
    speed_ratio = torch.clamp(descent_vel / target_descent_speed, 0.0, 2.0)
    r_descent_speed = mask_descent * (
        torch.where(descent_vel > 0.01,  # Only reward if actually moving down
                   speed_ratio * 6.0,    # High reward (6.0 max, was 3.0)
                   -2.0)                 # Penalty for not moving down
    )
    
    # 2. Time pressure - penalize slow descent
    # Estimate time in descent phase
    if hasattr(env, '_descent_start_time'):
        time_in_descent = env.episode_length_buf - env._descent_start_time
    else:
        # Initialize descent timer
        env._descent_start_time = torch.where(mask_descent > 0, 
                                            env.episode_length_buf, 
                                            getattr(env, '_descent_start_time', env.episode_length_buf))
        time_in_descent = torch.zeros_like(env.episode_length_buf)
    
    # Strong time penalty after reasonable descent time
    max_descent_time = 50  # 50 steps = ~2.5 seconds at 20Hz
    time_penalty = mask_descent * torch.clamp((time_in_descent - max_descent_time) / 20.0, 0.0, 5.0)
    r_descent_time_pressure = -time_penalty * 4.0  # Strong penalty
    
    # 3. XY stability during descent (essential for good contact)
    r_descent_xy_stability = mask_descent * torch.exp(-25.0 * pos_err) * 2.0  # Increased weight
    
    # 4. Progressive reward as approaching surface
    surface_z = 0.0
    distance_to_surface = torch.clamp(ee_pos[:, 2] - surface_z, 0.0, 0.2)
    # Higher reward as getting closer (exponential)
    r_descent_progress = mask_descent * torch.exp(-15.0 * distance_to_surface) * 3.0
    
    # 5. Major bonus for establishing contact quickly
    contact_established = (fz_local < F_TOUCH)
    r_descent_contact_bonus = mask_descent * contact_established.float() * W_CONTACT_BONUS  # Increased from 20.0
    
    # 6. Kp_Z preparation reward - encourage optimal stiffness for contact
    # During descent, prepare for contact with moderate-high stiffness
    kpz_normalized = (current_kpz - env.kp_lo[2]) / (env.kp_hi[2] - env.kp_lo[2])
    optimal_descent_kpz = 0.65  # Slightly higher than contact optimal (0.6)
    kpz_prep_error = torch.abs(kpz_normalized - optimal_descent_kpz)
    r_descent_kpz_prep = mask_descent * torch.exp(-kpz_prep_error / 0.3) * 2.0

    # =================================================================
    # PHASE 3: CONTACT - MAIN KP_Z CONTROL OPTIMIZATION
    # =================================================================
    
    # Basic trajectory following (reduced weight since focus is on force control)
    r_contact_positioning = mask_contact * torch.exp(-15.0 * pos_err) * 0.5
    r_contact_orientation = mask_contact * torch.exp(-5.0 * ori_err) * 0.5
    
    # === FORCE CONTROL REWARDS (Main focus) ===
    
    # 1. Target force tracking with adaptive targets
    # Adapt target force based on lateral movement speed (more force when moving faster)
    lateral_speed = torch.norm(ee_vel[:, :2], dim=1)
    adaptive_fz_target = env.fz_target - 1.5 * torch.clamp(lateral_speed / 0.05, 0.0, 1.0)

    # Contact ramp: soften targets and penalties in the first contact steps
    if not hasattr(env, "_contact_step_count"):
        env._contact_step_count = torch.zeros(env.num_envs, device=env.device)
    contact_mask = mask_contact > 0.5
    env._contact_step_count = torch.where(
        contact_mask,
        env._contact_step_count + 1,
        torch.zeros_like(env._contact_step_count),
    )
    ramp_den = max(CONTACT_RAMP_STEPS, 1)
    contact_ramp = torch.clamp(env._contact_step_count.float() / ramp_den, 0.0, 1.0)
    fz_target_ramped = adaptive_fz_target * contact_ramp
    
    # High-weight reward for accurate force tracking
    fz_error = torch.abs(fz_local - fz_target_ramped)
    r_force_tracking = mask_contact * torch.exp(-fz_error / (env.fz_eps * FORCE_ERROR_SCALE)) * W_FORCE_TRACKING
    
    # 2. Force stability reward (penalize oscillations)
    if hasattr(env, '_prev_fz_local'):
        fz_change = torch.abs(fz_local - env._prev_fz_local)
        r_force_stability = mask_contact * torch.exp(-fz_change / 2.0) * 4.0
        env._prev_fz_local = fz_local.clone()
    else:
        fz_change = torch.zeros(env.num_envs, device=env.device)
        r_force_stability = torch.zeros(env.num_envs, device=env.device)
        env._prev_fz_local = fz_local.clone()
    
    # 3. Optimal Kp_Z range reward
    # Encourage Kp_Z values in the optimal range for force control
    kpz_normalized = (current_kpz - env.kp_lo[2]) / (env.kp_hi[2] - env.kp_lo[2])
    optimal_kpz_range = torch.exp(-((kpz_normalized - OPTIMAL_KPZ_CENTER) / 0.3)**2)  # Peak at 60% of range
    r_optimal_kpz = mask_contact * optimal_kpz_range * 3.0
    
    # 4. Adaptive stiffness reward based on contact conditions
    # Higher stiffness for stable contact, lower for disturbances
    force_error_magnitude = torch.abs(fz_error)
    
    # If force error is high, reward lower stiffness (more compliance)
    # If force error is low, reward moderate-high stiffness (good tracking)
    desired_kpz_adaptive = torch.where(
        force_error_magnitude > env.fz_eps,
        torch.full_like(kpz_normalized, 0.4),  # Lower stiffness for large errors
        torch.full_like(kpz_normalized, 0.7)   # Higher stiffness for good tracking
    )
    
    kpz_adaptation_error = torch.abs(kpz_normalized - desired_kpz_adaptive)
    r_adaptive_kpz = mask_contact * torch.exp(-kpz_adaptation_error / 0.2) * W_ADAPTIVE_KPZ

    # Soft contact: encourage lower stiffness during initial contact
    soft_kpz_error = torch.abs(kpz_normalized - SOFT_KPZ_TARGET)
    r_soft_kpz = mask_contact * (1.0 - contact_ramp) * torch.exp(
        -soft_kpz_error / max(SOFT_KPZ_SCALE, 1e-6)
    ) * W_SOFT_KPZ_BONUS

    # Penalize excessive downward velocity right after contact
    down_vel = torch.clamp(-ee_vel[:, 2], 0.0, 1.0)
    p_contact_z_vel = mask_contact * (1.0 - contact_ramp) * torch.relu(
        (down_vel - CONTACT_ZVEL_LIMIT) / max(CONTACT_ZVEL_SCALE, 1e-6)
    ) * W_CONTACT_ZVEL_PENALTY

    # Soft contact bonus: small force error and low force variation early on
    soft_force = torch.exp(-fz_error / max(SOFT_FORCE_BAND, 1e-6))
    soft_df = torch.exp(-fz_change / max(SOFT_DF_BAND, 1e-6))
    r_contact_soft = mask_contact * (1.0 - contact_ramp) * soft_force * soft_df * W_CONTACT_SOFT_BONUS
    
    # 5. Contact maintenance with force quality
    contact_maintained = (fz_local < F_LOST)
    force_quality = torch.exp(-fz_error / env.fz_eps)  # How well force is controlled
    r_contact_quality = mask_contact * contact_maintained.float() * force_quality * 3.0
    
    # 6. Polishing work effectiveness
    # Reward lateral forces that indicate effective polishing work
    lateral_force_norm = torch.sqrt(fx_local**2 + fy_local**2)
    effective_work = torch.clamp(lateral_force_norm / 2.0, 0.0, 1.0)
    r_polishing_work = mask_contact * effective_work * 2.0
    
    # 7. Progress through trajectory waypoints
    if env.traj_mgr.T > 1:
        waypoint_progress = env.wpt_idx.float() / (env.traj_mgr.T - 1)
    else:
        waypoint_progress = torch.ones_like(env.wpt_idx.float())
    r_contact_progress = mask_contact * waypoint_progress * 2.0

    # Contact impact penalty (reduce initial force spikes)
    overshoot = torch.relu((fz_target_ramped - fz_local) - IMPACT_FORCE_MARGIN)
    p_contact_impact = mask_contact * (1.0 - contact_ramp) * (
        (overshoot / IMPACT_FORCE_SCALE) + (fz_change / IMPACT_DF_SCALE)
    ) * W_CONTACT_IMPACT_PENALTY

    # Hard limit penalty: clamp force spikes in early contact window
    early_contact = (env._contact_step_count <= CONTACT_HARD_WINDOW).float()
    hard_violation = torch.relu((CONTACT_HARD_LIMIT - fz_local) / max(CONTACT_HARD_SCALE, 1e-6))
    p_contact_hard = mask_contact * early_contact * hard_violation * W_CONTACT_HARD_PENALTY

    # =================================================================
    # PHASE 4: RISE - Simple completion
    # =================================================================
    rise_vel = torch.clamp(ee_vel[:, 2], 0.0, 0.08)
    r_rise_speed = mask_rise * (rise_vel / 0.04) * 2.0
    
    surface_z = 0.0
    rise_target_z = surface_z + 0.1
    rise_z_error = torch.abs(ee_pos[:, 2] - rise_target_z)
    r_rise_height = mask_rise * torch.exp(-10.0 * rise_z_error) * 1.5
    r_rise_completion = mask_rise * (ee_pos[:, 2] > rise_target_z - 0.02).float() * 8.0

    # =================================================================
    # GLOBAL REWARDS AND PENALTIES
    # =================================================================
    
    # Phase transition bonuses
    phase_transition_bonus = torch.zeros(env.num_envs, device=env.device)
    if hasattr(env, '_prev_phase_for_reward'):
        phase_advanced = (env.phase_ctrl.phase > env._prev_phase_for_reward).float()
        phase_transition_bonus = phase_advanced * 15.0
        
        # Special bonus for reaching CONTACT phase
        reached_contact = ((env.phase_ctrl.phase == FixedPhase.CONTACT) & 
                          (env._prev_phase_for_reward != FixedPhase.CONTACT)).float()
        phase_transition_bonus += reached_contact * 25.0
        
        env._prev_phase_for_reward = env.phase_ctrl.phase.clone()
    else:
        env._prev_phase_for_reward = env.phase_ctrl.phase.clone()
    
    # Reduced energy penalty (since we're only controlling one parameter)
    if hasattr(env, 'tau2'):
        tau_norm = torch.linalg.vector_norm(env.tau2, dim=1)
        tau_norm_scaled = tau_norm / (env.torque_limit.max() + 1e-6)
        p_energy = tau_norm_scaled * 0.1  # Reduced penalty
    else:
        p_energy = torch.zeros(env.num_envs, device=env.device)
    
    # Light time penalty
    time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.3
    
    # Kp_Z change penalty (discourage excessive parameter changes)
    if hasattr(env, '_prev_kpz'):
        kpz_change = torch.abs(current_kpz - env._prev_kpz)
        kpz_change_normalized = kpz_change / (env.kp_hi[2] - env.kp_lo[2])
        p_kpz_change = mask_contact * kpz_change_normalized * W_KPZ_CHANGE_PENALTY
        env._prev_kpz = current_kpz.clone()
    else:
        p_kpz_change = torch.zeros(env.num_envs, device=env.device)
        env._prev_kpz = current_kpz.clone()

    # =================================================================
    # FINAL REWARD COMPOSITION
    # =================================================================
    reward = (
        # Navigation phases (reduced weights)
        r_home_positioning + r_home_orientation + r_home_completion +
        r_approach_positioning + r_approach_orientation + r_approach_height + r_approach_completion +
        
        # DESCENT phase - ENHANCED for speed and efficiency
        r_descent_speed +           # 6.0 max - high reward for fast descent
        r_descent_time_pressure +   # -20.0 max penalty - urgency for quick descent
        r_descent_xy_stability +    # 2.0 max - maintain position accuracy
        r_descent_progress +        # 3.0 max - reward approaching surface
        r_descent_contact_bonus +   # 30.0 max - major bonus for contact
        r_descent_kpz_prep +        # 2.0 max - prepare optimal stiffness
        
        # Contact phase - MAIN FOCUS (high weights)
        r_contact_positioning + r_contact_orientation +
        r_force_tracking +        # 8.0 max - most important
        r_force_stability +       # 4.0 max - prevent oscillations  
        r_optimal_kpz +          # 3.0 max - encourage good Kp_Z range
        r_adaptive_kpz +         # 5.0 max - adapt to conditions
        r_soft_kpz +             # soft contact compliance early on
        r_contact_quality +       # 3.0 max - force + contact quality
        r_contact_soft +         # bonus for gentle contact
        r_polishing_work +        # 2.0 max - effective polishing
        r_contact_progress +      # 2.0 max - task completion
        
        # Rise phase
        r_rise_speed + r_rise_height + r_rise_completion +
        
        # Global terms
        phase_transition_bonus - p_energy - time_penalty - p_kpz_change - p_contact_impact - p_contact_z_vel - p_contact_hard
    )

    # =================================================================
    # FAILURE PENALTIES
    # =================================================================
    
    # Contact loss penalty (moderate)
    contact_lost = getattr(env, '_just_lost_contact', 
                          torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    reward[contact_lost] -= 20.0
    
    # Safety bounds penalty
    out_of_bounds = _check_safety_bounds(env, ee_pos)
    reward[out_of_bounds] -= 10.0

    # =================================================================
    # REWARD TERMS FOR LOGGING
    # =================================================================
    terms = {
        # Navigation phases
        'r_home_total': r_home_positioning + r_home_orientation + r_home_completion,
        'r_approach_total': r_approach_positioning + r_approach_orientation + r_approach_height + r_approach_completion,
        
        # DESCENT phase - detailed logging for speed analysis
        'r_descent_speed': r_descent_speed,
        'r_descent_time_pressure': r_descent_time_pressure,
        'r_descent_xy_stability': r_descent_xy_stability,
        'r_descent_progress': r_descent_progress,
        'r_descent_contact_bonus': r_descent_contact_bonus,
        'r_descent_kpz_prep': r_descent_kpz_prep,
        'r_descent_total': r_descent_speed + r_descent_time_pressure + r_descent_xy_stability + 
                          r_descent_progress + r_descent_contact_bonus + r_descent_kpz_prep,
        
        # Main Kp_Z control terms (detailed logging)
        'r_force_tracking': r_force_tracking,
        'r_force_stability': r_force_stability,
        'r_optimal_kpz': r_optimal_kpz,
        'r_adaptive_kpz': r_adaptive_kpz,
        'r_soft_kpz': r_soft_kpz,
        'r_contact_quality': r_contact_quality,
        'r_contact_soft': r_contact_soft,
        'r_polishing_work': r_polishing_work,
        'r_contact_progress': r_contact_progress,
        
        # Rise phase
        'r_rise_total': r_rise_speed + r_rise_height + r_rise_completion,
        
        # Global terms
        'phase_bonus': phase_transition_bonus,
        'p_energy': p_energy,
        'p_kpz_change': p_kpz_change,
        'p_contact_impact': p_contact_impact,
        'p_contact_z_vel': p_contact_z_vel,
        'p_contact_hard': p_contact_hard,
        'time_penalty': time_penalty,
        
        # Key monitoring values
        'current_phase': env.phase_ctrl.phase.float(),
        'current_kpz': current_kpz,
        'kpz_normalized': kpz_normalized,
        'fz_error': fz_error,
        'fz_target_ramped': fz_target_ramped,
        'contact_ramp': contact_ramp,
        'contact_step_count': env._contact_step_count,
        'force_stability_change': getattr(env, '_prev_fz_local', fz_local) - fz_local if hasattr(env, '_prev_fz_local') else torch.zeros_like(fz_local),
        'adaptive_fz_target': adaptive_fz_target,
        
        # DESCENT monitoring - critical for speed analysis
        'descent_velocity': descent_vel,  # Actual descent speed
        'descent_speed_ratio': speed_ratio,  # How fast relative to target
        'time_in_descent': time_in_descent if hasattr(env, '_descent_start_time') else torch.zeros_like(env.episode_length_buf),
        'distance_to_surface': distance_to_surface,
    }

    return reward, terms

# Public alias without underscore for external imports
compute_polish_kpz_control_reward = _compute_polish_kpz_control_reward

__all__ = [
    "_compute_polish_kpz_control_reward",
    "compute_polish_kpz_control_reward",
]


def _check_safety_bounds(env, ee_pos: torch.Tensor) -> torch.Tensor:
    """Check if end-effector is within safe workspace bounds."""
    x_min, x_max = 0.1, 0.9
    y_min, y_max = -0.6, 0.6
    z_min, z_max = -0.15, 1.2
    
    out_of_bounds = (
        (ee_pos[:, 0] < x_min) | (ee_pos[:, 0] > x_max) |
        (ee_pos[:, 1] < y_min) | (ee_pos[:, 1] > y_max) |
        (ee_pos[:, 2] < z_min) | (ee_pos[:, 2] > z_max)
    )
    
    return out_of_bounds
