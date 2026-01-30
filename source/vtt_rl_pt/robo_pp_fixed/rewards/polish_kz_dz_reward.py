from __future__ import annotations
import torch
from ..Trajectory_Manager import FixedPhase
from isaaclab.utils.math import quat_apply


def _safe_getattr(obj, name: str, default):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def compute_polish_kz_dz_reward(env):
    """
    Reward specialized for 1D Z-axis impedance control with variable Kz and damping ratio Dz.
    Goal: outperform pure OSC and OSC+Kz-only by stabilizing contact, tracking force/height,
    and encouraging smooth, well-damped transients.

        Terms:
    - Descent:
      * Encourage negative vz to reach contact without overshoot.
      * Preforce shaping around ~ -1N just before contact.
    - Contact:
      * Force tracking to env.fz_target with soft slope (uses 3x fz_eps).
      * Vertical velocity damping: penalize |vz| and |Δvz| (stability).
      * Damping sweet-spot: prefer Dz in [0.7, 1.2] (critically damped-ish).
      * Kz moderation: discourage extremes, prefer mid range normalized ~0.6.
            * Stability penalties (surrogates for Nyquist/passivity):
                - Discrete-time margin: penalize wn*dt near/exceeding stability bounds.
                - Passivity observer: penalize net positive injected energy at contact.
                - Force derivative and error zero-crossings to curb oscillations.
    - Rise:
      * Bonus to encourage clean detachment and completion.
    - Global:
      * Penalize big changes in Kz/Dz.
      * Time penalty and small energy proxy if available.
    """
    device = env.device
    phase = env.phase_ctrl.phase
    mask_home     = (phase == FixedPhase.HOME).float()
    mask_approach = (phase == FixedPhase.APPROACH).float()
    mask_descent  = (phase == FixedPhase.DESCENT).float()
    mask_contact  = (phase == FixedPhase.CONTACT).float()
    mask_rise     = (phase == FixedPhase.RISE).float()

    # EE state
    ee_vel_w  = env.robot.data.body_lin_vel_w[:, env.ee_body_idx]
    vz        = ee_vel_w[:, 2]

    # External force in EE frame to get fz
    f_ext_w = env.carpet_sensor.data.net_forces_w[:, 0]
    ee_quat = env.robot.data.body_quat_w[:, env.ee_body_idx]
    q_conj  = torch.cat([ee_quat[:, :1], -ee_quat[:, 1:]], dim=1)
    f_ext_b = quat_apply(q_conj, f_ext_w)
    fz      = f_ext_b[:, 2]

    # Gains (assume dynamic_kp[:,2] and dynamic_zeta[:,2] exist)
    kz   = env.dynamic_kp[:, 2]
    dz   = env.dynamic_zeta[:, 2] if hasattr(env, 'dynamic_zeta') else torch.full_like(kz, 0.9)
    kz_n = (kz - env.kp_lo[2]) / (env.kp_hi[2] - env.kp_lo[2] + 1e-8)

    # Descent shaping
    r_descent_vz = mask_descent * torch.clamp(torch.relu(-vz), 0.0, 0.2) * 4.0
    r_precontact = mask_descent * torch.exp(-torch.abs(fz + 1.0) / 2.0) * 2.0

    # Contact force tracking and stability
    fz_target = _safe_getattr(env, 'fz_target', -10.0)
    fz_eps    = _safe_getattr(env, 'fz_eps', 0.5)
    fz_err    = torch.abs(fz - fz_target)
    r_fz_track = mask_contact * torch.exp(-fz_err / (3.0 * fz_eps + 1e-6)) * 7.0

    # Vertical damping/stability: low velocity in steady-state and smooth changes
    if not hasattr(env, '_prev_vz'):
        env._prev_vz = vz.clone()
    dvz = torch.abs(vz - env._prev_vz)
    env._prev_vz = vz.clone()
    r_vz_stability = mask_contact * torch.exp(-(torch.abs(vz) + dvz) / 0.5) * 3.0

    # Damping sweet-spot around critical damping (normalized dz ~ 0.9)
    dz_opt   = 0.9
    dz_width = 0.35
    r_dz_opt = mask_contact * torch.exp(-((dz - dz_opt) / dz_width) ** 2) * 2.5

    # Kz moderation and adaptivity: prefer mid, soften if error is large
    kz_opt   = torch.where(fz_err > fz_eps, 0.45, 0.65)  # normalized target
    r_kz_opt = mask_contact * torch.exp(-torch.abs(kz_n - kz_opt) / 0.25) * 2.5

    # Smoothness penalties on Kz/Dz changes
    if hasattr(env, '_prev_kz_dz'):
        dkz = torch.abs(kz - env._prev_kz_dz[:, 0]) / (env.kp_hi[2] - env.kp_lo[2] + 1e-8)
        ddz = torch.abs(dz - env._prev_kz_dz[:, 1]) / (torch.max(dz).detach() - torch.min(dz).detach() + 1e-8)
        p_kz_dz_change = (dkz + ddz) * 1.0
        env._prev_kz_dz = torch.stack([kz, dz], dim=1)
    else:
        p_kz_dz_change = torch.zeros_like(kz)
        env._prev_kz_dz = torch.stack([kz, dz], dim=1)

    # --- Stability penalties (control-theory inspired, light-weight) ---
    # 1) Discrete-time margin surrogate via wn*dt (2nd-order approx)
    #    wn = sqrt(Kz / M_eff_z). Estimate M_eff_z or fallback.
    #    Penalize if wn*dt grows above a safe fraction of the ZOH stability limit.
    # Try multiple sources for dt robustly
    dt = None
    for attr in ('dt', 'sim_dt', 'step_dt'):
        if hasattr(env, attr):
            val = getattr(env, attr)
            if isinstance(val, (float, int)):
                dt = float(val)
                break
    if dt is None:
        cfg = getattr(env, 'cfg', None)
        if cfg is not None:
            sim = getattr(cfg, 'sim', None)
            if sim is not None and hasattr(sim, 'dt'):
                val = getattr(sim, 'dt')
                if isinstance(val, (float, int)):
                    dt = float(val)
    if dt is None:
        dt = 1.0 / 120.0
    eff_mass_z = getattr(env, 'eff_mass_z', None)
    if eff_mass_z is None:
        # Conservative constant effective mass along Z (kg). Tune if available.
        eff_mass_z = 3.0
    if not torch.is_tensor(eff_mass_z):
        eff_mass_z = torch.full_like(kz, float(eff_mass_z))
    wn = torch.sqrt(torch.clamp(kz / (eff_mass_z + 1e-8), min=1e-6))
    wn_dt = wn * dt
    # ZOH stability boundary is ~2 for undamped; use margin (e.g., 1.6) to be conservative
    p_wn_margin = mask_contact * torch.relu(wn_dt - 1.6) * 1.5

    # 2) Damping lower-bound: avoid underdamped oscillatory regime
    p_dz_low = mask_contact * torch.relu(0.6 - dz) * 2.0

    # 3) Passivity observer (energy-based). Penalize net positive injected energy at contact.
    #    E[k] = E[k-1] + fz*vz*dt. When E grows beyond a small margin, add penalty.
    if not hasattr(env, '_E_pass'):
        env._E_pass = torch.zeros_like(kz)
    # Reset observer outside contact to avoid leakage across phases
    env._E_pass = env._E_pass * (mask_contact > 0).float()
    env._E_pass = env._E_pass + (fz * vz * dt * mask_contact)
    p_passivity = mask_contact * torch.relu(env._E_pass - 0.5) * 0.5  # 0.5J margin, light weight

    # 4) Force derivative penalty to suppress high-frequency chatter
    if not hasattr(env, '_prev_fz'):
        env._prev_fz = fz.clone()
    dfz = torch.abs(fz - env._prev_fz) / (dt + 1e-9)
    env._prev_fz = fz.clone()
    p_dfz = mask_contact * (dfz / (50.0 + torch.abs(fz))) * 0.2  # scaled by magnitude

    # 5) Zero-crossings of force error (oscillation indicator)
    if not hasattr(env, '_prev_fz_err'):
        env._prev_fz_err = (fz - fz_target).clone()
    e_now = fz - fz_target
    e_prev = env._prev_fz_err
    zero_cross = (torch.sign(e_now) * torch.sign(e_prev) < 0).float() * (torch.abs(e_now) > 0.5 * fz_eps).float()
    env._prev_fz_err = e_now.clone()
    p_zero_cross = mask_contact * zero_cross * 0.5

    # Rise bonus
    r_rise = mask_rise * 2.0

    # Global penalties
    time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.04
    if hasattr(env, 'tau2'):
        tau_norm = torch.linalg.vector_norm(env.tau2, dim=1)
        p_energy = tau_norm / (env.torque_limit.max() + 1e-6) * 0.08
    else:
        p_energy = torch.zeros_like(time_penalty)

    reward = (
        r_descent_vz + r_precontact +
        r_fz_track + r_vz_stability + r_dz_opt + r_kz_opt +
        r_rise
        # penalties
        - p_kz_dz_change - time_penalty - p_energy
        - p_wn_margin - p_dz_low - p_passivity - p_dfz - p_zero_cross
    )

    terms = {
        'r_descent_vz': r_descent_vz,
        'r_precontact': r_precontact,
        'r_fz_track': r_fz_track,
        'r_vz_stability': r_vz_stability,
        'r_dz_opt': r_dz_opt,
        'r_kz_opt': r_kz_opt,
        'p_kz_dz_change': p_kz_dz_change,
        'time_penalty': time_penalty,
        'p_energy': p_energy,
        'kz_n': kz_n,
        'dz': dz,
        # stability diagnostics
        'wn_dt': wn_dt,
        'p_wn_margin': p_wn_margin,
        'p_dz_low': p_dz_low,
        'p_passivity': p_passivity,
        'p_dfz': p_dfz,
        'p_zero_cross': p_zero_cross,
    }
    return reward, terms

__all__ = ['compute_polish_kz_dz_reward']
