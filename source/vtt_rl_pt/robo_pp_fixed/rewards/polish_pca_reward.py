"""PCA-based polishing reward.

Idea:
 1. Use instantaneous feature vector x_t (kp, f_ext, lambda if available) standardized online.
 2. Project onto a fixed PC1 direction w (provided via config or hardcoded default from offline OSC-only analysis).
 3. Reward alignment with desirable region along PC1 plus moderate variance in orthogonal components.
 4. Maintain safety / basic task structure via lightweight auxiliary terms.

Assumptions:
 - A dictionary env.pca_cfg may exist with keys:
       'pc1_loadings': list[float] (length matches selected features)
       'feature_order': list[str]
       'scale_mean': list[float] (optional) offline means
       'scale_std': list[float] (optional) offline stds (>0)
 - If not present, we build feature vector using available tensors and normalize on-the-fly.

Design choices:
 - Do NOT backprop through a running PCA. We keep the direction fixed -> stable learning signal.
 - Core reward encourages moving feature vector toward a target scalar score along PC1 (e.g., median of offline distribution ~0).
 - Penalize extreme projections to avoid collapse.

"""

from __future__ import annotations
import torch
from ..Trajectory_Manager import FixedPhase
from ..cfg.config import F_TOUCH, F_LOST


DEF_FEATURES = [
    # Order matches table provided by user (lambda first, then forces, then kp)
    'lambda_diag_0','lambda_diag_1','lambda_diag_2','lambda_diag_3','lambda_diag_4','lambda_diag_5',
    'f_ext_0','f_ext_1','f_ext_2',
    'kp_0','kp_1','kp_2','kp_3','kp_4','kp_5'
]

# Default PC1 loadings from user-provided analysis (table). kp loadings ~0 retained.
DEFAULT_PC1 = torch.tensor([
    0.434,  # lambda_diag_0
   -0.363,  # lambda_diag_1
   -0.373,  # lambda_diag_2
   -0.320,  # lambda_diag_3
   -0.338,  # lambda_diag_4
   -0.096,  # lambda_diag_5
   -0.376,  # f_ext_0
   -0.376,  # f_ext_1
   -0.180,  # f_ext_2
    0.000,  # kp_0
    0.000,  # kp_1
    0.000,  # kp_2
    0.000,  # kp_3
    0.000,  # kp_4
    0.000,  # kp_5
], dtype=torch.float32)


def _gather_features(env, feature_order):
    feats = []
    N = env.num_envs
    # Lambda diag (operational inertia) if available
    if any(f.startswith('lambda_diag_') for f in feature_order):
        lam = getattr(env, '_last_lambda', None)
        if lam is not None:
            lam_t = torch.as_tensor(lam, device=env.device, dtype=torch.float32)
            # Accept shapes: (6,), (N,6). Broadcast if needed.
            if lam_t.ndim == 1 and lam_t.numel() == 6:
                lam_t = lam_t.view(1,6).repeat(N,1)
            elif lam_t.ndim == 2 and lam_t.shape[0] == N and lam_t.shape[1] == 6:
                pass
            else:
                if not hasattr(env, '_warned_lambda_shape'):
                    print('[PCA Reward] Unexpected lambda_diag shape', tuple(lam_t.shape), '-> using zeros')
                    env._warned_lambda_shape = True
                lam_t = torch.zeros(N,6, device=env.device)
            feats.append(lam_t)
        else:
            feats.append(torch.zeros(N,6, device=env.device))
    # External forces (world) -> optional rotation to local could be added
    f_ext = env.carpet_sensor.data.net_forces_w[:,0]
    feats.append(f_ext)
    # Dynamic kp
    if any(f.startswith('kp_') for f in feature_order):
        feats.append(env.dynamic_kp)
    X = torch.cat(feats, dim=1)
    return X


def _standardize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True) + 1e-6
    else:
        mean = torch.as_tensor(mean, device=X.device).view(1,-1)
        std = torch.as_tensor(std, device=X.device).view(1,-1)
    return (X-mean)/std, mean, std


def _compute_polish_pca_reward(env):
    """Compute PCA-aligned reward.

    Returns:
        reward (N,), terms dict
    """
    device = env.device
    pca_cfg = getattr(env, 'pca_cfg', None)

    feature_order = DEF_FEATURES if pca_cfg is None else pca_cfg.get('feature_order', DEF_FEATURES)
    X_raw = _gather_features(env, feature_order)

    # Map X_raw columns to feature_order (simple since we constructed aligned)
    if X_raw.shape[1] != len(feature_order):
        # truncate or pad (unlikely)
        if X_raw.shape[1] > len(feature_order):
            X_raw = X_raw[:, :len(feature_order)]
        else:
            pad = torch.zeros(X_raw.shape[0], len(feature_order)-X_raw.shape[1], device=device)
            X_raw = torch.cat([X_raw, pad], dim=1)

    # Standardize
    if pca_cfg is not None and 'scale_mean' in pca_cfg and 'scale_std' in pca_cfg:
        X, _, _ = _standardize(X_raw, pca_cfg['scale_mean'], pca_cfg['scale_std'])
    else:
        X, run_mean, run_std = _standardize(X_raw)

    # PC1 direction
    if pca_cfg is not None and 'pc1_loadings' in pca_cfg:
        pc1 = torch.as_tensor(pca_cfg['pc1_loadings'], device=device, dtype=torch.float32)
    else:
        pc1 = DEFAULT_PC1.to(device)
    pc1 = pc1 / (pc1.norm() + 1e-8)

    # Projection score
    proj = (X * pc1.view(1,-1)).sum(dim=1)  # (N,)
    # Sign normalization: make largest-magnitude loading positive for consistency
    if not hasattr(env, '_pca_sign_fixed'):
        max_idx = int(torch.argmax(torch.abs(pc1)))
        if pc1[max_idx] < 0:
            pc1 *= -1
            proj = -proj
        env._pca_sign_fixed = True

    # Target near zero (balanced forces pattern)
    target = getattr(env, 'pca_target', 0.0)
    proj_error = torch.abs(proj - target)

    # Core reward: high when |proj - target| small
    r_pca_alignment = torch.exp(-proj_error / 0.8) * 5.0

    # Penalize extreme magnitude along PC1 (regularization)
    r_pca_reg = -torch.clamp(torch.abs(proj) - 2.5, 0.0, 4.0) * 0.5

    # Simple phase shaping: only during CONTACT and DESCENT strongly influences
    phase = env.phase_ctrl.phase
    mask_contact = (phase == FixedPhase.CONTACT).float()
    mask_descent = (phase == FixedPhase.DESCENT).float()
    phase_scale = (0.3 + 0.7*mask_contact + 0.5*mask_descent)  # contact>descent>others
    r_pca = (r_pca_alignment + r_pca_reg) * phase_scale

    # Auxiliary mild penalties / bonuses
    # Encourage stable force (reuse pattern from existing reward if available)
    f_ext = env.carpet_sensor.data.net_forces_w[:,0]
    fz = f_ext[:,2]
    if hasattr(env, '_prev_fz_for_pca'):
        dz = torch.abs(fz - env._prev_fz_for_pca)
        r_force_stability = torch.exp(-dz/2.0) * 1.5 * mask_contact
        env._prev_fz_for_pca = fz.clone()
    else:
        r_force_stability = torch.zeros_like(fz)
        env._prev_fz_for_pca = fz.clone()

    # Light time penalty
    time_penalty = (env.episode_length_buf.float() / env.max_episode_length) * 0.2

    reward = r_pca + r_force_stability - time_penalty

    terms = {
        'r_pca_alignment': r_pca_alignment,
        'r_pca_reg': r_pca_reg,
        'r_pca_total': r_pca,
        'r_force_stability': r_force_stability,
        'proj_pc1': proj,
        'proj_error': proj_error,
        'phase': phase.float(),
        'time_penalty': time_penalty,
    }
    return reward, terms

# Public alias without underscore for external imports
compute_polish_pca_reward = _compute_polish_pca_reward

__all__ = [
    "_compute_polish_pca_reward",
    "compute_polish_pca_reward",
]

__all__ = ['_compute_polish_pca_reward']
