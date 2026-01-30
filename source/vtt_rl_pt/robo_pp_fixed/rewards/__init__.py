"""
Rewards module for Fixed Mode polishing with variable impedance control.
Thesis Section 4.6 - Phase-dependent reward structure.
"""
from ..cfg.config import F_TOUCH, F_LOST

# Import reward functions
from .polish_kpz_control_reward import compute_polish_kpz_control_reward
from .polish_kz_dz_control_reward import compute_polish_kz_dz_control_reward

# Public mapping of reward functions
REWARD_FUNCS = {
    "kpz": compute_polish_kpz_control_reward,      # Kz-only control
    "kz_dz": compute_polish_kz_dz_control_reward,  # Kz + damping ratio control
}

# Default reward function
compute_reward = compute_polish_kpz_control_reward

__all__ = [
    "F_TOUCH",
    "F_LOST",
    "compute_reward",
    "REWARD_FUNCS",
    "compute_polish_kpz_control_reward",
    "compute_polish_kz_dz_control_reward",
]
