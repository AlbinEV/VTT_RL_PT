"""
Fixed Mode Polishing Environment for Isaac Lab
Variable Impedance Control with OSC Controller

This package provides a reinforcement learning environment for robotic polishing
tasks using Operational Space Control (OSC) with variable stiffness and damping.

Author: Albin Bajrami
Institution: VTT Technical Research Centre of Finland and Università Politecnica delle Marche
"""

import gymnasium as gym
from gymnasium.envs.registration import register

# Import environment and config
from .Polish_Env_OSC import PolishEnv, PolishEnvCfg

# Import agents config
from . import agents

# Register the environment with gymnasium
gym.register(
    id="Polish-Fixed-v0",
    entry_point="robo_pp_fixed.Polish_Env_OSC:PolishEnv",
    kwargs={
        "env_cfg_entry_point": "robo_pp_fixed.Polish_Env_OSC:PolishEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

__all__ = [
    "PolishEnv",
    "PolishEnvCfg",
]

__version__ = "1.0.0"
__author__ = "Albin Bajrami"
