"""robot_cfg.py

Articulation configuration for the Franka Panda robot used in the polishing task.
Import as:
    from cfg.robot_cfg import panda_cfg
"""

# ---------- Imports ----------
import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

# ---------- Constants ----------
# Get path relative to this file's location
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(_THIS_DIR, "..", "assets")

# ---------- Robot Configuration ----------

panda_cfg: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",  # Regex-friendly prim path template
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/panda_vtt.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.3760,
            "panda_joint2": 0.2049,
            "panda_joint3": 0.1918,
            "panda_joint4": -1.9632,
            "panda_joint5": -0.04678,
            "panda_joint6": 2.16445,
            "panda_joint7": 1.37637,
        }
    ),
    actuators={
        # Actuator groups with effort/velocity limits and PD gains
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=0,
            damping=0.000,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=0.0,
            damping=0.00,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["EE_01"],
            effort_limit=12.0,
            velocity_limit=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

# Public interface
__all__ = ["panda_cfg"]
