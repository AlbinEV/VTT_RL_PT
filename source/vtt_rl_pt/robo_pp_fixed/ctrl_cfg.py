# ctrl_cfg.py
from __future__ import annotations
import torch
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.utils import configclass


# ───────────────────────────────────────
# CONFIGURATION (data only, no logic)
# ───────────────────────────────────────

@configclass
class DefaultOSCCfg(OperationalSpaceControllerCfg):
    
    # Task-space target types: absolute pose and absolute wrench
    target_types = ["pose_abs", "wrench_abs"]

    # Which axes of the pose to control (X, Y, Z, rotX, rotY, rotZ)
    motion_control_axes_task = [1, 1, 1, 1, 1, 1]

    # Which axes of the contact wrench to control
    contact_wrench_control_axes_task = [0, 0, 0, 0, 0, 0]

    # Whether to decouple inertial dynamics for motion control
    inertial_dynamics_decoupling = False
    """Whether to perform inertial dynamics decoupling for motion control (inverse dynamics)."""

    partial_inertial_dynamics_decoupling: bool = False
    """Whether to ignore the inertial coupling between the translational & rotational motions."""

    # Whether to include gravity compensation term
    gravity_compensation = True

    # Impedance mode:
    # - "fixed"      : constant stiffness & damping from the config
    # - "variable"   : controller expects both stiffness and damping inputs
    # - "variable_kp": controller expects only stiffness; uses default damping
    impedance_mode = "variable"

    kp_pos = 3000 # Proportional gain for position control
    kp_rot = 1500  # Proportional gain for rotation control
    # TODO: parametrize gains for force control

    # Default stiffness values per axis for fixed mode
    motion_stiffness_task = [3500, 1900, 5000, 460, 460, 410]  # Z=5000 from optimization
    contact_wrench_stiffness_task = [0, 0, 3500, 0, 0, 0]

    # Limits for stiffness when mode is "variable" or "variable_kp"
    motion_stiffness_limits_task = (0.0, 2*kp_pos)
    contact_stiffness_limits_task = (0.9*contact_wrench_control_axes_task[2], 1.1*contact_wrench_control_axes_task[2]) 

    # Default damping ratio values per axis for fixed mode
    motion_damping_ratio_task = [1.0, 1.0, 0.9, 1.0, 1.0, 1.0]  # zeta_z=0.9 from optimization
    contact_damping_ratio_task = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]

    # Limits for damping ratio when mode is "variable"
    motion_damping_ratio_limits_task = (0.01, 2.0)
    contact_damping_ratio_limits_task = (0.01, 2.0)

    # Null-space control method ("none" or "position" to drive redundant DOFs)
    nullspace_control = "position"

    # Null-space stiffness and damping ratio
    nullspace_stiffness = 50.0
    nullspace_damping_ratio = 1.0


# ───────────────────────────────────────
# WRAPPER (thin facade around the IsaacLab controller)
# ───────────────────────────────────────

class OSCWrapper:
    """
    Wrapper around IsaacLab's OperationalSpaceController.
    Keeps controller config and state isolated from the environment.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str = "cuda:0",
        cfg: OperationalSpaceControllerCfg | None = None
    ):
        # Use provided config or default one
        self.cfg = cfg # or DefaultOSCCfg()
        # Instantiate the actual controller
        self.impl = OperationalSpaceController(
            self.cfg,
            num_envs=num_envs,
            device=str(device)
        )

    def set_command(self, cmd: torch.Tensor, current_ee_pose_b: torch.Tensor):
        """
        Set the task-space targets and impedance parameters.

        For impedance_mode="variable":
          - cmd should be [pose_abs (7 dims), wrench_abs (6 dims),
                          stiffness (6 dims), damping (6 dims)] → total action_dim = 25

        For impedance_mode="variable_kp":
          - cmd should be [pose_abs, wrench_abs, stiffness] → action_dim = 19

        For impedance_mode="fixed":
          - cmd should be [pose_abs, wrench_abs] → action_dim = 13
        """
        self.impl.set_command(cmd, current_ee_pose_b=current_ee_pose_b)

    def compute(self, **kwargs) -> torch.Tensor:
        """
        Compute the joint efforts given the current state and command.

        Expected kwargs include:
          - jacobian_b: (num_envs, 6, dof)
          - current_ee_pose_b, current_ee_vel_b, current_ee_force_b
          - mass_matrix, gravity, current_joint_pos, current_joint_vel
          - nullspace_joint_pos_target

        Returns:
          - Tensor of shape (num_envs, dof), the computed joint efforts.
        """

        if hasattr(self, 'current_force_gains'):
          self.impl.cfg.contact_wrench_stiffness_task = self.current_force_gains

        return self.impl.compute(**kwargs)
