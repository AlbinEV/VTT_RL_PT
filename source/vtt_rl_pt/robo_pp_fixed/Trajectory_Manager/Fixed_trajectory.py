"""
Fixed Trajectory Management Module

This module provides trajectory management for fixed, predefined polishing sequences.
Instead of learning from demonstration, this system uses programmatically defined
keypoints for systematic polishing tasks.

Components:
- FixedPhaseController: Manage 5-phase polishing state machine (home, approach, descent, contact, shift, rise)
- FixedTrajectoryManager: Handle fixed keypoint trajectories with randomization
- TrajectoryKeypoints: Define and generate keypoint sequences

This module handles a complete fixed workflow:
1. Start from home position
2. Move to approach position above surface
3. Descend until contact is detected
4. Shift laterally while maintaining contact
5. Rise to completion height
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from enum import IntEnum





MANUAL_WP = [
    # x,     y,     z,     qw,  qx,  qy,  qz
    [0.55,  0.00,  0.55,   0,   1,   0,   0],   # home
    [0.55,  0.00,  0.10,   0,   1,   0,   0],   # sopra tavolo
    [0.55,  0.00,  0.00,   0,   1,   0,   0],   # contatto
    [0.55,  0.2,  0.00,   0,   1,   0,   0],   # striscia a destra
    [0.55, -0.2,  0.00,   0,   1,   0,   0],   # striscia a sinistra
    [0.55,  0.00,  0.15,   0,   1,   0,   0],   # risalita
    ]

class FixedPhase(IntEnum):
    """Fixed trajectory phases enumeration"""
    HOME = 0       # Moving to home position
    APPROACH = 1   # Moving to approach position above surface
    DESCENT = 2     # Descending until contact
    CONTACT = 3    # In contact, performing lateral motion
    RISE    = 4       # Rising to completion



class WaypointTrajectory:
    """
    Simple waypoint trajectory manager for fixed, predefined waypoints.
    
    This class provides a basic implementation for manual waypoint sequences
    without the complexity of the full FixedTrajectoryManager.
    """
    def __init__(self, waypoints: Optional[list[list[float]]] = None,
                 device: str = "cuda:0"):
        if waypoints is None:
            waypoints = MANUAL_WP     # fallback di default
        
        # Store device
        self.device = device
        # Initialize trajectories on the specified device
        wp = torch.as_tensor(waypoints, dtype=torch.float32, device=self.device)
        self.p_traj = wp[:, :3]  # positions, shape [T,3]
        self.q_traj = wp[:, 3:]  # orientations (quaternions), shape [T,4]
        self.T = wp.shape[0]     # number of waypoints
        
        # Prepare per-env buffers; initially only one env
        self.p_traj_env = self.p_traj.unsqueeze(0)  # (1, T, 3)
        self.q_traj_env = self.q_traj.unsqueeze(0)  # (1, T, 4)

    def sample(self, env_ids: torch.Tensor):
        """Replicates the base trajectory for each environment."""
        batch_size = env_ids.shape[0]
        # Repeat base trajectory for all envs
        p = self.p_traj.unsqueeze(0).repeat(batch_size, 1, 1)  # [N_envs, T, 3]
        q = self.q_traj.unsqueeze(0).repeat(batch_size, 1, 1)  # [N_envs, T, 4]
        # Move to correct device
        p = p.to(self.device)
        q = q.to(self.device)
        # Update per-env buffers
        self.p_traj_env = p
        self.q_traj_env = q

        
    def get_targets(self, env_ids: torch.Tensor, wpt_idx: torch.Tensor):
        # If the per-env buffer hasn’t been expanded yet, do it now:
        if self.p_traj_env.shape[0] != env_ids.shape[0]:
            # env_ids must be a CPU LongTensor for sample()
            self.sample(env_ids.to("cpu"))

        # clamp and bring everything to the same device
        max_idx = self.T - 1
        wpt_idx_safe = torch.clamp(wpt_idx, 0, max_idx)
        device = self.p_traj_env.device
        env_ids = env_ids.to(device)
        wpt_idx_safe = wpt_idx_safe.to(device)

        # safe row-wise indexing
        p_env = self.p_traj_env[env_ids]    # now shape [N_envs, T, 3]
        q_env = self.q_traj_env[env_ids]    #               [N_envs, T, 4]
        batch = torch.arange(env_ids.shape[0], device=device)
        target_p = p_env[batch, wpt_idx_safe]
        target_q = q_env[batch, wpt_idx_safe]
        return target_p, target_q


    
    def get_trajectory_info(self):
        # Always inspect the first environment's trajectory
        traj = self.p_traj_env[0]  # shape [T, dim]
        p_min = traj.min(dim=0)[0]
        p_max = traj.max(dim=0)[0]
        return {
            "p_min": p_min.cpu().tolist(),
            "p_max": p_max.cpu().tolist(),
            "T": traj.shape[0],
        }

    
class TrajectoryKeypoints:
    """
    Defines and generates keypoint sequences for fixed polishing trajectories.
    
    This class creates systematic polishing trajectories with precise phase control:
    1. HOME: Start from robot's natural spawn position
    2. APPROACH: Move to position above contact surface
    3. DESCENT: Lower vertically until contact is detected
    4. CONTACT: Execute lateral polishing motion while maintaining contact
    5. RISE: Lift vertically to end the polishing task
    """
    
    def __init__(self,
                 home_position: List[float] = [0.45, 0.0, 0.55],       # Default fallback - should be overridden
                 approach_height: float = 0.05,
                 contact_surface_z: float = 0.0,
                 lateral_distance: float = 0.15,
                 rise_height: float = 0.1,
                 approach_position: Optional[List[float]] = None,    # Optional explicit approach position
                 orientation: List[float] = [0.0, 1.0, 0.0, 0.0]):     # Z pointing downward (180° X-axis rotation)
        """
        Initialize trajectory keypoints configuration.
        
        Args:
            home_position: [x, y, z] home position matching robot spawn position
            approach_height: Height above surface for approach (m)
            contact_surface_z: Z coordinate of contact surface (m)
            lateral_distance: Distance to move laterally during polishing (m)
            rise_height: Height to rise after completion (m)
            approach_position: Optional explicit approach position, if None will be computed from home_position
            orientation: Fixed quaternion [w,x,y,z] for end-effector
        """
        # Ensure home_position is not None
        if home_position is None:
            print("[WARNING] home_position was None in TrajectoryKeypoints! Using fallback [0.3, 0.0, 0.5]")
            home_position = [0.3, 0.0, 0.5]
        
        self.home_pos = np.array(home_position)
        print(f"[INFO] TrajectoryKeypoints using home position: {self.home_pos}")
        
        self.approach_height = approach_height
        self.contact_z = contact_surface_z
        self.lateral_dist = lateral_distance
        self.rise_height = rise_height
        self.orientation = np.array(orientation)
        
        # Center position for polishing 
        # Important! Use specified approach position or derive from home position
        if approach_position:
            # Use explicit approach position (x,y), but with contact_z for consistency
            self.center_pos = np.array([approach_position[0], approach_position[1], contact_surface_z])
            print(f"[INFO] Using explicit approach position: {self.center_pos}")
        else:
            # Keep same X as home, Y=0 for centered approach
            self.center_pos = np.array([home_position[0], 0.0, contact_surface_z])
            print(f"[INFO] Using derived approach position: {self.center_pos}")
    
    def generate_keypoints(self, randomize: bool = False) -> List[List[float]]:
        """
        Generate sequence of keypoints for fixed polishing trajectory.
        
        Creates a complete path:
        1. Start at home position
        2. Move to approach position above surface
        3. Create descent point on surface
        4. Generate multiple lateral waypoints for polishing motion
        5. Add final rise point for task completion
        
        Args:
            randomize: If True, apply small random variations to positions
            
        Returns:
            List of keypoints, each [x, y, z, qw, qx, qy, qz]
        """
        keypoints = []
        
        # Print complete debug information
        print(f"\n{'='*60}")
        print(f"GENERATING KEYPOINTS:")
        print(f"  Home position: {self.home_pos}")
        print(f"  Approach height: {self.approach_height}")
        print(f"  Contact surface Z: {self.contact_z}")
        print(f"  Center position: {self.center_pos}")
        print(f"  Orientation: {self.orientation}")
        print(f"{'='*60}\n")
        
        # Phase 0: HOME - Start from robot's natural spawn position
        home = np.concatenate([self.home_pos, self.orientation])
        keypoints.append(home.tolist())
        print(f"[INFO] HOME keypoint: {home.tolist()}")
        
        # Add only ONE intermediate point for direct and controlled movement
        intermediate_pos = 0.5 * self.home_pos + 0.5 * np.array([
            self.center_pos[0], 
            self.center_pos[1], 
            self.center_pos[2] + self.approach_height
        ])
        intermediate = np.concatenate([intermediate_pos, self.orientation])
        keypoints.append(intermediate.tolist())
        print(f"[INFO] INTERMEDIATE keypoint: {intermediate.tolist()}")
        
        # Phase 1: APPROACH - Move to position directly above contact point
        approach_pos = np.array([self.center_pos[0], self.center_pos[1], self.center_pos[2] + self.approach_height])
        approach = np.concatenate([approach_pos, self.orientation])
        keypoints.append(approach.tolist())
        print(f"[INFO] APPROACH keypoint: {approach.tolist()}")
        
        # Phase 2: DESCENT - Contact position (at surface level)
        contact_pos = self.center_pos.copy()
        contact = np.concatenate([contact_pos, self.orientation])
        keypoints.append(contact.tolist())
        print(f"[INFO] CONTACT keypoint: {contact.tolist()}")
        
        # Phase 3: CONTACT - Lateral shift positions for polishing motion
        # Use just 3 points for lateral movement
        num_lateral_points = 3
        for i in range(num_lateral_points):
            y_offset = (i / (num_lateral_points - 1)) * self.lateral_dist - self.lateral_dist / 2
            lateral_pos = self.center_pos.copy()
            lateral_pos[1] += y_offset
            lateral = np.concatenate([lateral_pos, self.orientation])
            keypoints.append(lateral.tolist())
            print(f"[INFO] LATERAL {i} keypoint: {lateral.tolist()}")
        
        # Phase 4: RISE - Lift to completion height
        rise_pos = self.center_pos.copy()
        rise_pos[2] += self.rise_height
        rise = np.concatenate([rise_pos, self.orientation])
        keypoints.append(rise.tolist())
        print(f"[INFO] RISE keypoint: {rise.tolist()}")
        print(f"[INFO] Total keypoints: {len(keypoints)}")
        
        # Apply randomization if requested
        if randomize:
            keypoints = self._apply_randomization(keypoints)
        
        # Safety check for large jumps between waypoints
        for i in range(1, len(keypoints)):
            prev_pos = np.array(keypoints[i-1][:3])
            curr_pos = np.array(keypoints[i][:3])
            dist = np.linalg.norm(curr_pos - prev_pos)
            if dist > 0.5:  # If waypoints are more than 50cm apart
                print(f"[WARNING] Large distance ({dist:.3f}m) detected between waypoints {i-1} and {i}")
        
        return keypoints
    
    def _apply_randomization(self, keypoints: List[List[float]]) -> List[List[float]]:
        """Apply small random variations to keypoints for training diversity."""
        randomized = []
        for point in keypoints:
            pos = np.array(point[:3])
            quat = np.array(point[3:])
            
            # Add small XY randomization (±2cm)
            pos[:2] += np.random.uniform(-0.02, 0.02, 2)
            
            # Keep Z and orientation unchanged for safety
            randomized_point = np.concatenate([pos, quat])
            randomized.append(randomized_point.tolist())
        
        return randomized


class FixedPhaseController:
    """
    Controls the five-phase fixed trajectory polishing sequence.

    Phases:
    - 0: HOME - Move to home position
    - 1: APPROACH - Move to approach position above surface
    - 2: DESCENT - Descend until contact is detected
    - 3: CONTACT - Maintain contact while following lateral trajectory
    - 4: RISE - Rise to completion height

    This controller provides force feedback during contact phase and
    automatic phase transitions based on position reaching and force detection.
    """
    
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        F_TOUCH: float = -0.1,
        F_LOST: float = 0.1,
        DESCENT_DZ: float = 0.001,  # 1mm steps for gentle descent
        DESCENT_VEL: float = 0.01,  # 1cm/s descent speed
        ADM_KP: float = 0.002,
        ADM_KI: float = 0.0005,
        ADM_DZ_CLAMP: float = 0.003
    ):
        # Force thresholds
        self.F_TOUCH = F_TOUCH
        self.F_LOST = F_LOST
        self.DESCENT_DZ = DESCENT_DZ
        self.DESCENT_VEL = DESCENT_VEL
        self.ADM_KP = ADM_KP
        self.ADM_KI = ADM_KI
        self.ADM_DZ_CLAMP = ADM_DZ_CLAMP
        
        # State buffers
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._z_hold = torch.zeros(num_envs, device=device)
        self._int_F_err = torch.zeros(num_envs, device=device)
        self._target_positions = torch.zeros((num_envs, 3), device=device)
        self._rise_target_z = torch.zeros(num_envs, device=device)
        
        # Phase transition tolerances
        self.position_tolerance = 0.01  # 1cm
        self.orientation_tolerance = 0.1  # radians

    def reset(self, env_ids: torch.Tensor):
        """Reset phase controller state for specified environments."""
        self.phase[env_ids] = FixedPhase.HOME
        self._z_hold[env_ids] = 0.0
        self._int_F_err[env_ids] = 0.0
        self._target_positions[env_ids] = 0.0
        self._rise_target_z[env_ids] = 0.0

    def apply(self,
              p_des: torch.Tensor,
              ee_pos: torch.Tensor,
              fz: torch.Tensor,
              dt: float,
              wpt_idx: torch.Tensor) -> torch.Tensor:
        """
        Apply phase control logic to modify desired positions.
        
        Args:
            p_des: (N, 3) desired positions
            ee_pos: (N, 3) actual end-effector positions
            fz: (N,) contact forces in Z direction
            dt: Time step
            wpt_idx: (N,) current waypoint indices
            
        Returns:
            Modified desired positions with phase control applied
        """
        # Apply force control only during CONTACT phase
        contact_mask = (self.phase == FixedPhase.CONTACT)
        
        if contact_mask.any():
            # Target force for polishing (-2.0 N downward)
            target_force = -100
            force_error = target_force - fz[contact_mask]
            
            # PI controller for force
            self._int_F_err[contact_mask] += force_error * dt
            self._int_F_err[contact_mask] = torch.clamp(
                self._int_F_err[contact_mask], -15.0, 15.0
            )
            
            # Calculate Z adjustment
            dz = self.ADM_KP * force_error + self.ADM_KI * self._int_F_err[contact_mask]
            dz = torch.clamp(dz, -self.ADM_DZ_CLAMP, self.ADM_DZ_CLAMP)
            
            # Apply force control to Z coordinate
            p_des[contact_mask, 2] = self._z_hold[contact_mask] + dz
        
        # Apply descent control during DESCENT phase
        descent_mask = (self.phase == FixedPhase.DESCENT)
        if descent_mask.any():
            p_des[descent_mask, 2] = ee_pos[descent_mask, 2] - self.DESCENT_VEL * dt

        # Apply RISE behavior: move up gently towards rise target
        rising_mask = (self.phase == FixedPhase.RISE)
        if rising_mask.any():
            # Use descent velocity as baseline; ascend faster (x2)
            climb = self.DESCENT_VEL * 2.0 * dt
            p_des[rising_mask, 2] = torch.minimum(
                ee_pos[rising_mask, 2] + climb,
                self._rise_target_z[rising_mask]
            )

        return p_des

    def set_z_hold(self, env_ids: torch.Tensor, z_values: torch.Tensor):
        """Set the Z hold position for specified environments."""
        self._z_hold[env_ids] = z_values

    def reset_integral_error(self, env_ids: torch.Tensor):
        """Reset the integral force error for specified environments."""
        self._int_F_err[env_ids] = 0.0

    def set_rise_target(self, env_ids: torch.Tensor, target_z: torch.Tensor):
        """Set target Z height for RISE phase for specified environments."""
        self._rise_target_z[env_ids] = target_z


class FixedTrajectoryManager:
    """
    Manages fixed keypoint trajectories with per-environment variations.
    
    This class generates systematic polishing trajectories using predefined
    keypoints rather than learned demonstrations. It supports randomization
    for training diversity while maintaining the core task structure.
    """
    
    def __init__(self,
                 num_envs: int,
                 device: torch.device | str = "cuda:0",
                 keypoint_config: Optional[Dict[str, Any]] = None):
        """
        Initialize fixed trajectory manager.
        
        Args:
            num_envs: Number of parallel environments
            device: PyTorch device for computations
            keypoint_config: Optional configuration for keypoint generation
        """
        self.num_envs = num_envs
        self.device = device
        
        # Initialize keypoint generator
        config = keypoint_config or {}
        self.keypoint_gen = TrajectoryKeypoints(**config)
        
        # Generate base trajectory
        base_keypoints = self.keypoint_gen.generate_keypoints(randomize=False)
        base_goals = torch.as_tensor(base_keypoints, dtype=torch.float32, device=device)
        
        # Separate positions and orientations
        self.p_traj = base_goals[:, :3]    # (T, 3)
        self.q_traj = base_goals[:, 3:]    # (T, 4)
        self.T = self.p_traj.shape[0]
        
        # Per-environment trajectory buffers
        self.p_traj_env = torch.zeros((num_envs, self.T, 3), device=device)
        self.q_traj_env = torch.zeros((num_envs, self.T, 4), device=device)
        
        # Randomization parameters
        self.xy_randomization_range = 0.03  # ±3cm for fixed trajectories

    def sample(self, env_ids: torch.Tensor):
        """Generate randomized trajectories for specified environments."""
        # Apply randomization to create training diversity
        xy_offset = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) * self.xy_randomization_range
        offset_3d = torch.zeros((len(env_ids), 3), device=self.device)
        offset_3d[:, :2] = xy_offset
        
        # Apply offset to position trajectory
        self.p_traj_env[env_ids] = self.p_traj.unsqueeze(0) + offset_3d.unsqueeze(1)
        
        # Orientation remains fixed
        self.q_traj_env[env_ids] = self.q_traj.unsqueeze(0)

    def get_targets(self,
                    env_ids: torch.Tensor,
                    wpt_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current position and orientation targets."""
        # CRITICAL SAFETY CHECK: Ensure wpt_idx is valid (in case of errors)
        valid_wpt_idx = torch.clamp(wpt_idx, 0, self.T - 1)
        
        # Check if any indices were clamped, which would indicate an error
        if not torch.all(wpt_idx == valid_wpt_idx):
            print(f"[WARNING] Invalid waypoint indices detected: {wpt_idx[wpt_idx != valid_wpt_idx].tolist()}")
            print(f"[WARNING] Clamped to valid range: 0-{self.T - 1}")
            wpt_idx = valid_wpt_idx
        
        # Debug the current target positions
        # ensure all tensors are on the same device
        device = self.p_traj_env.device
        env_ids = env_ids.to(device)
        wpt_idx = wpt_idx.to(device)

        # first slice out each env’s whole trajectory
        p_env = self.p_traj_env[env_ids]       # shape [N, T, 3]
        q_env = self.q_traj_env[env_ids]       # shape [N, T, 4]

        # then pick the per-row waypoint
        batch     = torch.arange(env_ids.shape[0], device=device)
        target_p  = p_env[batch, wpt_idx]      # [N,3]
        target_q  = q_env[batch, wpt_idx]      # [N,4]
        if wpt_idx[0] == 0:
            # print(f"[DEBUG] First target position: {target_p[0].cpu().numpy().tolist()}")
            pass
        
        # return (
        #     target_p,
        #     self.q_traj_env[env_ids, wpt_idx]
        # )

        return (
            target_p,
            target_q
        )

    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get trajectory metadata."""
        p_min = self.p_traj.min(dim=0)[0]
        p_max = self.p_traj.max(dim=0)[0]
        
        return {
            "num_waypoints": self.T,
            "trajectory_type": "fixed_keypoints",
            "spatial_bounds": {
                "min": p_min.cpu().numpy().tolist(),
                "max": p_max.cpu().numpy().tolist(),
                "range": (p_max - p_min).cpu().numpy().tolist()
            },
            "phases": [phase.name for phase in FixedPhase],
            "xy_randomization": self.xy_randomization_range
        }

    def reset_to_start(self, env_ids: torch.Tensor):
        """Reset environments to start of fixed trajectory."""
        self.sample(env_ids)


# Alias for backward compatibility
TrajectoryManager = FixedTrajectoryManager

# Export classes for external use
__all__ = [
    'FixedPhase',
    'TrajectoryKeypoints',
    'FixedPhaseController',
    'FixedTrajectoryManager',
    'TrajectoryManager',
    'WaypointTrajectory',
    'MANUAL_WP'
]
