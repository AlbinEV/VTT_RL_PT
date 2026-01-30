"""
Unified Trajectory Management System - FIXED MODE

This package provides trajectory management for robotic polishing tasks
using Fixed Keypoints strategy with discrete waypoints:
- HOME → APPROACH → DESCENT → CONTACT → RISE

The Fixed mode uses programmatically defined waypoints for systematic
polishing tasks, rather than learned demonstrations.
"""

# Import from Fixed trajectory module (PRIMARY for this package)
from .Fixed_trajectory import (
    FixedPhase,
    TrajectoryKeypoints,
    FixedPhaseController,
    FixedTrajectoryManager,
    WaypointTrajectory,
    MANUAL_WP
)

# DEFAULT: FixedTrajectoryManager for Fixed Mode package
TrajectoryManager = FixedTrajectoryManager

# Define what gets imported with "from Trajectory_Manager import *"
__all__ = [
    # Fixed components (PRIMARY)
    'FixedPhase',
    'TrajectoryKeypoints',
    'FixedPhaseController',
    'FixedTrajectoryManager',
    'WaypointTrajectory',
    'MANUAL_WP',
    
    # Default (backward compatibility)
    'TrajectoryManager',
]

# Package metadata
__version__ = "3.0.0"
__author__ = "VTT Research Team"
