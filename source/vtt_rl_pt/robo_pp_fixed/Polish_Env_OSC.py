# ---------- Imports ----------

# Standard library imports
from __future__ import annotations  # Postpone evaluation of type annotations until needed

# Third-party dependencies
import numpy as np  # NumPy for numerical array operations
import torch  # PyTorch for tensor computations and GPU acceleration
from gymnasium import spaces  # Gymnasium for defining action/observation spaces
import carb  # NVIDIA Omniverse CARB utilities (logging, warnings, etc.)
from isaaclab.utils import configclass

# IsaacLab simulation utilities
import isaaclab.sim as sim_utils  # Core simulation helper functions and configurations
from isaaclab.sim import SimulationCfg, PhysxCfg  # Simulation and PhysX configuration classes
from isaaclab.sim import DomeLightCfg  # Dome lighting configuration for environment
from isaaclab.scene import InteractiveSceneCfg  # Base class for interactive scene configurations
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # Direct RL environment and its configuration
from isaaclab.assets import Articulation, ArticulationCfg  # Articulation (robot) asset classes
from isaaclab.assets import AssetBaseCfg  # Base asset configuration for static objects
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane  # Ground plane spawning utilities
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg  # Implicit actuator (joint) configurations
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg  # Rigid body physics material properties
from isaaclab.sim.spawners.lights import spawn_light, SphereLightCfg  # Light spawners and sphere light configs
from isaaclab.terrains import TerrainImporterCfg  # Terrain importer configuration
from isaaclab.sensors import ContactSensorCfg  # Contact sensor configuration

# Controller utilities
from .ctrl_cfg import OSCWrapper, DefaultOSCCfg, OperationalSpaceControllerCfg
# Trajectory manager (LfD/Fixed) lives under Trajectory_Manager package
from .Trajectory_Manager import TrajectoryManager, FixedPhaseController, FixedPhase  # Fixed mode trajectory

import os

# Marker visualization utilities
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers  # Marker config and visualization
from isaaclab.markers.config import FRAME_MARKER_CFG  # Default frame marker configuration

from .cfg.robot_cfg import panda_cfg
from .cfg.scene_cfg import JustPushSceneCfg
# Reward module
from .rewards import REWARD_FUNCS, compute_polish_kpz_control_reward, compute_polish_kz_dz_control_reward

from dataclasses import field
# LfD not used in Fixed mode - using FixedTrajectoryManager with discrete waypoints


# ======= SWITCHES ========
JOINT_TORQUE   = False   # 7‑D additive torque
KP_DELTA       = True    # 6‑D ΔKp  (già presente)
DAMPING_DELTA  = False   # 6‑D Δζ   (nuovo)
OBS_DAMPING    = True   # se True includi ζ nell'osservazione
SEQ_LEN = 128                  # PolishEnvCfg.seq_length di default

# quali indici del vettore [X Y Z Rx Ry Rz] sono allenabili
TRAIN_AXES = [2]            # solo asse Z di traslazione per controllo forza


TORQUE_SCALE = 0.1
KPZ_MIN = 500.0
KPZ_MAX = 3000.0
KPZ_DELTA_SCALE = 0.001
ZETA_DELTA_SCALE = 0.005

DESCENT_DZ = 0.05       # 1 cm; cambia a piacere
F_TOUCH    = -0.01       # N  → identifica il primo contatto
F_LOST     = -0.01       # N  → isteresi per perdita contatto

# === FORCE‑CONTROL GAINS (asse Z) ===
ADM_KP   = 0.002    # m/N  ->  2 mm di spostamento per 1 N di errore
ADM_KI   = 0.0005   # m/(N·s)  integrale (stabilizza il bias)
ADM_DZ_CLAMP = 0.003  # ±3 mm di escursione massima per step

RISE_DZ = 0.10          # 10 cm

# =========================

def _make_action_space() -> spaces.Box:
    dim = 0
    if KP_DELTA or DAMPING_DELTA:
        dim = len(TRAIN_AXES) * (KP_DELTA + DAMPING_DELTA)   # 1 per kp, 1 per ζ
    if JOINT_TORQUE:
        dim += 7
    return spaces.Box(-1.0, 1.0, shape=(dim,), dtype=np.float32)


def _make_observation_space() -> spaces.Box:
    one_step = 14 + 1 + 6 + (6 if (DAMPING_DELTA or OBS_DAMPING) else 0)
    return spaces.Box(-np.inf, np.inf,
                      shape=(one_step * SEQ_LEN,), dtype=np.float32)

## This part to move in a Util function
def print_joint_debug(q_rad: torch.Tensor,
                      q_lo_deg, q_hi_deg,
                      names=None,
                      width: int = 40):
    """
    Stampa su stdout una barra orizzontale che visualizza la posizione
    di ogni giunto fra il suo limite minimo e massimo.

    Args
    ----
    q_rad : (N,) tensor con la posa corrente [rad]
    q_lo_deg, q_hi_deg : sequence di float, limiti [deg]
    names : lista di label (facoltativo)
    width : numero di caratteri nella barra fra i delimitatori '|'
    """
    q_deg = q_rad.cpu().numpy() * 180.0 / 3.141592653589793
    for i, (val, lo, hi) in enumerate(zip(q_deg, q_lo_deg, q_hi_deg)):
        lo, hi = float(lo), float(hi)
        name = names[i] if names else f"J{i+1}"
        # normalizza val ∈ [0, width]
        span  = hi - lo if hi > lo else 1e-6
        pos   = int(round((val - lo) / span * width))
        pos   = max(0, min(width, pos))

        # costruisci barra: 'x' a sinistra, '-' a destra
        bar_left  = "x" * pos
        bar_right = "-" * (width - pos)
        bar = bar_left + bar_right

        # sovrascrivi qualche char al centro con il valore numerico
        sval = f"{val:+6.1f}"
        idx  = max(0, min(width - len(sval), pos - len(sval)//2))
        bar = bar[:idx] + sval + bar[idx + len(sval):]

        print(f"{name:<10s}{lo:>7.1f} |{bar}| {hi:>7.1f}")


# ------------------------------------------------------------------ #
#  Barra per Kp / ζ per ciascun asse task‑space [X Y Z Rx Ry Rz]
# ------------------------------------------------------------------ #
def print_impedance_debug(kp, zeta, kp_lo, kp_hi,
                          axes=("X","Y","Z","Rx","Ry","Rz"), width=30):
    kp   = kp.cpu().tolist()
    zeta = zeta.cpu().tolist()

    def _bar(val, lo, hi):
        span = hi - lo if hi > lo else 1e-6
        pos  = int(round((val - lo) / span * width))
        pos  = max(0, min(width, pos))
        return "x"*pos + "-"*(width-pos)

    print("\n─ Impedance (env‑0) ───────────────────────────────────────────")
    for i, axis in enumerate(axes):
        bar  = _bar(kp[i], kp_lo, kp_hi)
        sval = f"{kp[i]:4.1f}"
        mid  = width//2 - len(sval)//2
        bar  = bar[:mid] + sval + bar[mid+len(sval):]
        print(f"Kp {axis:<2}: |{bar}|")

    for i, axis in enumerate(axes):
        bar  = _bar(zeta[i], 0.0, 2.0)
        sval = f"{zeta[i]:4.2f}"
        mid  = width//2 - len(sval)//2
        bar  = bar[:mid] + sval + bar[mid+len(sval):]
        print(f"ζ  {axis:<2}: |{bar}|")
    print("────────────────────────────────────────────────────────────────\n")



@configclass
class PolishEnvCfg(DirectRLEnvCfg):
    """Configuration for the Polish reinforcement learning environment."""
    episode_length_s: float = 120.0  # Seconds per episode
    decimation: int = 10             # Physics substep decimation
    debug_interval: int = 0          # Debug print interval (0=disabled, 50=every 50 steps)
    seq_length: int = 128           # Sequence length for observations/actions
    reward_type: str = "kpz"        # Reward function: "kpz" (Kz-only) or "kz_dz" (Kz+damping)

    action_space:      spaces.Box = field(default_factory=_make_action_space)
    observation_space: spaces.Box = field(default_factory=_make_observation_space)
    state_space:       spaces.Box = field(init=False)

    # # HARDCODED: adjust if dims change
    # action_space: spaces.Box = spaces.Box(
    #     low=-0.0, high=0.0, shape=(13,), dtype=np.float64
    # )
    # observation_space: spaces.Box = spaces.Box(
    #     low=-np.inf, high=np.inf, shape=(self.seq_length * self.obs_dim,), dtype=np.float64
    # )
    # state_space: spaces.Box = observation_space  # Alias to observation_space

    scene: JustPushSceneCfg = JustPushSceneCfg(
        num_envs=1, env_spacing=2.0, replicate_physics=True
    )

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0", dt=1/100, gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=32,
            max_velocity_iteration_count=1,
            gpu_max_rigid_contact_count=2**20
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=float(os.environ.get("STATIC_FRICTION", "0.0")),
            dynamic_friction=float(os.environ.get("DYNAMIC_FRICTION", "0.0"))
        )
    )
    
    robot: ArticulationCfg = panda_cfg
    ctrl: OperationalSpaceControllerCfg = DefaultOSCCfg()

    def __post_init__(self):
        # alias per compatibilità
        self.state_space = self.observation_space

# ---------- Environment Class ----------

class PolishEnv(DirectRLEnv):
    """Reinforcement learning environment for polishing task using Panda robot."""
    def __init__(self, cfg: PolishEnvCfg, render_mode: str, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._fz_ema   = torch.zeros(self.num_envs, device=self.device)
        self._ema_alpha = 0.1         # 0.1‑0.3 di solito basta
        self._int_F_err = torch.zeros(self.num_envs, device=self.device)


        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Imposta la modalità del controller in base ai canali azione
        if DAMPING_DELTA:
            self.cfg.ctrl.impedance_mode = "variable"
        elif KP_DELTA:
            self.cfg.ctrl.impedance_mode = "variable_kp"
        else:
            self.cfg.ctrl.impedance_mode = "fixed"

        # after: explicitly float
        init_kp = torch.tensor(
            self.cfg.ctrl.motion_stiffness_task,
            dtype=torch.float32,
            device=self.device
        )
        self.dynamic_kp = init_kp.unsqueeze(0).repeat(self.num_envs, 1)

        # ----- default impedance for resets -----
        self.init_kp   = init_kp.clone()
        self.init_zeta = torch.tensor(
            self.cfg.ctrl.motion_damping_ratio_task,
            dtype=torch.float32, device=self.device
        )
        self._rise_target_z = torch.zeros(self.num_envs, device=self.device)

        # -------- damping ratio buffer ---------------------------------
        if DAMPING_DELTA or OBS_DAMPING:
            self.dynamic_zeta = torch.tensor(
                self.cfg.ctrl.motion_damping_ratio_task,
                dtype=torch.float32, device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)   # (N,6)
        else:
            # definita comunque per evitare AttributeError nelle stampe
            self.dynamic_zeta = torch.tensor(
                self.cfg.ctrl.motion_damping_ratio_task,
                dtype=torch.float32, device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)


        self.ctrl_dim = self.dynamic_kp.shape[1]   # =6

        self.wpt_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_target_change_time = torch.zeros(self.num_envs, dtype=torch.float32,
                                                device=self.device)
        
        # now we know action_dim == 7
        self.tau_history  : list[list[float]] = []
        self.log_path      = "./logs/tau_history.csv"
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Print available robot links
        print("LINKS disponibili:")
        for i, name in enumerate(self.robot.body_names):
            print(f"{i:2d}: {name}")

        self.cube_sensor = self.scene["cube_contact"]
        self.carpet_sensor = self.cube_sensor  # Alias for reward module compatibility
        self.frame = 0
        self.seq_length = cfg.seq_length
        # self.max_episode_length = int(cfg.episode_length_s / self.physics_dt)
        self.wp_eps = 0.01          # distanza (m) per dire “way-point raggiunto”
        self.wp_ori_eps = 0.5       # angolo (rad) per dire “orientamento raggiunto”
        # ---- bersaglio di forza (N) e tolleranza ----------
        self.fz_target = -20.0          # spingi verso -Z con 20 N
        self.fz_eps    = 2.0           # ±2.0 N di finestra accettabile


        self._last_ori_err = torch.full((self.num_envs,), float("inf"), device=self.device)
        self._last_pos_err = torch.full((self.num_envs,), float("inf"), device=self.device)

        # Identify joint names and indices
        joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.joint_ids, _ = self.robot.find_joints(joint_names)
        self.action_dim = len(self.joint_ids)

        shoulder_lim = self.robot.actuators["panda_shoulder"].cfg.effort_limit  # 87 Nm
        forearm_lim  = self.robot.actuators["panda_forearm"].cfg.effort_limit   # 12 Nm

        self.torque_limit = torch.tensor(
            [shoulder_lim]*4 + [forearm_lim]*3,     # joint1‑4, joint5‑7
            dtype=torch.float32, device=self.device)

        # End-effector body and jacobian indices
        bodies, _ = self.robot.find_bodies(["EE"])
        self.ee_body_idx = bodies[0]
        self.ee_jac_idx = bodies[0]-1

        # Override spaces now that dims are known
        # Convert stiffness limits to tensors (6 axes)
        _kp_lo, _kp_hi = self.cfg.ctrl.motion_stiffness_limits_task
        self.kp_lo = torch.full((6,), _kp_lo, device=self.device)
        self.kp_hi = torch.full((6,), _kp_hi, device=self.device)
        self.kp_lo[2] = KPZ_MIN
        self.kp_hi[2] = KPZ_MAX
        self.dynamic_kp.clamp_(self.kp_lo, self.kp_hi)
        self.init_kp = self.dynamic_kp.clone()
        # self.policy_act_dim    = self.action_dim + self.ctrl_dim          # 13
        # self.action_space      = spaces.Box(low=-1., high=1.,
                                            # shape=(self.policy_act_dim,), dtype=np.float32)
        # self.obs_dim = 2 * self.action_dim + 1 + self.ctrl_dim
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf,
        #     shape=(self.seq_length * self.obs_dim,), dtype=np.float32
        # )

        # ----- action dim -----
        # act_dim = 0
        # if JOINT_TORQUE:  act_dim += 7
        # if KP_DELTA:      act_dim += 6
        # if DAMPING_DELTA: act_dim += 6
        # self.policy_act_dim = act_dim
        # self.action_space = spaces.Box(-1., 1., shape=(act_dim,), dtype=np.float32)

        # ----- observation dim -----
        # obs_dim = 0
        # obs_dim += 14                     # always jp + jv
        # obs_dim += 1                      # Fz
        # obs_dim += 6                      # Kp
        # if OBS_DAMPING: obs_dim += 6
        # self.obs_dim = obs_dim
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.seq_length * obs_dim,), dtype=np.float32
        # )

        self.state_space = self.observation_space

        self.state_space = self.observation_space

        self.action_space       = cfg.action_space
        self.observation_space  = cfg.observation_space
        self.obs_dim            = self.observation_space.shape[0] // self.seq_length
        # Rolling buffer for observations
        self.obs_buffer = torch.zeros(
            (self.num_envs, self.seq_length, self.obs_dim),
            device=self.device, dtype=torch.float32
        )


        ## 
        # FIXED MODE: Discrete waypoints for polishing
        # Waypoints: HOME → APPROACH → DESCENT → CONTACT (lateral) → RISE
        # Configure keypoint positions based on robot workspace
        # Use MANUAL_WP waypoints from original framework:
        # HOME:     [0.55, 0.00, 0.55]  - starting position
        # APPROACH: [0.55, 0.00, 0.10]  - above table
        # CONTACT:  [0.55, 0.00, 0.00]  - on surface (Z=0)
        # LATERAL:  [0.55, ±0.20, 0.00] - polishing stroke
        # RISE:     [0.55, 0.00, 0.15]  - lift up
        keypoint_config = {
            "home_position": [0.55, 0.0, 0.55],       # Original: same as MANUAL_WP
            "approach_height": 0.10,                   # 10cm above surface (Z=0.10)
            "contact_surface_z": 0.00,                 # Surface at Z=0 (ground plane)
            "lateral_distance": 0.40,                  # ±0.20m = 0.40m total stroke
            "rise_height": 0.15,                       # Rise to Z=0.15
            "orientation": [0.0, 1.0, 0.0, 0.0],      # EE pointing down (qw,qx,qy,qz)
        }
        
        # Initialize Fixed Trajectory Manager with discrete waypoints
        self.traj_mgr = TrajectoryManager(
            num_envs=self.num_envs,
            device=self.device,
            keypoint_config=keypoint_config
        )
        self.traj_mgr.sample(torch.arange(self.num_envs, device=self.device))
        
        # Initialize Phase Controller for force feedback during contact
        self.phase_ctrl = FixedPhaseController(
            num_envs=self.num_envs,
            device=self.device,
            F_TOUCH=-1.0,       # Force threshold to detect contact [N]
            F_LOST=0.5,         # Force threshold to detect lost contact [N]
            DESCENT_VEL=0.02,   # Descent velocity [m/s]
        )
        
        # Print waypoint info
        traj_info = self.traj_mgr.get_trajectory_info()
        print(f"[INFO] Fixed Mode: {traj_info['num_waypoints']} waypoints")
        print(f"[INFO] Spatial bounds: {traj_info['spatial_bounds']}")

        # Initialize reward function based on config
        self.reward_type = cfg.reward_type
        if self.reward_type not in REWARD_FUNCS:
            print(f"[WARNING] Unknown reward_type '{self.reward_type}', using 'kpz'")
            self.reward_type = 'kpz'
        self._reward_fn = REWARD_FUNCS[self.reward_type]
        print(f"[INFO] Reward function: {self.reward_type}")
        # print(f"[DEBUG] EE body index: {self.ee_body_idx}")
        # print(f"[DEBUG] EE jacobian index: {self.ee_jac_idx}")
        # print(f"[DEBUG] EE body name: {self.robot.body_names[self.ee_body_idx]}")
        # print(f"[DEBUG] Jacobian name: {self.robot.body_names[self.ee_jac_idx]}")
        # view = self.robot.root_physx_view
        # print("Link names:", view.shared_metatype.link_names)
        # print("DOF names: ", view.shared_metatype.dof_names)
        self.tau2 = torch.zeros((self.num_envs, self.action_dim),
                        dtype=torch.float32, device=self.device)
        
        self._prev_ori_err = torch.zeros(self.num_envs, device=self.device)

 
    def get_env_info(self):
        """Return environment info for RL-Games compatibility."""
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "agents": self.num_envs,
        }
    
    def set_train_info(self, env_frames: int, *args, **kwargs):
        """RL-Games callback for training info. No-op for this env."""
        pass
    
    def get_number_of_agents(self):
        """Return number of agents (environments)."""
        return self.num_envs
    
    def _setup_scene(self):
        # Call parent setup
        super()._setup_scene()

        # Spawn ground plane
        spawn_ground_plane("/World/ground", GroundPlaneCfg(), translation=(0, 0, 0))
        carb.log_warn("SETUP: Ground plane spawned")

        # Instantiate and store robot articulation
        self.robot = Articulation(self.cfg.robot)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        carb.log_warn("SETUP: Robot cloned into scene")

        # Add dome lighting
        dome_cfg = DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
        spawn_light("/World/Light", dome_cfg)
        carb.log_warn("SETUP: Dome light spawned")
        
        ###
        #  Build OSC via wrapper
        self.osc = OSCWrapper(
            num_envs=self.num_envs,
            device=self.device,
            cfg=self.cfg.ctrl
        )

        # ────────────────────────────────────────────────────────────
        #  MASK per il toggle Posizione ↔ Forza sull'asse Z
        #    • _mask_free  : full‑pose (XYZ+rot)  – nessuna forza
        #    • _mask_touch : pose su XY+rot, forza su Z
        # ────────────────────────────────────────────────────────────
        self._mask_free  = ([1,1,1, 1,1,1],   [0,0,0, 0,0,0])
        self._mask_touch = ([1,1,0, 1,1,1],   [0,0,1, 0,0,0])

        # Z che congeleremo al primo contatto
        self._z_hold = torch.zeros(self.num_envs, device=self.device)

        ###
        #  Visualization markers for goal and end-effector
        self.goal_marker = VisualizationMarkers(
            FRAME_MARKER_CFG.replace(prim_path="/World/Debug/GoalFrame")
        )
        self.ee_marker = VisualizationMarkers(
            FRAME_MARKER_CFG.replace(prim_path="/World/Debug/EEFrame")
        )




    def _pre_physics_step(self, actions):
        # # split incoming action
        # joint_cmd, kp_delta = actions[:, :7], actions[:, 7:]

        # # --------- rescale + apply kp updates ------------ 
        # scale = 0.05*(self.kp_hi - self.kp_lo)       # 5 % step each call
        # self.dynamic_kp += kp_delta * scale
        # self.dynamic_kp.clamp_(self.kp_lo, self.kp_hi)
        idx = 0
        if JOINT_TORQUE:
            self.joint_actions = actions[:, idx:idx+7]; idx += 7
        else:
            self.joint_actions = None

        # ---- Kp delta (solo assi liberi) ------------------------------------
        if KP_DELTA:
            kp_delta = actions[:, idx:idx+len(TRAIN_AXES)]; idx += len(TRAIN_AXES)
            for j, axis in enumerate(TRAIN_AXES):
                scale = KPZ_DELTA_SCALE * (self.kp_hi[axis] - self.kp_lo[axis])
                self.dynamic_kp[:, axis] += kp_delta[:, j] * scale
            self.dynamic_kp.clamp_(self.kp_lo, self.kp_hi)

        # ---- Damping ratio delta -------------------------------------------
        if DAMPING_DELTA:
            z_delta = actions[:, idx:idx+len(TRAIN_AXES)]; idx += len(TRAIN_AXES)
            for j, axis in enumerate(TRAIN_AXES):
                self.dynamic_zeta[:, axis] += z_delta[:, j] * ZETA_DELTA_SCALE
            z_lo, z_hi = self.cfg.ctrl.motion_damping_ratio_limits_task
            self.dynamic_zeta.clamp_(z_lo, z_hi)




        # print(f"[Frame {self.frame:5d}] dynamic_kp env0 = {self.dynamic_kp[0].cpu().numpy()}")
        # print(f"[Frame {self.frame:5d}] dynamic_zeta env0 = {self.dynamic_zeta[0].cpu().numpy()}")
        # print(f"[Frame {self.frame:5d}] τ_computed env0 = {self.tau2[0].cpu().numpy()}", flush=True)


        # --------- stash joint action for the old pipe ----
        # self.joint_actions = joint_cmd
        self.frame += 1


    def _apply_action(self):
        # ────────────────────────────────────────────────────────────────
        # 0) Environment origins
        # ────────────────────────────────────────────────────────────────
        env_origins = torch.as_tensor(self.scene.env_origins, device=self.device)  # [N,3]

        # ─────────────────────────────────────────────────────────────
        # 1)  Update contact sensor  ➜  f_ext, fz
        # ─────────────────────────────────────────────────────────────
        self.cube_sensor.update(self.physics_dt)
        f_ext = self.cube_sensor.data.net_forces_w[:, 0]          # (N, 3)   world‑frame

        fz_raw    = f_ext[:, 2]    
        self._fz_ema = (1.0 - self._ema_alpha) * self._fz_ema + \
               self._ema_alpha * fz_raw
        fz = self._fz_ema

        ee_pos  = self.robot.data.body_pos_w[:,  self.ee_body_idx]  # (N, 3)
        ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx]  # (N, 4)


        touch  = (self.phase == 0) & (fz < F_TOUCH)      # entrata in contatto
        lost   = (self.phase == 1) & (fz > F_LOST)       # perdita contatto
        rising = self.phase == 2

        if touch.any():                                   # ⇢ forza  ON
            self._z_hold[touch] = ee_pos[touch, 2]        # congela Z
            self.dynamic_kp[touch, 2]   = 0.2 * self.kp_hi[2]
            self.dynamic_zeta[touch, 2] = 1.5
            mc, fc = self._mask_touch
            self.osc.impl.cfg.motion_control_axes_task         = mc
            self.osc.impl.cfg.contact_wrench_control_axes_task = fc
            self.phase[touch] = 1

        if lost.any(): 
            self._int_F_err[lost] = 0.0                                   # ⇢ forza OFF
            mc, fc = self._mask_free
            self.osc.impl.cfg.motion_control_axes_task         = mc
            self.osc.impl.cfg.contact_wrench_control_axes_task = fc
            self.phase[lost] = 0




        # ─────────────────────────────────────────────────────────────
        # 2)  Get current EE pose and desired waypoint pose
        # ─────────────────────────────────────────────────────────────
        ee_pos  = self.robot.data.body_pos_w[:,  self.ee_body_idx]  # (N, 3)
        ee_quat = self.robot.data.body_quat_w[:, self.ee_body_idx]  # (N, 4)

        batch   = torch.arange(self.num_envs, device=self.device)
        p_des, q_des = self.traj_mgr.get_targets(batch, self.wpt_idx)  # (N, 3), (N, 4)
        # ---- override per gli env ancora in discesa (fase 0)
        descent = self.phase == 0
        # ---- PHASE 0 : scendi finché non senti F_TOUCH
        if descent.any():
            # XY = waypoint nominale (per evitare drift), Z scende di dz_step
            p_des[descent, 2] = ee_pos[descent, 2] - DESCENT_DZ
            # orientazione: tieni quella del path, non serve cambiarla
            q_des[descent]    = q_des[descent]

        follow = self.phase == 1
        if follow.any():
            # errore di forza (positivo se spingi meno del target)
            errF = (self.fz_target - fz[follow])

            # integrale (anti‑wind‑up: tienila entro ±50 N·s)
            self._int_F_err[follow] = torch.clamp(
                self._int_F_err[follow] + errF * self.physics_dt,
                min=-15.0, max=15.0
            )

            dz = ADM_KP * errF + ADM_KI * self._int_F_err[follow]
            dz = torch.clamp(dz, -ADM_DZ_CLAMP, ADM_DZ_CLAMP)

            # comando finale
            p_des[follow, 2] = self._z_hold[follow] + dz

        if touch.any():
            self._int_F_err[touch] = 0.0          # azzera integrale

        if rising.any():
            p_des[rising, 2] = torch.minimum(
                p_des[rising, 2] + DESCENT_DZ,      # usa lo stesso step ma in su
                self._rise_target_z[rising]
            )
            done_rise = torch.isclose(p_des[rising, 2], self._rise_target_z[rising])
            if done_rise.any():
                idx = rising.nonzero(as_tuple=True)[0][done_rise]
                self.phase[idx] = 0                 # ricomincia il ciclo


        # p_des[self.phase == 1, 2] = self._z_hold[self.phase == 1]
        # Position error in the XY‑plane (Z is force–controlled)
        delta_xy = (p_des + env_origins)[:, :2] - ee_pos[:, :2]     # (N, 2)
        pos_err  = torch.linalg.vector_norm(delta_xy, dim=1)        # (N,)

        # Orientation error (angle‑axis distance)
        dot      = torch.abs((q_des * ee_quat).sum(dim=1))
        ang_err  = 2 * torch.acos(torch.clamp(dot, -1.0, 1.0))      # (N,)

        # ─────────────────────────────────────────────────────────────
        # 3)  Way‑point advancement logic
        #     • Must satisfy XY tolerance  *and*  force tolerance
        # ─────────────────────────────────────────────────────────────

        reached_wp = (
            (pos_err < self.wp_eps) &
            (ang_err < self.wp_ori_eps) &
            (self.phase == 1)
        )

        reached_last = (self.wpt_idx == self.traj_mgr.T-1) & reached_wp
        if reached_last.any():
            self.phase[reached_last] = 2
            self._rise_target_z[reached_last] = ee_pos[reached_last, 2] + RISE_DZ




        sim_time    = self.frame * self.physics_dt
        need_change = (sim_time - self.last_target_change_time) >= 2.0
        mask        = reached_wp | need_change

        within_traj = self.wpt_idx < (self.traj_mgr.T - 1)
        self.wpt_idx[mask & within_traj] += 1
        self.last_target_change_time[mask] = sim_time
        self.wpt_idx.clamp_(0, self.traj_mgr.T - 1)

        # self._last_ori_err = ang_err  

        # pos_err = torch.linalg.vector_norm(p_des + env_origins - ee_pos, dim=1)           # [N]
        
        # within_traj = self.wpt_idx < (self.traj_mgr.T - 1)

        # self.wpt_idx[mask & within_traj] += 1
        # self.last_target_change_time[mask] = sim_time
        # ───── protezione: assicura che l'indice resti nel range [0, T-1] ─────
        # self.wpt_idx.clamp_(min=0, max=self.traj_mgr.T - 1)

        # idx     = self.wpt_idx                       # (N,) long
        # batch   = torch.arange(self.num_envs, device=self.device)

        # Sync phase_ctrl.phase with self.phase for reward functions
        # Mapping: env phase 0→DESCENT(2), 1→CONTACT(3), 2→RISE(4)
        phase_map = torch.tensor([FixedPhase.DESCENT, FixedPhase.CONTACT, FixedPhase.RISE], device=self.device)
        self.phase_ctrl.phase = phase_map[self.phase]

        # ─────────────────────────────────────────────────────────────
        # 4)  Build OSC command  (pose + wrench + impedance)
        # ─────────────────────────────────────────────────────────────
        p_des_offset = p_des + env_origins                      # [N,3]
        target_pose  = torch.cat((p_des_offset, q_des), dim=1)  # [N,7]

        kp    = self.dynamic_kp                                     # (N, 6)
        mode  = self.cfg.ctrl.impedance_mode

        target_wrench = torch.zeros(self.num_envs, 6, device=self.device)
        target_wrench[:, 2] = self.fz_target

        if mode == "variable":            # pose + wrench + kp + zeta  → 25
            cmd = torch.cat((target_pose, target_wrench, kp, self.dynamic_zeta), dim=1)

        elif mode == "variable_kp":       # pose + wrench + kp         → 19
            cmd = torch.cat((target_pose, target_wrench, kp), dim=1)

        else:                             # "fixed": pose + wrench     → 13
            cmd = torch.cat((target_pose, target_wrench), dim=1)

        current_pose = torch.cat((ee_pos, ee_quat), dim=1)  # [N,7]
        self.osc.set_command(cmd, current_ee_pose_b=current_pose)

        # delta_xy     = p_des_offset[:, :2] - ee_pos[:, :2]
        # pos_err      = torch.linalg.vector_norm(delta_xy, dim=1)

        # 3) stiffness per impedenza (variable_kp)
        # kp = self.dynamic_kp  # [N,6]

        # static damping from your CtrlCfg – zeta shape [N,6]
        # zeta = torch.tensor(self.cfg.ctrl.motion_damping_ratio_task,
        #                     dtype=torch.float32, device=self.device
        #            ).unsqueeze(0).repeat(self.num_envs, 1)

        # 2) target wrench (solo Fz = 2 N)
        # ee_quat      = self.robot.data.body_quat_w[:, self.ee_body_idx]            # [N,4]

        # concateniamo pose (7) + stiffness (6) → comando shape [N,13]
        # cmd = torch.cat((target_pose, target_wrench, kp, zeta), dim=1)
        # ---------------------------------------------------------------
        #  Costruisci il comando OSC in base alla modalità di impedenza
        # ---------------------------------------------------------------
        # mode = self.cfg.ctrl.impedance_mode

        # self.osc.set_command(cmd, current_ee_pose_b=current_pose)

        # ────────────────────────────────────────────────────────────────
        # 5) Dinamica
        # ────────────────────────────────────────────────────────────────
        jacobian = self.robot.root_physx_view.get_jacobians()[
                :, self.ee_jac_idx, :, self.joint_ids]                          # [N,6,7]
        mass_mat = self.robot.root_physx_view.get_generalized_mass_matrices()[:, :7, :7]
        ee_vel   = torch.cat((self.robot.data.body_lin_vel_w[:, self.ee_body_idx],
                            self.robot.data.body_ang_vel_w[:, self.ee_body_idx]), dim=1)

        gravity_vec = self.robot.root_physx_view.get_gravity_compensation_forces()[:, :7]
        tau = self.osc.compute(
            jacobian_b               = jacobian,
            current_ee_pose_b        = current_pose,
            current_ee_vel_b         = ee_vel,
            current_ee_force_b       = f_ext,        # ← qui
            mass_matrix              = mass_mat,
            gravity                  = gravity_vec,
            current_joint_pos        = self.robot.data.joint_pos[:, self.joint_ids],
            current_joint_vel        = self.robot.data.joint_vel[:, self.joint_ids],
            nullspace_joint_pos_target = torch.zeros(
                (self.num_envs, self.action_dim), device=self.device),
        )
        self.tau2 = tau.detach()

        # tau = torch.clamp(tau, -self.cfg.ctrl.torque_limit, self.cfg.ctrl.torque_limit)
        
        # prendo solo la coppia (calcolata vs applicata) del primo env e la salvo
        comp = tau[0].cpu().tolist()   # [τ1..τ7]
        grp_shoulder = self.robot.actuators["panda_shoulder"]
        grp_forearm  = self.robot.actuators["panda_forearm"]
        eff_sh = grp_shoulder.applied_effort
        eff_fr = grp_forearm.applied_effort
        applied = []
        # se non inizializzate, metto zero
        applied += (eff_sh[0].cpu().tolist() if eff_sh is not None else [0.0]*grp_shoulder.num_joints)
        applied += (eff_fr[0].cpu().tolist() if eff_fr is not None else [0.0]*grp_forearm.num_joints)
        # salvo solo 1 riga per frame, con env=0
        self.tau_history.append([self.frame, *comp, *applied])

        if JOINT_TORQUE and self.joint_actions is not None:
            tau += self.joint_actions * self.torque_limit * TORQUE_SCALE    



        self.robot.set_joint_effort_target(tau, joint_ids=self.joint_ids)
        self.robot.write_data_to_sim()
        self._last_pos_err = pos_err       # (N,) tensor
        self._last_ori_err = ang_err

        # ────────────────────────────────────────────────────────────────
        # 4) DEBUG (controlled by cfg.debug_interval, 0=disabled)
        # ────────────────────────────────────────────────────────────────
        if self.cfg.debug_interval > 0 and self.frame % self.cfg.debug_interval == 0:
            q = self.robot.data.joint_pos[0]                   # env 0, tensor (7,)
            q_lo = [-166.0, -101.0, -166.0, -176.0, -166.0,   -1.0, -166.0]
            q_hi = [ 166.0,  101.0,  166.0,   -4.0,  166.0,  215.0,  166.0]
            print_joint_debug(q, q_lo, q_hi,
                            names=["joint1","joint2","joint3",
                                    "joint4","joint5","joint6","joint7"])
            

            # impedance debug
            kp0   = self.dynamic_kp[0]        # (6,)
            zeta0 = self.dynamic_zeta[0]
            # Convert tensors to scalars for debug print
            kp_lo_scalar = self.kp_lo[0].item()
            kp_hi_scalar = self.kp_hi[0].item()
            print_impedance_debug(kp0, zeta0, kp_lo_scalar, kp_hi_scalar)

            # inside the debug section of _apply_action
            if hasattr(self, "_last_reward"):
                print(f"Reward env‑0: {self._last_reward[0]:+.4f}")

            print(f"[Frame {self.frame}] Phase env‑0: {self.phase[0].item()}")
            # dentro _apply_action(), subito prima del print_joint_debug()
            fz0 = f_ext[0, 2].item()               # env‑0, asse Z
            print(f"Contact Fz: {fz0:+6.2f} N")
            e = 0                                   
            print(f"[STEP {self.frame}] wp{e}={self.wpt_idx[e].item()} "
                f"pos_err{e}={pos_err[e].item():.4f}")
            print("  target p :", p_des[e].cpu().numpy())
            print("  actual p :", ee_pos[e].cpu().numpy())
            print("  target q :", q_des[e].cpu().numpy())
            print("  actual q :", ee_quat[e].cpu().numpy())
            detJJT = torch.det(jacobian[e] @ jacobian[e].T)
            print("  manip    :", torch.sqrt(detJJT).item())
            print("  tau      :", tau[e].cpu().numpy())
            f_env = self.cube_sensor.data.net_forces_w[e, 0].cpu().numpy()
            print("  contact F:", f_env)           # (Fx, Fy, Fz)

        # --------------- marker visualization (always active) ---------------
        e = 0  # always visualize env 0
        goal_p  = p_des_offset[e:e+1].cpu().numpy()   # shape (1,3)
        goal_q  = q_des[e:e+1].cpu().numpy()          # shape (1,4)
        ee_p    = ee_pos[e:e+1].cpu().numpy()         # shape (1,3)
        ee_q    = ee_quat[e:e+1].cpu().numpy()        # shape (1,4)
        self.goal_marker.visualize(goal_p, goal_q)
        self.ee_marker.visualize(ee_p, ee_q)

    def _get_observations(self):
        jp = self.robot.data.joint_pos[:, self.joint_ids]
        jv = self.robot.data.joint_vel[:, self.joint_ids]
        fz = self.cube_sensor.data.net_forces_w[:, 0, 2:3]
        kp = self.dynamic_kp
        pieces = [jp, jv, fz, kp]
        if OBS_DAMPING:
            pieces.append(self.dynamic_zeta)
        obs_t = torch.cat(pieces, dim=1)
        self.obs_buffer = torch.roll(self.obs_buffer, -1, dims=1)
        self.obs_buffer[:, -1, :] = obs_t
        flat = self.obs_buffer.reshape(self.num_envs, -1)
        return {"policy": flat, "critic": flat}



    def _get_rewards(self):
        """
        Compute rewards using phase-aware reward function from rewards module.
        The reward function is selected based on cfg.reward_type:
        - "kpz": Kz-only control (thesis Section 4.6)
        - "kz_dz": Kz + damping ratio control
        """
        # Call the configured reward function
        reward, reward_terms = self._reward_fn(self)
        
        # Store reward terms for logging (optional)
        self._last_reward_terms = reward_terms
        self._last_reward = reward.detach()
        
        return reward



    def _get_dones(self):
        """
        done    →  usato dal PPO (reset immediato)
        timeout →  usato da RL-Games per statistiche (non forza reset se False)
        """
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        reached_pos = (self._last_pos_err < self.wp_eps)
        reached_ori = (self._last_ori_err < self.wp_ori_eps)
        reached_last = (self.wpt_idx == self.traj_mgr.T - 1) & reached_pos & reached_ori

        done = timeout | reached_last
        return done, timeout


    def _reset_idx(self, env_ids=None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        self.phase[env_ids] = 0
        self._z_hold[env_ids]    = 0.0
        self._int_F_err[env_ids] = 0.0

        self.dynamic_kp[env_ids] = self.init_kp
        self.dynamic_zeta[env_ids] = self.init_zeta

        mc, fc = self._mask_free
        self.osc.impl.cfg.motion_control_axes_task         = mc
        self.osc.impl.cfg.contact_wrench_control_axes_task = fc

        self.wpt_idx[env_ids] = 0
        self.last_target_change_time[env_ids] = 0.0
        self.traj_mgr.sample(env_ids)       



        self.frame = 0




        # Base pose noise for reset
        # carb.log_warn_custom(f"RESET: {self.cfg.robot.init_state.joint_pos} envs reset")
        joint_order = [f"panda_joint{i}" for i in range(1, 8)]
        base_pos = torch.tensor(
            [self.cfg.robot.init_state.joint_pos[j] for j in joint_order],
            device=self.device
        )
        # base_pos = torch.tensor([0.3760, 0.2049, 0.1918, -1.9632, -0.04678, 2.16445, 1.37637], device=self.device)
        noise = (torch.rand((len(env_ids), 7), device=self.device) - 0.5) * 0.1  # ±5°
        rand_pos = base_pos.unsqueeze(0) + noise
        rand_vel = torch.zeros_like(rand_pos)
        self.robot.write_joint_state_to_sim(rand_pos, rand_vel, joint_ids=self.joint_ids, env_ids=env_ids)

        # # Randomize base translation within range
        # rand_x = torch.FloatTensor(len(env_ids)).uniform_(*self.base_pos_range['x']).to(self.device)
        # rand_y = torch.FloatTensor(len(env_ids)).uniform_(*self.base_pos_range['y']).to(self.device)
        # rand_z = torch.zeros(len(env_ids), device=self.device)
        # jitter = torch.stack((rand_x, rand_y, rand_z), dim=1)
        # env_origins = torch.tensor(self.scene.env_origins, device=self.device)[env_ids]
        # global_translations = env_origins + jitter
        # orientations = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        # root_pose = torch.cat((global_translations, orientations), dim=1)
        # self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)

        # 1) rigenera i key-points
        # self._sample_waypoints(env_ids)
        # self.traj_mgr.sample(env_ids)

        # 2) azzera il contatore waypoint e il timer per quegli env
        self.wpt_idx[env_ids]                = 0
        self._last_pos_err[env_ids] = torch.inf   # qualunque valore > wp_eps

        self.last_target_change_time[env_ids] = 0.0






        super()._reset_idx(env_ids)
        done = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        # Initialize observation buffer with current state
        jp = self.robot.data.joint_pos[env_ids][:, self.joint_ids]
        jv = self.robot.data.joint_vel[env_ids][:, self.joint_ids]
        f0 = torch.zeros((len(env_ids), 1), device=self.device)
        kp0 = self.dynamic_kp[env_ids]

        pieces = [jp, jv, f0, kp0]
        if OBS_DAMPING:                                                # <‑‑ NEW
            pieces.append(self.dynamic_zeta[env_ids])                  # <‑‑ NEW

        init = torch.cat(pieces, dim=1).unsqueeze(1)                   # (N,1,obs_dim)
        init = init.expand(-1, self.seq_length, -1)                    # (N,T,obs_dim)
        self.obs_buffer[env_ids].copy_(init)

        # Reset last EE height
        z = self.robot.data.body_pos_w[:, self.ee_body_idx][:, 2]
        self.last_ee_z = z.clone()
                # --- NEW: dump last episode’s torques to CSV ---
        # if self.tau_history:
        #     arr = np.array(self.tau_history, float)
        #     header = (
        #         "step,"
        #         + ",".join(f"tau{i+1}"     for i in range(self.action_dim)) + ","
        #         + ",".join(f"applied{i+1}" for i in range(self.action_dim))
        #     )
        #     np.savetxt(self.log_path, arr, delimiter=",", header=header, comments="")
        #     carb.log_warn(f"Saved torque history to {self.log_path}")

        self._prev_ori_err[env_ids] = 0.0  # or a more suitable reset value

        #     self.tau_history.clear()

        return done, timeout
