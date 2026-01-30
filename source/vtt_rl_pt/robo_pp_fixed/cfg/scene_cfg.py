"""scene_cfg.py

Interactive scene configuration with a red carpet that acts as the polishing
surface. Contact forces are read directly from PhysX.
"""

# ---------- Imports ----------
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass


# ---------- Scene Configuration ----------

@configclass
class JustPushSceneCfg(InteractiveSceneCfg):
    """Simple scene with a fixed red carpet acting as polishing surface."""

    # Static polishing surface (the "red carpet")
    red_carpet: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RedCarpet",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.0, 0.01),          # 80×100 cm, 1 cm thick
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Bright red
                opacity=1.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,        # Fixed in world frame
            ),
            activate_contact_sensors=True,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0),  # Slightly in front of robot base
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Contact sensor attached to the red carpet (polishing surface)
    # This allows detecting contact forces when EE presses on surface
    cube_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/RedCarpet",
        update_period=0.0,  # Update each sim step
        history_length=10,
        debug_vis=False,
    )

# Public interface
__all__ = ["JustPushSceneCfg"]
