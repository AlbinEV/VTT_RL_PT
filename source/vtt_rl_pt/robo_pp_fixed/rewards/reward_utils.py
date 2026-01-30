# vttRL/tasks/robo_pp/rewards/reward_utils.py
import torch

# Contact thresholds shared by env and rewards -----------------------------
F_TOUCH = -0.01      # [N]  contact established below this value
F_LOST  = -1e-6      # [N]  contact considered lost above this value
# --------------------------------------------------------------------------

def quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Batched quaternion -> rotation-matrix conversion.
    q: (*,4) with scalar part first.
    """
    # normalise once to avoid NaNs
    q = torch.nn.functional.normalize(q, dim=-1)
    qw, qx, qy, qz = q.unbind(-1)
    R = torch.stack(
        (
            1 - 2*(qy*qy + qz*qz),
            2*(qx*qy - qw*qz),
            2*(qx*qz + qw*qy),

            2*(qx*qy + qw*qz),
            1 - 2*(qx*qx + qz*qz),
            2*(qy*qz - qw*qx),

            2*(qx*qz - qw*qy),
            2*(qy*qz + qw*qx),
            1 - 2*(qx*qx + qy*qy),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))
    return R
