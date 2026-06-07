"""One-shot: load grasp_info → sapien scene → compute T_ee_to_mug → save.

T_ee_to_mug is the rigid relative transform from the hand's `base` link to
the mug. Constant for a given grasp. Used by motion planning to attach the
mug to the EE in planning (and to derive target EE poses from desired mug
poses).

Output:
  data/ee_to_mug.npy   shape (4,4) homogeneous SE(3)
"""
from pathlib import Path
import numpy as np
import sapien
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.scene import build_scene, ROOT


def pose_to_mat(pose: sapien.Pose) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
    M[:3, 3] = pose.p
    return M


def main():
    scene, robot, mug, g = build_scene(gui=False)

    ee_link = next(l for l in robot.get_links() if l.name == "base")
    T_base_to_ee = pose_to_mat(ee_link.pose)
    T_base_to_mug = pose_to_mat(mug.pose)
    T_ee_to_mug = np.linalg.inv(T_base_to_ee) @ T_base_to_mug

    # Sanity: mug z-axis in EE frame
    mug_z_in_ee = T_ee_to_mug[:3, 2]
    print(f"[ee→mug] translation = {T_ee_to_mug[:3,3]}")
    print(f"[ee→mug] mug z-axis (in EE frame) = {mug_z_in_ee}")
    print(f"[ee→mug] mug z-axis (in world @ this qpos) = {T_base_to_mug[:3,2]}")

    out = ROOT / "data/ee_to_mug.npy"
    np.save(out, T_ee_to_mug)
    print(f"[save] {out}")

    # Also save the xhand frozen config (qpos[6:] in grasp order)
    np.save(ROOT / "data/xhand_qpos.npy", g['qpos'][6:])
    print(f"[save] xhand_qpos.npy  shape={g['qpos'][6:].shape}")


if __name__ == "__main__":
    main()
