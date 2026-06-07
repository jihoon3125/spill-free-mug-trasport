"""Sanity check: PyTorch URDF FK output vs sapien's link.pose at the same qpos.

Both should produce identical T_base_link → base (=xhand root) for any q[:6].
We use grasp_info qpos as the test config.
"""
from pathlib import Path
import numpy as np
import torch
import sapien
from scipy.spatial.transform import Rotation as R

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import build_scene, URDF
from mp.kinematics import URDFForwardKinematics


def main():
    scene, robot, mug, g = build_scene(gui=False)
    ee_link = next(l for l in robot.get_links() if l.name == "base")
    sapien_p = np.array(ee_link.pose.p)
    sapien_q = np.array(ee_link.pose.q)   # wxyz
    sapien_R = R.from_quat([sapien_q[1], sapien_q[2], sapien_q[3], sapien_q[0]]).as_matrix()

    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    print(f"[fk] {fk.n_active} active joints in chain")

    # Use only UR5e q (first 6) — xhand joints aren't in this chain
    # The chain base_link → base goes through wrist_3 → arm_hand_joint → base (fixed)
    # so only 6 active joints (UR5e revolutes).
    q = torch.tensor(g['qpos'][:fk.n_active], dtype=torch.float64)
    T = fk(q).detach().numpy()
    my_p = T[:3, 3]
    my_R = T[:3, :3]

    print(f"sapien EE pos  = {sapien_p}")
    print(f"my FK  EE pos  = {my_p}")
    print(f"pos diff norm  = {np.linalg.norm(sapien_p - my_p):.6e}")
    rot_diff = (my_R @ sapien_R.T - np.eye(3))
    print(f"rot diff |·|_F = {np.linalg.norm(rot_diff):.6e}")
    print(f"my  R[:3,2] (EE +z in world) = {my_R[:3,2]}")
    print(f"sap R[:3,2] (EE +z in world) = {sapien_R[:3,2]}")


if __name__ == "__main__":
    main()
