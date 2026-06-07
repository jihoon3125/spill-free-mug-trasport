"""Given a desired mug pose in world (e.g., upright on a table), solve IK to
find UR5e qpos[:6] such that the held mug ends up at that pose.

Workflow:
  T_world_ee_target = T_world_mug_target @ inv(T_ee_to_mug)
  q = IK(T_world_ee_target, q_init = grasp_info qpos[:6])

Render side-by-side: original (grasp pose as-generated) vs upright (transport ready).
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
import sapien

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import build_scene, URDF, parse_grasp_info, render_headless
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT


def upright_mug_target_pose(p_world=(0.5, 0.0, 0.2)) -> np.ndarray:
    """Return 4x4 T such that the mug stands upright (local +y aligned with world +z),
    translated to p_world."""
    T = np.eye(4)
    T[:3, :3] = R_UPRIGHT
    T[:3, 3] = np.asarray(p_world, dtype=float)
    return T


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_ee_to_mug = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    T_mug_target = upright_mug_target_pose(p_world=(0.5, 0.0, 0.25))
    T_ee_target = T_mug_target @ np.linalg.inv(T_ee_to_mug)

    q_init = torch.tensor(g['qpos'][:fk.n_active], dtype=torch.float64)
    q_opt, info = ik_solve(fk, q_init, torch.tensor(T_ee_target, dtype=torch.float64),
                           n_iters=1500, lr=0.03, verbose=True)
    print(f"[ik] iters={info['iters']}  pos_err={info['pos_err']:.5f}  rot_err={info['rot_err']:.5f}")
    print(f"[ik] q_opt = {q_opt.numpy()}")

    # Save target qpos for downstream use
    qpos_transport = g['qpos'].copy()
    qpos_transport[:fk.n_active] = q_opt.numpy()
    np.save(ROOT / "data/qpos_transport_start.npy", qpos_transport)
    print(f"[save] qpos_transport_start.npy")

    # Render: rebuild scene with new qpos
    scene, robot, mug, _ = build_scene(gui=False)
    sapien_order = [j.name for j in robot.get_active_joints()]
    JOINT_ORDER = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
        "left_hand_mid_joint1", "left_hand_mid_joint2",
        "left_hand_ring_joint1", "left_hand_ring_joint2",
        "left_hand_pinky_joint1", "left_hand_pinky_joint2",
    ]
    name_to_idx = {n: i for i, n in enumerate(JOINT_ORDER)}
    qpos_sapien = np.array([qpos_transport[name_to_idx[n]] for n in sapien_order])
    robot.set_qpos(qpos_sapien)
    # Move mug to the new target pose
    from scipy.spatial.transform import Rotation as R
    Rq_wxyz = R.from_matrix(T_mug_target[:3, :3]).as_quat()
    Rq_wxyz = [Rq_wxyz[3], Rq_wxyz[0], Rq_wxyz[1], Rq_wxyz[2]]
    mug.set_pose(sapien.Pose(T_mug_target[:3, 3].tolist(), Rq_wxyz))

    render_headless(scene, ROOT / "figs/scene_transport_start.png")


if __name__ == "__main__":
    main()
