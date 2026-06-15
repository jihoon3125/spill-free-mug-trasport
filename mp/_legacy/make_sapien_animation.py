"""Render each method's trajectory in sapien at a few keyframes, stitch into
a single 3×N grid image (3 methods × N keyframes).

Output:
  figs/fig_3way_frames.png      ← grid image for slides
  figs/anim_<method>.gif        ← (optional, off by default) gif per method
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sapien
import PIL.Image

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info, look_at_quat
from mp.kinematics import URDFForwardKinematics
from scipy.spatial.transform import Rotation as R

METHODS = ["minjerk", "chomp", "chomp_spill"]
METHOD_LABEL = {"minjerk": "Min-jerk (naive)",
                "chomp": "CHOMP (smooth)",
                "chomp_spill": "CHOMP + spill (ours)"}
JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]


def build_sapien_scene(g, xhand_qpos):
    scene = sapien.Scene(); scene.set_timestep(1/240); scene.add_ground(0)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0, 0.5, -1], [0.9, 0.9, 0.9])

    loader = scene.create_urdf_loader(); loader.fix_root_link = True
    robot = loader.load(str(URDF))
    robot.set_root_pose(sapien.Pose([0, 0, 0]))

    b = scene.create_actor_builder()
    b.add_visual_from_file(str(ROOT / "data/mug_mesh.obj"), scale=[g['scale']] * 3)
    mug = b.build_kinematic(name="mug")

    tb = scene.create_actor_builder()
    tb.add_box_visual(half_size=[0.5, 0.5, 0.005],
                      material=sapien.render.RenderMaterial(base_color=[0.7, 0.6, 0.5, 1]))
    table = tb.build_kinematic(name="table"); table.set_pose(sapien.Pose([0.5, 0.05, 0.0]))

    cam = scene.add_camera("cam", 640, 480, fovy=1.0, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[1.1, -0.9, 0.85],
                              q=look_at_quat([1.1, -0.9, 0.85], [0.5, 0.05, 0.30])))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, cam, sapien_order


def apply_qpos(robot, sapien_order, q_ur5e, xhand_qpos_arr):
    """Set 18-DoF qpos: first 6 = UR5e, last 12 = frozen xhand grasp config."""
    qpos_18 = np.concatenate([q_ur5e, xhand_qpos_arr])
    name_to_idx = {n: i for i, n in enumerate(JOINT_ORDER)}
    qpos_sapien = np.array([qpos_18[name_to_idx[n]] for n in sapien_order])
    robot.set_qpos(qpos_sapien)


def render_frame(scene, cam):
    scene.update_render(); cam.take_picture()
    img = cam.get_picture("Color")[..., :3]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def annotate(img, text, color=(20, 20, 20)):
    from PIL import ImageDraw, ImageFont
    im = PIL.Image.fromarray(img.copy())
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (im.width, 30)], fill=(245, 245, 245))
    draw.text((8, 4), text, fill=color, font=font)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-keyframes", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "figs/fig_3way_frames.png"))
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand_qpos = np.load(ROOT / "data/xhand_qpos.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")

    scene, robot, mug, cam, sapien_order = build_sapien_scene(g, xhand_qpos)

    import torch
    grid_rows = []
    for method in METHODS:
        q_traj = d[f"{method}_q"]                       # (N, 6)
        N = q_traj.shape[0]
        idxs = np.linspace(0, N - 1, args.n_keyframes).astype(int)
        row_imgs = []
        for k, i in enumerate(idxs):
            apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
            # Compute mug pose via FK then T_em
            with torch.no_grad():
                T = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
            T_mug_world = T @ T_em
            qxyzw = R.from_matrix(T_mug_world[:3, :3]).as_quat()
            mug.set_pose(sapien.Pose(T_mug_world[:3, 3].tolist(),
                                     [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
            img = render_frame(scene, cam)
            row_imgs.append(img)

        # Stitch row with label on left
        H, W, _ = row_imgs[0].shape
        row = np.concatenate(row_imgs, axis=1)
        # Add method label on the left side
        label_w = 130
        label_panel = np.full((H, label_w, 3), 245, np.uint8)
        from PIL import ImageDraw, ImageFont
        lp = PIL.Image.fromarray(label_panel); draw = ImageDraw.Draw(lp)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        # word wrap label
        label = METHOD_LABEL[method]
        words = label.split()
        line1 = words[0]
        line2 = " ".join(words[1:]) if len(words) > 1 else ""
        draw.text((10, H // 2 - 24), line1, fill=(20, 20, 20), font=font)
        draw.text((10, H // 2 - 4), line2, fill=(20, 20, 20), font=font)
        row = np.concatenate([np.array(lp), row], axis=1)
        grid_rows.append(row)

    # Add a top header row with timestamps
    H, W, _ = grid_rows[0].shape
    label_w = 130
    cell_w = (W - label_w) // args.n_keyframes
    header = np.full((30, W, 3), 230, np.uint8)
    hp = PIL.Image.fromarray(header); draw = ImageDraw.Draw(hp)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    dt = float(d['dt'])
    N0 = d['minjerk_tilt'].shape[0]
    T_total = (N0 - 1) * dt
    for k, i in enumerate(np.linspace(0, N0 - 1, args.n_keyframes).astype(int)):
        t = i * dt
        x = label_w + k * cell_w + cell_w // 2 - 30
        draw.text((x, 6), f"t={t:.2f}s", fill=(20, 20, 20), font=font)
    header = np.array(hp)

    grid = np.concatenate([header] + grid_rows, axis=0)
    PIL.Image.fromarray(grid).save(args.out)
    print(f"saved {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
