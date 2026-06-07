"""Particle (water) visualization for the 3 methods.

Approach (quasi-static):
  Place N particles inside the mug interior cylinder at a 'half-fill' water level
  (mug local frame). The particles are rigidly attached to the mug at start. As
  the trajectory plays out, particle world positions = T_world_mug @ p_local.
  Color = blue when the per-waypoint tilt is below θ_max (safe), red when above
  θ_max ('would spill'). We display N keyframes per method side-by-side, same
  as the arm-only render.

This gives an intuitive visual: when the mug tilts beyond the spill threshold,
the water particles look on the verge of leaving the cup.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sapien
import PIL.Image
from PIL import ImageDraw, ImageFont
import torch
from scipy.spatial.transform import Rotation as R

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info, look_at_quat
from mp.kinematics import URDFForwardKinematics
from mp.constants import MUG_UP_AXIS_LOCAL

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


def sample_water_particles(n: int, radius: float, y_low: float, y_high: float):
    """Sample n particles inside a cylindrical water region in mug local frame.
    Mug local +y is up. Cylinder axis = y. radius around (x=0, z=0)."""
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < n:
        x = rng.uniform(-radius, radius)
        z = rng.uniform(-radius, radius)
        if x * x + z * z <= radius ** 2:
            y = rng.uniform(y_low, y_high)
            pts.append([x, y, z])
    return np.asarray(pts)


def build_scene_with_particles(g, xhand_qpos, n_particles=25, mug_scale=0.12):
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

    # Particles: two pools (blue=safe, red=spill). At each frame we position
    # one set inside the mug; the other set we hide far below the ground.
    # Water region: half-fill of mug (mesh y in [-0.04, 0.03]), radius 0.045
    radius = 0.045
    pts_local = sample_water_particles(n_particles, radius=radius, y_low=-0.04, y_high=0.03)
    blue_mat = sapien.render.RenderMaterial(base_color=[0.15, 0.45, 0.95, 1.0],
                                             metallic=0.0, roughness=0.3)
    red_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.20, 0.15, 1.0],
                                            metallic=0.0, roughness=0.3)

    particles_blue = []
    particles_red = []
    for i in range(n_particles):
        pb = scene.create_actor_builder()
        pb.add_sphere_visual(radius=0.012, material=blue_mat)
        a = pb.build_kinematic(name=f"pb_{i}")
        particles_blue.append(a)
        pr = scene.create_actor_builder()
        pr.add_sphere_visual(radius=0.012, material=red_mat)
        a2 = pr.build_kinematic(name=f"pr_{i}")
        particles_red.append(a2)

    # Camera framed on the working volume — close enough to see particles, wide enough to see arm
    cam = scene.add_camera("cam", 720, 540, fovy=0.95, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[1.05, -0.75, 0.85],
                              q=look_at_quat([1.05, -0.75, 0.85], [0.5, 0.05, 0.30])))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, particles_blue, particles_red, pts_local, cam, sapien_order


def apply_qpos(robot, sapien_order, q_ur5e, xhand_qpos_arr):
    qpos_18 = np.concatenate([q_ur5e, xhand_qpos_arr])
    name_to_idx = {n: i for i, n in enumerate(JOINT_ORDER)}
    qpos_sapien = np.array([qpos_18[name_to_idx[n]] for n in sapien_order])
    robot.set_qpos(qpos_sapien)


def hide_pose():
    return sapien.Pose([0, 0, -5.0])


def render_frame(scene, cam):
    scene.update_render(); cam.take_picture()
    img = cam.get_picture("Color")[..., :3]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-keyframes", type=int, default=5)
    ap.add_argument("--n-particles", type=int, default=25)
    ap.add_argument("--out", default=str(ROOT / "figs/fig_particles.png"))
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d['theta_max_deg'])
    g_info = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand_qpos = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    (scene, robot, mug, particles_blue, particles_red, pts_local,
     cam, sapien_order) = build_scene_with_particles(g_info, xhand_qpos, n_particles=args.n_particles)

    grid_rows = []
    for method in METHODS:
        q_traj = d[f"{method}_q"]
        tilt = d[f"{method}_tilt"]
        N = q_traj.shape[0]
        idxs = np.linspace(0, N - 1, args.n_keyframes).astype(int)
        row_imgs = []
        for i in idxs:
            apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
            with torch.no_grad():
                T_ee = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
            T_mug = T_ee @ T_em
            qxyzw = R.from_matrix(T_mug[:3, :3]).as_quat()
            mug.set_pose(sapien.Pose(T_mug[:3, 3].tolist(),
                                     [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))

            # Place particles based on spill flag
            spill = tilt[i] > theta_max
            for pl, blue_a, red_a in zip(pts_local, particles_blue, particles_red):
                p_w = (T_mug[:3, :3] @ pl) + T_mug[:3, 3]
                if spill:
                    red_a.set_pose(sapien.Pose(p_w.tolist()))
                    blue_a.set_pose(hide_pose())
                else:
                    blue_a.set_pose(sapien.Pose(p_w.tolist()))
                    red_a.set_pose(hide_pose())
            img = render_frame(scene, cam)
            # Annotate tilt value in the corner
            im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
            except OSError:
                font = ImageFont.load_default()
            txt = f"tilt={tilt[i]:.1f}°"
            color = (180, 30, 30) if spill else (40, 80, 30)
            draw.rectangle([(im.width - 100, im.height - 26), (im.width, im.height)],
                            fill=(245, 245, 245))
            draw.text((im.width - 95, im.height - 22), txt, fill=color, font=font)
            row_imgs.append(np.array(im))

        H, W, _ = row_imgs[0].shape
        row = np.concatenate(row_imgs, axis=1)
        # method label on left
        label_w = 150
        label_panel = np.full((H, label_w, 3), 245, np.uint8)
        lp = PIL.Image.fromarray(label_panel); draw = ImageDraw.Draw(lp)
        try:
            font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font_lbl = ImageFont.load_default()
        label = METHOD_LABEL[method]
        words = label.split()
        line1 = words[0]
        line2 = " ".join(words[1:]) if len(words) > 1 else ""
        draw.text((10, H // 2 - 20), line1, fill=(20, 20, 20), font=font_lbl)
        draw.text((10, H // 2 + 4), line2, fill=(20, 20, 20), font=font_lbl)
        row = np.concatenate([np.array(lp), row], axis=1)
        grid_rows.append(row)

    # Header with timestamps
    H, W, _ = grid_rows[0].shape
    label_w = 150
    cell_w = (W - label_w) // args.n_keyframes
    header = np.full((34, W, 3), 230, np.uint8)
    hp = PIL.Image.fromarray(header); draw = ImageDraw.Draw(hp)
    try:
        fh = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
    except OSError:
        fh = ImageFont.load_default()
    dt = float(d['dt'])
    N0 = d['minjerk_tilt'].shape[0]
    for k, i in enumerate(np.linspace(0, N0 - 1, args.n_keyframes).astype(int)):
        x = label_w + k * cell_w + cell_w // 2 - 30
        draw.text((x, 8), f"t={i*dt:.2f}s", fill=(20, 20, 20), font=fh)
    header = np.array(hp)

    grid = np.concatenate([header] + grid_rows, axis=0)
    PIL.Image.fromarray(grid).save(args.out)
    print(f"saved {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
