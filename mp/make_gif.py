"""Render the full trajectory of each method in sapien (with particles inside
the mug), then stitch 3 methods side-by-side into one animated GIF.

Output:
  figs/anim_3way.gif       — 3-method side-by-side GIF
  figs/anim_<method>.gif   — per-method GIF
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
from mp.make_particle_demo import (METHODS, METHOD_LABEL, JOINT_ORDER,
                                      sample_water_particles, apply_qpos, hide_pose,
                                      render_frame)


def build_scene_with_particles(g, n_particles=25):
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

    pts_local = sample_water_particles(n_particles, radius=0.045, y_low=-0.04, y_high=0.03)
    blue_mat = sapien.render.RenderMaterial(base_color=[0.15, 0.45, 0.95, 1.0],
                                             metallic=0.0, roughness=0.3)
    red_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.20, 0.15, 1.0],
                                            metallic=0.0, roughness=0.3)

    particles_blue, particles_red = [], []
    for i in range(n_particles):
        pb = scene.create_actor_builder()
        pb.add_sphere_visual(radius=0.012, material=blue_mat)
        particles_blue.append(pb.build_kinematic(name=f"pb_{i}"))
        pr = scene.create_actor_builder()
        pr.add_sphere_visual(radius=0.012, material=red_mat)
        particles_red.append(pr.build_kinematic(name=f"pr_{i}"))

    cam = scene.add_camera("cam", 640, 480, fovy=0.95, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[1.05, -0.75, 0.85],
                              q=look_at_quat([1.05, -0.75, 0.85], [0.5, 0.05, 0.30])))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, particles_blue, particles_red, pts_local, cam, sapien_order


def render_trajectory(scene, robot, mug, particles_blue, particles_red, pts_local,
                       cam, sapien_order, q_traj, xhand_qpos, T_em, fk, tilt_arr,
                       theta_max, method_label):
    frames = []
    N = q_traj.shape[0]
    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        with torch.no_grad():
            T_ee = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
        T_mug = T_ee @ T_em
        qxyzw = R.from_matrix(T_mug[:3, :3]).as_quat()
        mug.set_pose(sapien.Pose(T_mug[:3, 3].tolist(),
                                  [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
        spill = tilt_arr[i] > theta_max
        for pl, blue_a, red_a in zip(pts_local, particles_blue, particles_red):
            p_w = (T_mug[:3, :3] @ pl) + T_mug[:3, 3]
            if spill:
                red_a.set_pose(sapien.Pose(p_w.tolist()))
                blue_a.set_pose(hide_pose())
            else:
                blue_a.set_pose(sapien.Pose(p_w.tolist()))
                red_a.set_pose(hide_pose())
        img = render_frame(scene, cam)
        # Annotate method label (top) + tilt (bottom)
        im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_tilt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font_title = ImageFont.load_default()
            font_tilt = ImageFont.load_default()
        draw.rectangle([(0, 0), (im.width, 30)], fill=(245, 245, 245))
        draw.text((8, 5), method_label, fill=(20, 20, 20), font=font_title)
        bg_color = (255, 220, 220) if spill else (220, 240, 220)
        txt_color = (180, 30, 30) if spill else (40, 110, 40)
        draw.rectangle([(im.width - 130, im.height - 30), (im.width, im.height)],
                        fill=bg_color)
        draw.text((im.width - 125, im.height - 26),
                   f"tilt={tilt_arr[i]:.1f}°", fill=txt_color, font=font_tilt)
        frames.append(np.array(im))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=25, help="GIF playback fps (real-time = 50 for our 1s traj)")
    ap.add_argument("--out", default=str(ROOT / "figs/anim_3way.gif"))
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d['theta_max_deg'])
    g_info = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand_qpos = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    (scene, robot, mug, particles_blue, particles_red, pts_local,
     cam, sapien_order) = build_scene_with_particles(g_info)

    per_method_frames: dict[str, list[np.ndarray]] = {}
    for method in METHODS:
        print(f"[render] {method}")
        q_traj = d[f"{method}_q"]
        tilt = d[f"{method}_tilt"]
        frames = render_trajectory(scene, robot, mug, particles_blue, particles_red,
                                    pts_local, cam, sapien_order, q_traj, xhand_qpos,
                                    T_em, fk, tilt, theta_max, METHOD_LABEL[method])
        per_method_frames[method] = frames

    # Save per-method GIFs
    duration_ms = int(1000 / args.fps)
    for method, frames in per_method_frames.items():
        out_p = ROOT / f"figs/anim_{method}.gif"
        pil_frames = [PIL.Image.fromarray(f) for f in frames]
        pil_frames[0].save(out_p, save_all=True, append_images=pil_frames[1:],
                            duration=duration_ms, loop=0, optimize=False)
        print(f"  saved {out_p}  ({len(frames)} frames @ {args.fps} fps)")

    # Side-by-side 3-method (skip STOMP to keep 3 columns)
    cols = ["minjerk", "chomp", "chomp_spill"]
    combined = []
    N = min(len(per_method_frames[c]) for c in cols)
    for i in range(N):
        row = np.concatenate([per_method_frames[c][i] for c in cols], axis=1)
        combined.append(row)
    pil_combined = [PIL.Image.fromarray(f) for f in combined]
    pil_combined[0].save(args.out, save_all=True, append_images=pil_combined[1:],
                          duration=duration_ms, loop=0, optimize=False)
    print(f"saved {args.out}  ({N} frames, 3-col)")


if __name__ == "__main__":
    main()
