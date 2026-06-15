"""Render a gallery of trajectories for multiple (start, goal, grasp) scenarios.

Generates:
  - For each scenario: a per-method (or single-method) GIF with water particles.
  - One big composite GIF stitching N scenarios side-by-side (synchronized).

Scenarios designed to show variety:
  1. Long lateral transport (default whole_200)
  2. Big vertical lift (whole_200)
  3. Diagonal swing (whole_200)
  4. Different grasp on default path (3finger_200)
  5. Pouring (whole_200, tilted goal)
  6. Different grasp on default path (2finger_0)
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
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize
from mp.run_grasp_comparison import extract_ee_to_mug

JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def tilted_pose(p, tilt_deg=70.0, axis="x"):
    rot = R.from_euler(axis, tilt_deg, degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = rot @ R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


SCENARIOS = [
    dict(name="A_long_lateral",   label="A. Long lateral (70cm)",
         start_pose=upright_pose((0.5, -0.3, 0.25)),
         goal_pose=upright_pose((0.5, 0.4, 0.40)),
         grasp="mug_0_left_whole_200", T_budget=1.0, theta_max=18.0),
    dict(name="B_vertical_lift",  label="B. Vertical lift (30cm)",
         start_pose=upright_pose((0.5, 0.0, 0.18)),
         goal_pose=upright_pose((0.5, 0.0, 0.50)),
         grasp="mug_0_left_whole_200", T_budget=0.8, theta_max=18.0),
    dict(name="C_diagonal_swing", label="C. Diagonal swing (60cm 3D)",
         start_pose=upright_pose((0.45, -0.25, 0.25)),
         goal_pose=upright_pose((0.65, 0.30, 0.45)),
         grasp="mug_0_left_whole_200", T_budget=1.0, theta_max=18.0),
    dict(name="D_3finger_grasp",  label="D. Same path w/ 3finger_200",
         start_pose=upright_pose((0.5, -0.3, 0.25)),
         goal_pose=upright_pose((0.5, 0.4, 0.40)),
         grasp="mug_0_left_3finger_200", T_budget=1.0, theta_max=18.0),
    dict(name="E_pouring",        label="E. Pouring (tilted goal 70°)",
         start_pose=upright_pose((0.5, -0.3, 0.30)),
         goal_pose=tilted_pose((0.5, 0.3, 0.40), tilt_deg=70.0),
         grasp="mug_0_left_whole_200", T_budget=1.5, theta_max=70.0),
    dict(name="F_2finger_grasp",  label="F. Same path w/ 2finger_0",
         start_pose=upright_pose((0.5, -0.3, 0.25)),
         goal_pose=upright_pose((0.5, 0.4, 0.40)),
         grasp="mug_0_left_2finger_0", T_budget=1.0, theta_max=18.0),
]


def build_scene(g, n_particles=30):
    scene = sapien.Scene(); scene.set_timestep(1/240); scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
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
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < n_particles:
        x = rng.uniform(-0.045, 0.045); z = rng.uniform(-0.045, 0.045)
        if x * x + z * z <= 0.045 ** 2:
            pts.append([x, rng.uniform(-0.05, 0.03), z])
    pts = np.asarray(pts)
    mat = sapien.render.RenderMaterial(base_color=[0.15, 0.55, 0.95, 1.0], roughness=0.3)
    particles = []
    for i in range(n_particles):
        pb = scene.create_actor_builder(); pb.add_sphere_visual(radius=0.013, material=mat)
        particles.append(pb.build_kinematic(name=f"p_{i}"))
    cam = scene.add_camera("cam", 640, 480, fovy=0.85, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[1.0, -0.7, 0.75],
                              q=look_at_quat([1.0, -0.7, 0.75], [0.5, 0.05, 0.32])))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, particles, pts, cam, sapien_order


def apply_qpos(robot, sapien_order, q_ur5e, xhand_qpos):
    qpos_18 = np.concatenate([q_ur5e, xhand_qpos])
    n2i = {n: i for i, n in enumerate(JOINT_ORDER)}
    qpos_s = np.array([qpos_18[n2i[n]] for n in sapien_order])
    robot.set_qpos(qpos_s)


def render_frame(scene, cam):
    scene.update_render(); cam.take_picture()
    img = cam.get_picture("Color")[..., :3]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def simulate_scenario(scene, robot, mug, particles, pts_local, cam, sapien_order,
                       q_traj, xhand_qpos, T_em, fk, tilt_arr, theta_max, label, dt,
                       mug_up_local=np.array([0.0, 1.0, 0.0])):
    """Render N frames; release particles with free-fall when spill triggers."""
    G_W = np.array([0., 0., -9.81])
    N = q_traj.shape[0]; M = len(pts_local)
    released = np.zeros(M, dtype=bool)
    p_w_state = np.zeros((M, 3)); v_w_state = np.zeros((M, 3))

    mug_centers = np.zeros((N, 3)); mug_Rw = np.zeros((N, 3, 3))
    for i in range(N):
        with torch.no_grad():
            T_ee = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
        T_mug = T_ee @ T_em
        mug_centers[i] = T_mug[:3, 3]
        mug_Rw[i] = T_mug[:3, :3]

    frames = []
    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        R_mug = mug_Rw[i]; t_mug = mug_centers[i]
        qxyzw = R.from_matrix(R_mug).as_quat()
        mug.set_pose(sapien.Pose(t_mug.tolist(),
                                  [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
        v_mug = (mug_centers[i+1] - mug_centers[i-1]) / (2*dt) if 0 < i < N-1 else np.zeros(3)
        a_mug = (mug_centers[i+1] - 2*mug_centers[i] + mug_centers[i-1]) / (dt**2) if 0 < i < N-1 else np.zeros(3)
        g_eff_local = R_mug.T @ (G_W - a_mug)
        up_local_eff = -g_eff_local / (np.linalg.norm(g_eff_local) + 1e-9)
        spill_state = tilt_arr[i] > theta_max

        for k in range(M):
            if released[k]:
                p_w_state[k] = p_w_state[k] + v_w_state[k] * dt + 0.5 * G_W * dt**2
                v_w_state[k] = v_w_state[k] + G_W * dt
            else:
                p_w_state[k] = R_mug @ pts_local[k] + t_mug
                proj = np.dot(pts_local[k], up_local_eff)
                if spill_state and proj > 0.02:
                    released[k] = True
                    v_w_state[k] = v_mug + 0.3 * (R_mug @ mug_up_local)
            if p_w_state[k][2] < 0.012:
                p_w_state[k][2] = 0.012
                if released[k]:
                    v_w_state[k] = np.zeros(3)
            particles[k].set_pose(sapien.Pose(p_w_state[k].tolist()))

        img = render_frame(scene, cam)
        im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
        try:
            ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            fd = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except OSError:
            ft = ImageFont.load_default(); fd = ImageFont.load_default()
        draw.rectangle([(0, 0), (im.width, 28)], fill=(245, 245, 245))
        draw.text((8, 5), label, fill=(20, 20, 20), font=ft)
        bg = (255, 220, 220) if spill_state else (220, 240, 220)
        fg = (180, 30, 30) if spill_state else (40, 110, 40)
        draw.rectangle([(im.width-130, im.height-46), (im.width, im.height)], fill=bg)
        draw.text((im.width-125, im.height-42), f"tilt={tilt_arr[i]:.1f}°", fill=fg, font=fd)
        draw.text((im.width-125, im.height-22), f"spill {int(released.sum())}/{M}", fill=fg, font=fd)
        frames.append(np.array(im))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-particles", type=int, default=30)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--duration-ms", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "figs/anim_scenario_gallery.gif"))
    args = ap.parse_args()

    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    per_scenario = {}
    for s in SCENARIOS:
        print(f"\n=== {s['name']} : {s['label']}")
        T_em, xhand_qpos, ginfo = extract_ee_to_mug(s["grasp"])
        # IK
        def ik(T, q_seed):
            T_ee = T @ np.linalg.inv(T_em)
            q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                                torch.tensor(T_ee, dtype=torch.float64),
                                n_iters=2500, lr=0.03)
            return q.numpy(), info
        q_s, info_s = ik(s["start_pose"], ginfo['qpos'][:6])
        if info_s['pos_err'] > 0.02:
            print(f"  start IK fail: {info_s['pos_err']:.3f}"); continue
        q_g, info_g = ik(s["goal_pose"], q_s)
        if info_g['pos_err'] > 0.02:
            print(f"  goal  IK fail: {info_g['pos_err']:.3f}"); continue
        # CHOMP+spill
        N = args.N; T_budget = s["T_budget"]; dt = T_budget / (N - 1)
        sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=s["theta_max"], dt=dt), fk)
        cfg = CHOMPConfig(N=N, dt=dt, n_iters=1200, alpha_smooth=1.0,
                          gamma_spill=5.0, step_size=0.003)
        res = chomp_optimize(fk, q_s, q_g, cfg, spill=sp)
        d = res["spill_diag"]
        print(f"  CHOMP+spill: spill_ratio={d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")
        # Build a fresh scene per scenario (to support different xhand_qpos)
        scene, robot, mug, particles, pts_local, cam, sapien_order = build_scene(ginfo, n_particles=args.n_particles)
        frames = simulate_scenario(scene, robot, mug, particles, pts_local, cam,
                                    sapien_order, res["q_traj"], xhand_qpos, T_em, fk,
                                    d['tilt_deg'], s["theta_max"], s["label"], dt)
        # Save individual GIF
        outp = ROOT / f"figs/anim_scn_{s['name']}.gif"
        pil = [PIL.Image.fromarray(f) for f in frames]
        pil[0].save(outp, save_all=True, append_images=pil[1:],
                    duration=args.duration_ms, loop=0, optimize=False)
        print(f"  saved {outp}")
        per_scenario[s['name']] = frames

    # Composite — 3x2 grid synchronized (pad shorter sequences)
    keys = list(per_scenario.keys())
    if len(keys) >= 6:
        rows_def = [keys[:3], keys[3:6]]
    elif len(keys) >= 4:
        rows_def = [keys[:2], keys[2:4]]
    else:
        rows_def = [keys]
    N_max = max(len(per_scenario[k]) for k in keys)
    composite_frames = []
    H, W, _ = per_scenario[keys[0]][0].shape
    for i in range(N_max):
        row_imgs = []
        for row in rows_def:
            cells = []
            for k in row:
                seq = per_scenario[k]
                cells.append(seq[i] if i < len(seq) else seq[-1])
            row_imgs.append(np.concatenate(cells, axis=1))
        composite_frames.append(np.concatenate(row_imgs, axis=0))
    pil = [PIL.Image.fromarray(f) for f in composite_frames]
    pil[0].save(args.out, save_all=True, append_images=pil[1:],
                duration=args.duration_ms, loop=0, optimize=False)
    print(f"\nsaved composite {args.out}  ({len(pil)} frames)")


if __name__ == "__main__":
    main()
