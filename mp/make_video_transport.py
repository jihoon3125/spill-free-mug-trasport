"""ONE clean transport video: mug carried along the trajectory + water particles
that spill (or not). CORRECT convention (uses _video_utils helpers).

Default: side-by-side naive (min-jerk) vs ours (CHOMP+spill) so "spills or not"
is directly visible. Use --method to render a single method.

Trajectories come from results/ablation.npz (re-planned with the corrected
mug-up convention). No render-time mesh flip needed.

Usage:
  python mp/make_video_transport.py                 # naive | ours side-by-side
  python mp/make_video_transport.py --method minjerk
Output: figs/video_transport.gif  (or figs/video_transport_<method>.gif)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as R

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp._video_utils import (MUG_UP_LOCAL, sample_water_particles,
                             compute_spill_state, render_camera, take_rgb,
                             annotate_frame, save_gif)

JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]
METHOD_LABEL = {"minjerk": "Naive (min-jerk)", "chomp": "CHOMP (smooth)",
                "chomp_spill": "Ours (CHOMP + spill)"}
G_WORLD = np.array([0.0, 0.0, -9.81])
N_PARTICLES = 60
P_RADIUS = 0.008
COL_RADIUS = 0.025      # water column radius (inside the cup)
FILL_Y_LOW = -0.050     # water surface near the brim (toward opening / -y)
FILL_Y_HIGH = 0.050     # water bottom (toward base / +y)
WATER_CENTER = np.array([0.0, 0.0, 0.0])     # local center of the water body
RIM_PROJ = 0.052        # rim height (proj along MUG_UP_LOCAL); surface just below
OVERFLOW_RATE = 2       # max particles spilling per frame (continuous stream)


def align_rot(a, b):
    """Rotation matrix R with R @ a = b (a, b unit vectors)."""
    a = a / (np.linalg.norm(a) + 1e-12); b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def build_scene(g):
    scene = sapien.Scene(); scene.set_timestep(1 / 240); scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.9, 0.9, 0.9])
    loader = scene.create_urdf_loader(); loader.fix_root_link = True
    robot = loader.load(str(URDF)); robot.set_root_pose(sapien.Pose([0, 0, 0]))
    b = scene.create_actor_builder()
    b.add_visual_from_file(str(ROOT / "data/mug_mesh.obj"), scale=[g['scale']] * 3)
    mug = b.build_kinematic(name="mug")
    tb = scene.create_actor_builder()
    tb.add_box_visual(half_size=[0.5, 0.6, 0.005],
                      material=sapien.render.RenderMaterial(base_color=[0.7, 0.6, 0.5, 1]))
    table = tb.build_kinematic(name="table"); table.set_pose(sapien.Pose([0.5, 0.05, 0.0]))
    pmat = sapien.render.RenderMaterial(base_color=[0.15, 0.55, 0.95, 1.0],
                                        metallic=0.0, roughness=0.3)
    particles = []
    for i in range(N_PARTICLES):
        pb = scene.create_actor_builder()
        pb.add_sphere_visual(radius=P_RADIUS, material=pmat)
        particles.append(pb.build_kinematic(name=f"p_{i}"))
    cam = render_camera(scene, w=640, h=560, fovy=0.95,
                        eye=(1.15, -0.95, 0.80), target=(0.5, 0.05, 0.30))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, particles, cam, sapien_order


def project_pts(centers, cam):
    """Project world mug-center positions to image pixels (OpenCV K @ E)."""
    K = np.asarray(cam.get_intrinsic_matrix()); E = np.asarray(cam.get_extrinsic_matrix())
    px = []
    for p in centers:
        pc = E @ np.array([p[0], p[1], p[2], 1.0]); uv = K @ pc[:3]
        px.append((uv[0] / uv[2], uv[1] / uv[2]))
    return px


def overlay_path(frame, px, i):
    import PIL.Image
    from PIL import ImageDraw
    im = PIL.Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.line([tuple(p) for p in px], fill=(200, 200, 200), width=1)          # full path (faint)
    if i > 0:
        d.line([tuple(p) for p in px[:i + 1]], fill=(255, 215, 0), width=3)  # traveled
    s, gpt, c = px[0], px[-1], px[i]
    d.ellipse([s[0] - 6, s[1] - 6, s[0] + 6, s[1] + 6], outline=(30, 160, 30), width=3)
    d.rectangle([gpt[0] - 6, gpt[1] - 6, gpt[0] + 6, gpt[1] + 6], outline=(200, 30, 30), width=3)
    d.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], fill=(0, 200, 255))
    return np.array(im)


def apply_qpos(robot, sapien_order, q_ur5e, xhand_qpos):
    qpos_18 = np.concatenate([q_ur5e, xhand_qpos])
    n2i = {n: i for i, n in enumerate(JOINT_ORDER)}
    robot.set_qpos(np.array([qpos_18[n2i[n]] for n in sapien_order]))


def simulate(scene, robot, mug, particles, cam, sapien_order,
             q_traj, xhand_qpos, T_em, fk, tilt_arr, theta_max, label, dt):
    N = q_traj.shape[0]; M = N_PARTICLES
    # water body filling the cup up to near the rim (local frame)
    pts0 = sample_water_particles(M, radius=COL_RADIUS, y_low=FILL_Y_LOW, y_high=FILL_Y_HIGH)
    up0 = MUG_UP_LOCAL / np.linalg.norm(MUG_UP_LOCAL)   # cup "up" (opening dir) in local
    released = np.zeros(M, dtype=bool)
    p_w = np.zeros((M, 3)); v_w = np.zeros((M, 3))
    centers = np.zeros((N, 3)); Rs = np.zeros((N, 3, 3))
    for i in range(N):
        with torch.no_grad():
            T = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy() @ T_em
        centers[i] = T[:3, 3]; Rs[i] = T[:3, :3]
    px = project_pts(centers, cam)        # mug-center pixels for path overlay

    frames = []
    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        T_mug = np.eye(4); T_mug[:3, :3] = Rs[i]; T_mug[:3, 3] = centers[i]
        qxyzw = R.from_matrix(Rs[i]).as_quat()
        mug.set_pose(sapien.Pose(centers[i].tolist(),
                                 [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
        if 0 < i < N - 1:
            v_mug = (centers[i + 1] - centers[i - 1]) / (2 * dt)
            a_mug = (centers[i + 1] - 2 * centers[i] + centers[i - 1]) / dt ** 2
        else:
            v_mug = np.zeros(3); a_mug = np.zeros(3)
        spill_state, tilt_deg, _, _ = compute_spill_state(T_mug, a_mug, theta_max)
        # water surface settles perpendicular to effective gravity → slosh.
        g_eff_w = G_WORLD - a_mug
        g_eff_local = Rs[i].T @ g_eff_w
        surf_up_local = -g_eff_local / (np.linalg.norm(g_eff_local) + 1e-9)
        R_slosh = align_rot(up0, surf_up_local)        # tilt water body toward effective gravity
        pour = g_eff_w.copy(); pour[2] = 0.0           # horizontal pour direction
        pn = np.linalg.norm(pour); pour = pour / pn if pn > 1e-6 else np.zeros(3)

        # 1) update positions; track how far each attached particle reaches toward rim
        proj_up = np.full(M, -1e9)
        for k in range(M):
            if released[k]:
                p_w[k] = p_w[k] + v_w[k] * dt + 0.5 * G_WORLD * dt ** 2
                v_w[k] = v_w[k] + G_WORLD * dt
            else:
                p_tilt = R_slosh @ (pts0[k] - WATER_CENTER) + WATER_CENTER
                p_w[k] = Rs[i] @ p_tilt + centers[i]
                proj_up[k] = float(np.dot(p_tilt, up0))
        # 2) overflow: when the spill metric is on, the highest (surface) particles
        #    pour OVER THE RIM (not through the wall): relocate to the lip edge on the
        #    pour side, then flow outward + down.
        if spill_state:
            mug_up_w = Rs[i] @ up0; mug_up_w = mug_up_w / (np.linalg.norm(mug_up_w) + 1e-9)
            rim_lip = centers[i] + mug_up_w * RIM_PROJ + pour * 0.032   # cup opening edge
            cand = sorted([k for k in range(M) if not released[k]],
                          key=lambda k: proj_up[k], reverse=True)
            for k in cand[:OVERFLOW_RATE]:
                released[k] = True
                p_w[k] = rim_lip                       # exit at the lip, outside the wall
                v_w[k] = v_mug + pour * 0.35 + mug_up_w * 0.05
        # 3) ground clamp + render
        for k in range(M):
            if p_w[k][2] < 0.012 + P_RADIUS:
                p_w[k][2] = 0.012 + P_RADIUS
                if released[k]:
                    v_w[k] = np.zeros(3)
            particles[k].set_pose(sapien.Pose(p_w[k].tolist()))

        img = take_rgb(scene, cam)
        n_rel = int(released.sum())
        col = "r" if spill_state else "g"
        frame = annotate_frame(img, label,
                      [(f"t = {i*dt:.2f}/{(N-1)*dt:.2f}s", "k"),
                       (f"tilt = {tilt_deg:.0f}deg", col),
                       (f"spilled: {n_rel}/{M}", col)],
                      title_bg=(255, 224, 224) if spill_state else (224, 240, 224))
        frames.append(overlay_path(frame, px, i))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["minjerk", "chomp", "chomp_spill", "compare"],
                    default="compare")
    ap.add_argument("--duration-ms", type=int, default=80)
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d['theta_max_deg']); dt = float(d['dt'])
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    scene, robot, mug, particles, cam, sapien_order = build_scene(g)

    methods = ["minjerk", "chomp_spill"] if args.method == "compare" else [args.method]
    rendered = {}
    for m in methods:
        print(f"[render] {m}", flush=True)
        rendered[m] = simulate(scene, robot, mug, particles, cam, sapien_order,
                               d[f"{m}_q"], xhand, T_em, fk, d[f"{m}_tilt"],
                               theta_max, METHOD_LABEL[m], dt)

    if args.method == "compare":
        N = min(len(rendered[m]) for m in methods)
        combined = [np.concatenate([rendered[m][i] for m in methods], axis=1)
                    for i in range(N)]
        out = ROOT / "figs/video_transport.gif"
        save_gif(combined, out, args.duration_ms)
    else:
        out = ROOT / f"figs/video_transport_{args.method}.gif"
        save_gif(rendered[args.method], out, args.duration_ms)


if __name__ == "__main__":
    main()
