"""GIF v3: actual water spilling physics.

Each particle is one of:
  - 'in_cup': rigidly attached to mug. Position = T_world_mug @ p_local.
  - 'free':   detached, integrating free-fall p += v*dt, v += g*dt.

Release rule: a particle goes 'free' when its position in MUG LOCAL FRAME
satisfies BOTH:
  1) it is above the water surface as defined by the effective gravity, i.e.
     the local-frame component along g_eff^local exceeds the rim threshold
  2) the effective gravity exits the cup opening (cup tilt > θ_max)
At release we give it the world-frame velocity of the mug center at that step
(plus a small upward kick along mug +y).

After release, the particle ignores everything except gravity (no collision).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
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

JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]
METHODS = ["minjerk", "chomp", "chomp_spill"]
METHOD_LABEL = {"minjerk": "Min-jerk (naive)",
                "chomp": "CHOMP (smooth)",
                "chomp_spill": "CHOMP + spill (ours)"}
G_WORLD = np.array([0.0, 0.0, -9.81])


def build_scene(g, n_particles=40):
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

    # Particle local positions: cylindrical water column (mug local: y is up)
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < n_particles:
        x = rng.uniform(-0.045, 0.045); z = rng.uniform(-0.045, 0.045)
        if x * x + z * z <= 0.045 ** 2:
            pts.append([x, rng.uniform(-0.05, 0.04), z])     # y ∈ [-0.05, 0.04] cup interior
    pts = np.asarray(pts)
    mat = sapien.render.RenderMaterial(base_color=[0.15, 0.55, 0.95, 1.0],
                                        metallic=0.0, roughness=0.3)
    particles = []
    for i in range(n_particles):
        pb = scene.create_actor_builder()
        pb.add_sphere_visual(radius=0.013, material=mat)
        particles.append(pb.build_kinematic(name=f"p_{i}"))

    cam = scene.add_camera("cam", 720, 540, fovy=0.7, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[0.9, -0.55, 0.65],
                              q=look_at_quat([0.9, -0.55, 0.65], [0.5, 0.05, 0.33])))
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


def simulate_method(scene, robot, mug, particles, pts_local, cam, sapien_order,
                     q_traj, xhand_qpos, T_em, fk, tilt_arr, theta_max, method_label,
                     dt, mug_up_local=np.array([0.0, 1.0, 0.0])):
    """Walk through the trajectory; release particles when spill condition exceeded.
    Returns the list of rendered frames."""
    frames = []
    N = q_traj.shape[0]
    M = len(pts_local)
    released = np.zeros(M, dtype=bool)
    p_w_state = np.zeros((M, 3))      # world position (for free particles)
    v_w_state = np.zeros((M, 3))      # world velocity (for free particles)
    rim_y_local = 0.05                 # mug local +y rim height

    # Precompute mug center world history for velocity estimation
    mug_centers_world = np.zeros((N, 3))
    mug_R_world = np.zeros((N, 3, 3))
    for i in range(N):
        with torch.no_grad():
            T_ee = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
        T_mug = T_ee @ T_em
        mug_centers_world[i] = T_mug[:3, 3]
        mug_R_world[i] = T_mug[:3, :3]

    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        R_mug = mug_R_world[i]; t_mug = mug_centers_world[i]
        qxyzw = R.from_matrix(R_mug).as_quat()
        mug.set_pose(sapien.Pose(t_mug.tolist(),
                                  [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
        # Mug velocity (central diff)
        if 0 < i < N - 1:
            v_mug = (mug_centers_world[i + 1] - mug_centers_world[i - 1]) / (2 * dt)
        elif i == 0:
            v_mug = (mug_centers_world[1] - mug_centers_world[0]) / dt
        else:
            v_mug = (mug_centers_world[-1] - mug_centers_world[-2]) / dt
        # Effective gravity in WORLD frame (a = approx 0 here at frame level — use full traj accel)
        if 0 < i < N - 1:
            a_mug = (mug_centers_world[i + 1] - 2 * mug_centers_world[i] + mug_centers_world[i - 1]) / (dt ** 2)
        else:
            a_mug = np.zeros(3)
        g_eff_w = G_WORLD - a_mug
        # Effective gravity in MUG LOCAL frame: g_eff_local = R_mug^T @ g_eff_w
        g_eff_local = R_mug.T @ g_eff_w
        # If g_eff projected onto mug +y is POSITIVE → effective gravity points UP through opening → spill
        # (Equivalent to tilt > 90°). For partial tilts < 90°, water still sloshes; release particles whose
        # local position's y projection on the local-up axis exceeds the rim.
        spill_state = tilt_arr[i] > theta_max

        for k in range(M):
            if released[k]:
                # free-fall integrate
                p_w_state[k] = p_w_state[k] + v_w_state[k] * dt + 0.5 * G_WORLD * dt ** 2
                v_w_state[k] = v_w_state[k] + G_WORLD * dt
            else:
                # rigidly attached
                p_w_state[k] = R_mug @ pts_local[k] + t_mug
                # Release condition: spill_state AND this particle is on the 'high' side of the cup
                #   high side = projection of local position onto direction OPPOSITE to local g_eff is high.
                # Use the local g_eff direction: water 'falls' along -g_eff_local in local frame.
                # Particle 'rises' (toward rim) in direction +(-g_eff_local) = -g_eff_local.
                up_local_eff = -g_eff_local / (np.linalg.norm(g_eff_local) + 1e-9)
                proj = np.dot(pts_local[k], up_local_eff)
                # Approx: rim is reached when proj > 0.04 (mug rim y) — but in tilted frame
                # this becomes the 'high lip' which is closer. Use 0.02 as effective threshold.
                if spill_state and proj > 0.02:
                    released[k] = True
                    # Give it mug's linear velocity + small upward kick
                    v_w_state[k] = v_mug + 0.3 * (R_mug @ mug_up_local)
            # Render: if hit the ground, freeze on ground
            if p_w_state[k][2] < 0.012:
                p_w_state[k][2] = 0.012
                if released[k]:
                    v_w_state[k] = np.zeros(3)
            particles[k].set_pose(sapien.Pose(p_w_state[k].tolist()))

        img = render_frame(scene, cam)
        # Annotation
        im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
        try:
            font_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_d = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        except OSError:
            font_t = ImageFont.load_default(); font_d = ImageFont.load_default()
        draw.rectangle([(0, 0), (im.width, 34)], fill=(245, 245, 245))
        draw.text((10, 6), method_label, fill=(20, 20, 20), font=font_t)
        bg = (255, 220, 220) if spill_state else (220, 240, 220)
        fg = (180, 30, 30) if spill_state else (40, 110, 40)
        n_rel = int(released.sum())
        draw.rectangle([(im.width - 175, im.height - 56), (im.width, im.height)], fill=bg)
        draw.text((im.width - 168, im.height - 52),
                   f"tilt={tilt_arr[i]:.1f}°", fill=fg, font=font_d)
        draw.text((im.width - 168, im.height - 30),
                   f"spilled: {n_rel}/{M}", fill=fg, font=font_d)
        frames.append(np.array(im))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-ms", type=int, default=100,
                    help="GIF frame duration in ms (100ms → 10fps)")
    ap.add_argument("--n-particles", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "figs/anim_3way_water.gif"))
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d['theta_max_deg']); dt = float(d['dt'])
    g_info = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand_qpos = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    scene, robot, mug, particles, pts_local, cam, sapien_order = build_scene(g_info, n_particles=args.n_particles)

    per_method = {}
    for method in METHODS:
        print(f"[render] {method}")
        q_traj = d[f"{method}_q"]; tilt = d[f"{method}_tilt"]
        frames = simulate_method(scene, robot, mug, particles, pts_local, cam,
                                  sapien_order, q_traj, xhand_qpos, T_em, fk, tilt,
                                  theta_max, METHOD_LABEL[method], dt)
        per_method[method] = frames
        outp = ROOT / f"figs/anim_{method}_water.gif"
        pil = [PIL.Image.fromarray(f) for f in frames]
        pil[0].save(outp, save_all=True, append_images=pil[1:],
                    duration=args.duration_ms, loop=0, optimize=False)
        print(f"  saved {outp}")

    # Side-by-side
    N = min(len(per_method[m]) for m in METHODS)
    combined = [np.concatenate([per_method[m][i] for m in METHODS], axis=1) for i in range(N)]
    pil = [PIL.Image.fromarray(f) for f in combined]
    pil[0].save(args.out, save_all=True, append_images=pil[1:],
                duration=args.duration_ms, loop=0, optimize=False)
    print(f"saved {args.out}  ({N} frames, side-by-side)")


if __name__ == "__main__":
    main()
