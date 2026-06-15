"""REAL rigid-body water sim: dynamic sphere particles in a cup-shaped cavity,
carried along the planned trajectory, simulated by SAPIEN PhysX.

Unlike make_video_transport.py (kinematic illustration of the analytic spill
model), here the particles are dynamic rigid bodies. The cup is a kinematic
container (bottom disk + ring of wall boxes) driven along the trajectory; PhysX
resolves all particle-wall and particle-particle contacts. Spilling emerges from
the physics, not from a scripted rule.

Caveat: rigid spheres are a granular approximation of liquid (no surface tension
/ viscosity). This is an independent cross-check of the analytic spill model.

Usage:
  python mp/make_video_sim.py --method compare
Output: figs/video_sim.gif
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sapien
import torch
from scipy.spatial.transform import Rotation as Rot, Slerp

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
METHOD_LABEL = {"minjerk": "Naive (min-jerk)", "chomp_spill": "Ours (CHOMP + spill)"}

# Cup cavity geometry in MUG LOCAL frame (opening = -y, base = +y).
R_IN = 0.033            # inner radius
WALL_T = 0.004          # wall thickness
BASE_Y = 0.055          # bottom (base) plane
OPEN_Y = -0.005         # rim (opening) plane  → cavity depth ~6 cm
OVERFILL_Y = -0.030     # grid-fill up to here (above the rim) → settles to brim-full
N_WALL = 18             # wall segments
SUBSTEPS = 8            # physx steps per trajectory waypoint
SETTLE_STEPS = 140      # settle; excess (above brim) overflows and is removed
P_RADIUS = 0.0065
N_PARTICLES = 150       # overfill cap; trimmed to the contained set after settling


def add_cup(scene, mat):
    """Kinematic cup: bottom disk + ring of wall boxes (collision + visual)."""
    b = scene.create_actor_builder()
    cav_half = (BASE_Y - OPEN_Y) / 2
    y_mid = (BASE_Y + OPEN_Y) / 2
    wall_vis = sapien.render.RenderMaterial(base_color=[0.75, 0.8, 0.85, 1.0],
                                            metallic=0.0, roughness=0.4)
    # bottom
    b.add_box_collision(pose=sapien.Pose([0, BASE_Y, 0]),
                        half_size=[R_IN + WALL_T, WALL_T, R_IN + WALL_T], material=mat)
    b.add_box_visual(pose=sapien.Pose([0, BASE_Y, 0]),
                     half_size=[R_IN + WALL_T, WALL_T, R_IN + WALL_T], material=wall_vis)
    # walls
    tang = (2 * np.pi * R_IN / N_WALL) * 0.62      # half-width, slight overlap
    for k in range(N_WALL):
        phi = 2 * np.pi * k / N_WALL
        pos = sapien.Pose([R_IN * np.cos(phi), y_mid, R_IN * np.sin(phi)],
                          Rot.from_rotvec([0, -phi, 0]).as_quat()[[3, 0, 1, 2]])
        b.add_box_collision(pose=pos, half_size=[WALL_T, cav_half, tang], material=mat)
        b.add_box_visual(pose=pos, half_size=[WALL_T, cav_half, tang], material=wall_vis)
    return b.build_kinematic(name="cup")


def add_particles(scene, mat, n):
    pmat = sapien.render.RenderMaterial(base_color=[0.15, 0.55, 0.95, 1.0],
                                        metallic=0.0, roughness=0.2)
    parts = []
    for i in range(n):
        b = scene.create_actor_builder()
        b.add_sphere_collision(radius=P_RADIUS, material=mat, density=1000)
        b.add_sphere_visual(radius=P_RADIUS, material=pmat)
        parts.append(b.build(name=f"w_{i}"))
    return parts


def initial_particle_offsets():
    """Grid-fill the cavity from the base up toward the brim (non-overlapping),
    so the cup starts ~full like a real mug. Returns first N_PARTICLES slots."""
    step = 2.15 * P_RADIUS
    rmax = R_IN - P_RADIUS
    xs = np.arange(-rmax, rmax + 1e-9, step)
    pts = []
    y = BASE_Y - WALL_T - P_RADIUS                      # bottom layer
    while len(pts) < N_PARTICLES and y > OVERFILL_Y:    # overfill above the rim
        for x in xs:
            for z in xs:
                if x * x + z * z <= rmax ** 2:
                    pts.append([x, y, z])
        y -= step                                       # next layer up (toward opening)
    return np.asarray(pts[:N_PARTICLES])


def cup_poses(fk, q_traj, T_em):
    poses = []
    for i in range(q_traj.shape[0]):
        with torch.no_grad():
            T = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy() @ T_em
        poses.append((T[:3, 3].copy(), T[:3, :3].copy()))
    return poses


def simulate(method, q_traj, T_em, fk, dt, theta_max):
    scene = sapien.Scene(); scene.set_timestep(dt / SUBSTEPS)
    scene.add_ground(-0.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.9, 0.9, 0.9])
    pmat = sapien.physx.PhysxMaterial(0.4, 0.3, 0.05)
    cup = add_cup(scene, pmat)
    offs = initial_particle_offsets()
    M = len(offs)
    parts = add_particles(scene, pmat, M)
    cam = scene.add_camera("cam", 640, 560, fovy=0.95, near=0.05, far=10.0)
    eye = [1.15, -0.95, 0.80]; cam.set_pose(sapien.Pose(p=eye, q=look_at_quat(eye, [0.5, 0.05, 0.30])))

    poses = cup_poses(fk, q_traj, T_em)
    t0, R0 = poses[0]
    cup.set_pose(sapien.Pose(t0.tolist(), Rot.from_matrix(R0).as_quat()[[3, 0, 1, 2]]))
    for k in range(M):
        wp = R0 @ offs[k] + t0
        parts[k].set_pose(sapien.Pose(wp.tolist()))

    for _ in range(SETTLE_STEPS):
        scene.step()
    # trim excess that overflowed during settling → clean brim-full baseline
    kept = []
    for p in parts:
        if p.get_pose().p[2] > t0[2] - 0.08:
            kept.append(p)
        else:
            scene.remove_actor(p)
    parts = kept
    M = len(parts)

    def render():
        scene.update_render(); cam.take_picture()
        img = cam.get_picture("Color")[..., :3]
        return (img * 255).clip(0, 255).astype(np.uint8) if img.dtype != np.uint8 else img

    frames = [render()]
    N = len(poses)
    for i in range(N - 1):
        (pa, Ra), (pb, Rb) = poses[i], poses[i + 1]
        slerp = Slerp([0, 1], Rot.from_matrix([Ra, Rb]))
        for s in range(1, SUBSTEPS + 1):
            a = s / SUBSTEPS
            pos = (1 - a) * pa + a * pb
            quat = slerp([a])[0].as_quat()[[3, 0, 1, 2]]
            cup.set_pose(sapien.Pose(pos.tolist(), quat))
            scene.step()
        frames.append(render())

    # count spilled = particles that fell well below the cup
    cup_z = poses[-1][0][2]
    spilled = sum(1 for p in parts if p.get_pose().p[2] < cup_z - 0.10)
    return frames, spilled, M


def annotate(img, label, spilled, M):
    import PIL.Image
    from PIL import ImageDraw, ImageFont
    im = PIL.Image.fromarray(img.copy()); d = ImageDraw.Draw(im)
    try:
        ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        fd = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except OSError:
        ft = fd = ImageFont.load_default()
    bad = spilled > 0
    d.rectangle([(0, 0), (im.width, 38)], fill=(255, 224, 224) if bad else (224, 240, 224))
    d.text((12, 8), label, fill=(20, 20, 20), font=ft)
    d.rectangle([(im.width - 200, im.height - 32), (im.width, im.height)],
                fill=(245, 245, 245))
    d.text((im.width - 192, im.height - 28), f"spilled: {spilled}/{M}",
           fill=(170, 30, 30) if bad else (30, 110, 30), font=fd)
    return np.array(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["minjerk", "chomp_spill", "compare"], default="compare")
    ap.add_argument("--duration-ms", type=int, default=80)
    args = ap.parse_args()
    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d['theta_max_deg']); dt = float(d['dt'])
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    methods = ["minjerk", "chomp_spill"] if args.method == "compare" else [args.method]
    rendered = {}
    for m in methods:
        print(f"[sim] {m}", flush=True)
        frames, spilled, M = simulate(m, d[f"{m}_q"], T_em, fk, dt, theta_max)
        print(f"  spilled {spilled}/{M}", flush=True)
        rendered[m] = [annotate(f, METHOD_LABEL[m], spilled, M) for f in frames]

    import PIL.Image
    if args.method == "compare":
        N = min(len(rendered[m]) for m in methods)
        out_frames = [np.concatenate([rendered[m][i] for m in methods], axis=1) for i in range(N)]
        out = ROOT / "figs/video_sim.gif"
    else:
        out_frames = rendered[args.method]; out = ROOT / f"figs/video_sim_{args.method}.gif"
    pil = [PIL.Image.fromarray(f) for f in out_frames]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=args.duration_ms, loop=0)
    print(f"saved {out}  ({len(pil)} frames)", flush=True)


if __name__ == "__main__":
    main()
