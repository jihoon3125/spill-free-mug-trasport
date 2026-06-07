"""Pedagogical demo: 'Even an upright cup spills when accelerated.'

Removes the robot from the scene to focus the viewer on cup + water.

Setup:
  - Mug is kinematic, ALWAYS perfectly upright (orientation locked).
  - Mug center translates linearly along world +y between two waypoints
    (60 cm horizontal motion at z = 0.3 m).
  - Three time budgets: SLOW (T=2s), MEDIUM (T=1s), FAST (T=0.4s).
  - Min-jerk velocity profile (rest-to-rest, smooth accel).

Physics:
  Effective gravity g_eff(t) = g_world − a_mug(t).
  When |g_eff tilt from mug-up| > θ_max, water 'spills' — released particles
  fly off in the appropriate horizontal direction under free fall.

Composite output: 3-column side-by-side GIF + per-condition GIFs.
Each panel labels:  cup orientation = 0° (always upright)
                    max accel        (in g's)
                    spilled count
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sapien
import PIL.Image
from PIL import ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import look_at_quat
from mp.constants import R_UPRIGHT

G_WORLD = np.array([0.0, 0.0, -9.81])


def minjerk(p0, p1, N):
    tau = np.linspace(0, 1, N)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    p = (1 - s)[:, None] * p0[None, :] + s[:, None] * p1[None, :]
    return p


def build_scene(n_particles=40, mug_scale=0.12):
    scene = sapien.Scene(); scene.set_timestep(1/240); scene.add_ground(0)
    scene.set_ambient_light([0.55, 0.55, 0.55])
    scene.add_directional_light([0, 0.5, -1], [0.9, 0.9, 0.9])

    b = scene.create_actor_builder()
    b.add_visual_from_file(str(ROOT / "data/mug_mesh.obj"), scale=[mug_scale]*3)
    mug = b.build_kinematic(name="mug")

    tb = scene.create_actor_builder()
    tb.add_box_visual(half_size=[0.5, 0.55, 0.005],
                      material=sapien.render.RenderMaterial(base_color=[0.78, 0.68, 0.55, 1]))
    table = tb.build_kinematic(name="table"); table.set_pose(sapien.Pose([0.0, 0.0, 0.0]))

    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < n_particles:
        x = rng.uniform(-0.045, 0.045); z = rng.uniform(-0.045, 0.045)
        if x*x + z*z <= 0.045**2:
            pts.append([x, rng.uniform(-0.05, 0.035), z])
    pts = np.asarray(pts)
    mat = sapien.render.RenderMaterial(base_color=[0.15, 0.55, 0.95, 1.0], roughness=0.3)
    particles = []
    for i in range(n_particles):
        pb = scene.create_actor_builder()
        pb.add_sphere_visual(radius=0.014, material=mat)
        particles.append(pb.build_kinematic(name=f"p_{i}"))

    cam = scene.add_camera("cam", 720, 540, fovy=0.7, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[0.65, -0.55, 0.55],
                              q=look_at_quat([0.65, -0.55, 0.55], [0.0, 0.0, 0.32])))
    return scene, mug, particles, pts, cam


def render_frame(scene, cam):
    scene.update_render(); cam.take_picture()
    img = cam.get_picture("Color")[..., :3]
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def simulate(scene, mug, particles, pts_local, cam,
              path, dt, theta_max_deg=18.0, label_top="", color_label="green"):
    """Run one trajectory. mug rigidly upright; particles attached, release
    when effective gravity tilt > θ_max."""
    G_W = G_WORLD
    N = path.shape[0]; M = len(pts_local)
    released = np.zeros(M, dtype=bool)
    p_w = np.zeros((M, 3)); v_w = np.zeros((M, 3))
    mug_up_local = np.array([0.0, 1.0, 0.0])
    cos_thmax = np.cos(np.radians(theta_max_deg))
    R_mug = R_UPRIGHT
    qxyzw = R.from_matrix(R_mug).as_quat()
    mug_quat = [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]
    mug_up_w = R_mug @ mug_up_local

    accel_max = 0.0
    frames = []
    for i in range(N):
        t_mug = path[i]
        mug.set_pose(sapien.Pose(t_mug.tolist(), mug_quat))
        # velocity + accel from path central diff
        if 0 < i < N - 1:
            v = (path[i+1] - path[i-1]) / (2 * dt)
            a = (path[i+1] - 2*path[i] + path[i-1]) / (dt**2)
        else:
            v = np.zeros(3); a = np.zeros(3)
        accel_max = max(accel_max, np.linalg.norm(a))
        g_eff = G_W - a
        g_eff_hat = g_eff / (np.linalg.norm(g_eff) + 1e-9)
        align = -np.dot(g_eff_hat, mug_up_w)
        # equivalent tilt:  align = cos(θ)
        theta_deg = np.degrees(np.arccos(np.clip(align, -1, 1)))
        spill_state = theta_deg > theta_max_deg

        # local effective up direction (perpendicular to water surface)
        g_eff_local = R_mug.T @ g_eff
        up_local_eff = -g_eff_local / (np.linalg.norm(g_eff_local) + 1e-9)
        for k in range(M):
            if released[k]:
                p_w[k] = p_w[k] + v_w[k] * dt + 0.5 * G_W * dt**2
                v_w[k] = v_w[k] + G_W * dt
            else:
                p_w[k] = R_mug @ pts_local[k] + t_mug
                proj = np.dot(pts_local[k], up_local_eff)
                if spill_state and proj > 0.02:
                    released[k] = True
                    v_w[k] = v + 0.4 * (R_mug @ mug_up_local) + 0.5 * (g_eff_hat * -1.0)
            if p_w[k][2] < 0.012:
                p_w[k][2] = 0.012
                if released[k]:
                    v_w[k] = np.zeros(3)
            particles[k].set_pose(sapien.Pose(p_w[k].tolist()))
        img = render_frame(scene, cam)
        # Annotate
        im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
        try:
            ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            f_orient = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
            fd = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
        except OSError:
            ft = ImageFont.load_default(); f_orient = ImageFont.load_default(); fd = ImageFont.load_default()
        title_bg = {"green": (200, 230, 200), "orange": (255, 220, 180), "red": (255, 200, 200)}[color_label]
        draw.rectangle([(0, 0), (im.width, 42)], fill=title_bg)
        draw.text((12, 9), label_top, fill=(20, 20, 20), font=ft)
        # Second banner row: cup orientation always 0° (the KEY message)
        draw.rectangle([(0, 42), (im.width, 72)], fill=(230, 245, 230))
        draw.text((12, 47), "Cup orientation: 0° (UPRIGHT throughout)",
                   fill=(20, 90, 20), font=f_orient)
        # Bottom panel: dynamics readouts
        box_w = 240
        bg = (255, 220, 220) if spill_state else (220, 240, 220)
        fg = (160, 20, 20) if spill_state else (30, 100, 30)
        draw.rectangle([(im.width-box_w, im.height-88), (im.width, im.height)], fill=bg)
        draw.text((im.width-box_w+8, im.height-86),
                   f"|a| = {np.linalg.norm(a):.1f} m/s² ({np.linalg.norm(a)/9.81:.2f}g)", fill=fg, font=fd)
        draw.text((im.width-box_w+8, im.height-64),
                   f"effective tilt = {theta_deg:.1f}°", fill=fg, font=fd)
        draw.text((im.width-box_w+8, im.height-42),
                   f"spilled {int(released.sum())}/{M}", fill=fg, font=fd)
        if spill_state:
            draw.text((im.width-box_w+8, im.height-20),
                       "→ SPILL!", fill=(180, 0, 0), font=fd)
        frames.append(np.array(im))
    return frames, accel_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=60)
    ap.add_argument("--n-particles", type=int, default=40)
    ap.add_argument("--theta-max-deg", type=float, default=18.0)
    ap.add_argument("--duration-ms", type=int, default=100)
    args = ap.parse_args()

    p_start = np.array([0.0, -0.3, 0.30])
    p_goal = np.array([0.0,  0.3, 0.30])

    conditions = [
        ("SLOW  (T=2.0s)",  2.0, "green"),
        ("MED.  (T=1.0s)",  1.0, "orange"),
        ("FAST  (T=0.4s)",  0.4, "red"),
    ]

    per_cond = {}
    for label, T_budget, color in conditions:
        print(f"\n=== {label}")
        scene, mug, particles, pts_local, cam = build_scene(n_particles=args.n_particles)
        path = minjerk(p_start, p_goal, args.N)
        dt = T_budget / (args.N - 1)
        frames, accel_max = simulate(scene, mug, particles, pts_local, cam,
                                      path, dt, args.theta_max_deg, label, color)
        per_cond[label] = frames
        print(f"  max |a_mug| = {accel_max:.2f} m/s² ({accel_max/9.81:.2f}g)")
        # Save per-condition GIF
        safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "p").replace("=", "")
        outp = ROOT / f"figs/anim_upright_{safe}.gif"
        pil = [PIL.Image.fromarray(f) for f in frames]
        pil[0].save(outp, save_all=True, append_images=pil[1:],
                    duration=args.duration_ms, loop=0, optimize=False)
        print(f"  saved {outp}")

    # Composite 3-column side-by-side
    labels = [c[0] for c in conditions]
    N_frames = max(len(per_cond[l]) for l in labels)
    composite = []
    for i in range(N_frames):
        row = np.concatenate([per_cond[l][min(i, len(per_cond[l])-1)] for l in labels], axis=1)
        composite.append(row)
    pil = [PIL.Image.fromarray(f) for f in composite]
    outp = ROOT / "figs/anim_upright_spill_demo.gif"
    pil[0].save(outp, save_all=True, append_images=pil[1:],
                duration=args.duration_ms, loop=0, optimize=False)
    print(f"\nsaved composite {outp}  ({len(pil)} frames)")


if __name__ == "__main__":
    main()
