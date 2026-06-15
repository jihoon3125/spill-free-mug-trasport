"""Improved GIF: slow playback + closer camera + effective-gravity vector arrow.

The arrow shows where 'water perceives down' (g_eff = g − a_mug). When it
diverges from the mug's up axis beyond θ_max, the mug is highlighted in red
('water would spill'). This makes the dynamic difference between methods
visible (tilt alone is subtle when cups are mostly upright).
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

    # Particles inside cup (mug local: cylinder half-fill at y ∈ [-0.04, 0.03], r ≤ 0.045)
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < n_particles:
        x = rng.uniform(-0.045, 0.045); z = rng.uniform(-0.045, 0.045)
        if x*x + z*z <= 0.045**2:
            pts.append([x, rng.uniform(-0.04, 0.03), z])
    pts = np.asarray(pts)
    blue_mat = sapien.render.RenderMaterial(base_color=[0.15, 0.45, 0.95, 1.0], roughness=0.3)
    red_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.20, 0.15, 1.0], roughness=0.3)
    particles_blue, particles_red = [], []
    for i in range(n_particles):
        pb = scene.create_actor_builder(); pb.add_sphere_visual(radius=0.014, material=blue_mat)
        particles_blue.append(pb.build_kinematic(name=f"pb_{i}"))
        pr = scene.create_actor_builder(); pr.add_sphere_visual(radius=0.014, material=red_mat)
        particles_red.append(pr.build_kinematic(name=f"pr_{i}"))

    # Camera — much closer, framed on the mug working volume
    cam = scene.add_camera("cam", 720, 540, fovy=0.7, near=0.05, far=10.0)
    cam.set_pose(sapien.Pose(p=[0.9, -0.55, 0.65],
                              q=look_at_quat([0.9, -0.55, 0.65], [0.5, 0.05, 0.33])))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, particles_blue, particles_red, pts, cam, sapien_order


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


def world_to_screen(p_w, cam):
    """Project world-space point into screen pixels for arrow overlay."""
    # Sapien camera intrinsics + extrinsics
    K = cam.get_intrinsic_matrix()                     # (3,3)
    pose = cam.get_model_matrix()                       # cam→world 4x4
    # Need world→cam: invert pose
    world_to_cam = np.linalg.inv(pose)
    p_cam = world_to_cam @ np.append(p_w, 1.0)
    # Sapien camera looks down -Z by default in the camera frame.
    # Actually sapien uses x-forward, y-left, z-up convention (need verification).
    # If get_intrinsic_matrix assumes OpenCV camera (z-forward, y-down),
    # we need an axis swap. Let's use OpenCV-like fallback:
    # p_cv = (x=-y_sapien, y=-z_sapien, z=x_sapien)
    p_cv = np.array([-p_cam[1], -p_cam[2], p_cam[0]])
    if p_cv[2] <= 1e-3:                                 # behind camera
        return None
    uv = K @ p_cv
    uv = uv[:2] / uv[2]
    return uv


def draw_gravity_arrow(im, p_mug, mug_up_w, g_eff_dir, cam, spill, scale_px=80):
    """Overlay two arrows from the mug top: green=mug up, red=g_eff (down-flipped for clarity).
    spill=True → outline arrow red."""
    H, W, _ = np.asarray(im).shape
    # Pre-compute screen positions
    base_uv = world_to_screen(p_mug, cam)
    if base_uv is None:
        return im
    up_world = p_mug + 0.18 * mug_up_w
    up_uv = world_to_screen(up_world, cam)
    g_world = p_mug - 0.18 * g_eff_dir / max(np.linalg.norm(g_eff_dir), 1e-9)  # opposite to g_eff so arrow points 'up against gravity'
    g_uv = world_to_screen(g_world, cam)
    if up_uv is None or g_uv is None:
        return im
    pim = PIL.Image.fromarray(np.asarray(im).copy())
    draw = ImageDraw.Draw(pim)
    # mug-up arrow (green)
    draw.line([(int(base_uv[0]), int(base_uv[1])), (int(up_uv[0]), int(up_uv[1]))],
               fill=(20, 180, 60), width=4)
    # g_eff (anti-g) arrow (red if spill, else orange)
    g_color = (220, 30, 30) if spill else (240, 130, 30)
    draw.line([(int(base_uv[0]), int(base_uv[1])), (int(g_uv[0]), int(g_uv[1]))],
               fill=g_color, width=4)
    return np.array(pim)


def render_trajectory_with_arrows(scene, robot, mug, particles_blue, particles_red,
                                    pts_local, cam, sapien_order, q_traj, xhand_qpos,
                                    T_em, fk, tilt_arr, p_mug_arr, theta_max, method_label,
                                    dt):
    frames = []
    N = q_traj.shape[0]
    # Compute mug accelerations (central diff) for g_eff arrow
    a = np.zeros_like(p_mug_arr)
    if N >= 3:
        a[1:-1] = (p_mug_arr[2:] - 2 * p_mug_arr[1:-1] + p_mug_arr[:-2]) / (dt ** 2)
        a[0] = a[1]; a[-1] = a[-2]
    g_world = np.array([0.0, 0.0, -9.81])
    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        with torch.no_grad():
            T_ee = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
        T_mug = T_ee @ T_em
        qxyzw = R.from_matrix(T_mug[:3, :3]).as_quat()
        mug.set_pose(sapien.Pose(T_mug[:3, 3].tolist(),
                                  [qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]]))
        spill = tilt_arr[i] > theta_max
        mug_up_w = T_mug[:3, :3] @ np.array([0.0, 1.0, 0.0])
        for pl, blue_a, red_a in zip(pts_local, particles_blue, particles_red):
            p_w = (T_mug[:3, :3] @ pl) + T_mug[:3, 3]
            if spill:
                red_a.set_pose(sapien.Pose(p_w.tolist()))
                blue_a.set_pose(sapien.Pose([0, 0, -5]))
            else:
                blue_a.set_pose(sapien.Pose(p_w.tolist()))
                red_a.set_pose(sapien.Pose([0, 0, -5]))
        img = render_frame(scene, cam)
        # Overlay gravity arrow (above mug rim)
        p_mug_top = p_mug_arr[i] + 0.04 * mug_up_w   # a bit above center for visibility
        g_eff = g_world - a[i]
        img = draw_gravity_arrow(img, p_mug_top, mug_up_w, g_eff, cam, spill)
        # Title + tilt panel
        im = PIL.Image.fromarray(img.copy()); draw = ImageDraw.Draw(im)
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_tilt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except OSError:
            font_title = ImageFont.load_default(); font_tilt = ImageFont.load_default()
        draw.rectangle([(0, 0), (im.width, 34)], fill=(245, 245, 245))
        draw.text((10, 6), method_label, fill=(20, 20, 20), font=font_title)
        bg = (255, 220, 220) if spill else (220, 240, 220)
        fg = (180, 30, 30) if spill else (40, 110, 40)
        draw.rectangle([(im.width - 150, im.height - 34), (im.width, im.height)], fill=bg)
        draw.text((im.width - 145, im.height - 30),
                   f"tilt={tilt_arr[i]:.1f}°", fill=fg, font=font_tilt)
        # Legend at bottom-left
        draw.rectangle([(0, im.height - 50), (270, im.height)], fill=(245, 245, 245))
        draw.line([(10, im.height - 36), (40, im.height - 36)], fill=(20, 180, 60), width=4)
        draw.text((46, im.height - 46), "mug ↑ (cup up)", fill=(20, 20, 20), font=font_tilt)
        draw.line([(10, im.height - 16), (40, im.height - 16)], fill=(240, 130, 30), width=4)
        draw.text((46, im.height - 26), "−g_eff (water ↑)", fill=(20, 20, 20), font=font_tilt)
        frames.append(np.array(im))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-ms", type=int, default=100,
                    help="GIF frame duration in ms (100ms → 10 fps, 0.2× real time)")
    ap.add_argument("--out", default=str(ROOT / "figs/anim_3way_v2.gif"))
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d['theta_max_deg'])
    dt = float(d['dt'])
    g_info = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand_qpos = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    scene, robot, mug, pb, pr, pts_local, cam, sapien_order = build_scene(g_info)

    per_method_frames = {}
    for method in METHODS:
        print(f"[render] {method}")
        q_traj = d[f"{method}_q"]; tilt = d[f"{method}_tilt"]; p_mug = d[f"{method}_pmug"]
        frames = render_trajectory_with_arrows(
            scene, robot, mug, pb, pr, pts_local, cam, sapien_order,
            q_traj, xhand_qpos, T_em, fk, tilt, p_mug, theta_max, METHOD_LABEL[method], dt)
        per_method_frames[method] = frames
        outp = ROOT / f"figs/anim_{method}_v2.gif"
        pil = [PIL.Image.fromarray(f) for f in frames]
        pil[0].save(outp, save_all=True, append_images=pil[1:],
                    duration=args.duration_ms, loop=0, optimize=False)
        print(f"  saved {outp}")

    N = min(len(per_method_frames[m]) for m in METHODS)
    combined = [np.concatenate([per_method_frames[m][i] for m in METHODS], axis=1) for i in range(N)]
    pil = [PIL.Image.fromarray(f) for f in combined]
    pil[0].save(args.out, save_all=True, append_images=pil[1:],
                duration=args.duration_ms, loop=0, optimize=False)
    print(f"saved {args.out}  ({N} frames @ {args.duration_ms}ms each)")


if __name__ == "__main__":
    main()
