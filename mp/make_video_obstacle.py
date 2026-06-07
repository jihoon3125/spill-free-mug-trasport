"""Obstacle scenario video — robot dodges a box obstacle while keeping the
cup upright and water in.

Renders the 3 methods side-by-side with the box visible:
  - Min-jerk (naive) → mug crashes through box AND spills
  - CHOMP+obs (no spill) → avoids box but spills (detour induces accel)
  - CHOMP+obs+spill (ours) → avoids box AND no spill

Uses _video_utils flip convention so cups appear upright.
Output: figs/video_obstacle_3way.gif (3-col composite) + per-method gifs.
"""
from __future__ import annotations
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
from mp.ik import ik_solve
from mp.spill_cost import SpillCost, SpillConfig
from mp.obstacle import ObstacleCost, ObstacleConfig
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from mp._video_utils import (
    R_X_MESH_180, MUG_UP_LOCAL,
    flip_pose_for_render, mat_to_sapien_pose, sample_water_particles,
    compute_spill_state, render_camera, take_rgb, annotate_frame, save_gif,
    ROOT as VU_ROOT,
)

JOINT_ORDER = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]
R_UPRIGHT_NEW = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT_NEW; T[:3, 3] = np.asarray(p, float)
    return T


def build_scene(g, box_center, box_halfext, n_particles=30):
    scene = sapien.Scene(); scene.set_timestep(1/240); scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0.5, -1], [0.9, 0.9, 0.9])

    loader = scene.create_urdf_loader(); loader.fix_root_link = True
    robot = loader.load(str(URDF))
    robot.set_root_pose(sapien.Pose([0, 0, 0]))

    b = scene.create_actor_builder()
    b.add_visual_from_file(str(VU_ROOT / "data/mug_mesh.obj"), scale=[g['scale']]*3)
    mug = b.build_kinematic(name="mug")

    tb = scene.create_actor_builder()
    tb.add_box_visual(half_size=[0.5, 0.5, 0.005],
                      material=sapien.render.RenderMaterial(base_color=[0.7, 0.6, 0.5, 1]))
    table = tb.build_kinematic(name="table"); table.set_pose(sapien.Pose([0.5, 0.05, 0.0]))

    # Obstacle box — visible
    obs_b = scene.create_actor_builder()
    obs_b.add_box_visual(half_size=list(box_halfext),
                          material=sapien.render.RenderMaterial(base_color=[0.3, 0.3, 0.6, 1.0]))
    obs_actor = obs_b.build_kinematic(name="obstacle")
    obs_actor.set_pose(sapien.Pose(list(box_center)))

    pts_local = sample_water_particles(n_particles, radius=0.030, y_low=0.00, y_high=0.045)
    mat = sapien.render.RenderMaterial(base_color=[0.15, 0.55, 0.95, 1.0], roughness=0.3)
    particles = []
    for i in range(n_particles):
        pb = scene.create_actor_builder()
        pb.add_sphere_visual(radius=0.009, material=mat)
        particles.append(pb.build_kinematic(name=f"p_{i}"))

    cam = render_camera(scene, w=720, h=540, fovy=0.85,
                         eye=(1.0, -0.7, 0.7), target=(0.5, 0.05, 0.30))
    sapien_order = [j.name for j in robot.get_active_joints()]
    return scene, robot, mug, particles, pts_local, cam, sapien_order


def apply_qpos(robot, sapien_order, q_ur5e, xhand_qpos):
    qpos_18 = np.concatenate([q_ur5e, xhand_qpos])
    n2i = {n: i for i, n in enumerate(JOINT_ORDER)}
    qpos_sapien = np.array([qpos_18[n2i[n]] for n in sapien_order])
    robot.set_qpos(qpos_sapien)


def render_method(g_info, T_em, fk, xhand_qpos, q_traj, method_label, color_bg,
                    box_center, box_halfext, theta_max_deg=18.0):
    scene, robot, mug, particles, pts_local, cam, sapien_order = build_scene(
        g_info, box_center, box_halfext)
    N = q_traj.shape[0]
    dt = 1.5 / (N - 1)
    # Precompute mug pose history
    mug_centers = np.zeros((N, 3)); mug_Rw = np.zeros((N, 3, 3))
    for i in range(N):
        with torch.no_grad():
            T_ee_i = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
        T_mug_i = T_ee_i @ T_em
        mug_centers[i] = T_mug_i[:3, 3]
        mug_Rw[i] = T_mug_i[:3, :3]
    G_W = np.array([0.0, 0.0, -9.81])
    M = len(pts_local)
    released = np.zeros(M, dtype=bool)
    p_w_state = np.zeros((M, 3)); v_w_state = np.zeros((M, 3))
    frames = []
    # Collision check (mug bounding sphere vs box SDF)
    def sdf_box(p):
        q = np.abs(p - np.array(box_center)) - np.array(box_halfext)
        out = np.linalg.norm(np.maximum(q, 0)); ins = min(np.max(q), 0)
        return out + ins

    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        T_mug = np.eye(4); T_mug[:3, :3] = mug_Rw[i]; T_mug[:3, 3] = mug_centers[i]
        T_render = flip_pose_for_render(T_mug)
        mug.set_pose(mat_to_sapien_pose(T_render))

        if 0 < i < N - 1:
            v_mug = (mug_centers[i+1] - mug_centers[i-1]) / (2*dt)
            a_mug = (mug_centers[i+1] - 2*mug_centers[i] + mug_centers[i-1]) / (dt**2)
        else:
            v_mug = np.zeros(3); a_mug = np.zeros(3)

        spill_state, tilt_deg, g_eff, up_local_eff = compute_spill_state(
            T_mug, a_mug, theta_max_deg)
        # Collision (mug radius 0.07)
        d_clear = sdf_box(mug_centers[i]) - 0.07
        collision_state = d_clear < 0

        R_render = T_render[:3, :3]
        for k in range(M):
            if released[k]:
                p_w_state[k] = p_w_state[k] + v_w_state[k]*dt + 0.5*G_W*(dt**2)
                v_w_state[k] = v_w_state[k] + G_W*dt
            else:
                p_w_state[k] = R_render @ pts_local[k] + T_render[:3, 3]
                proj_render = np.dot(pts_local[k], np.array([0.0, +1.0, 0.0]))
                proj_eff = np.dot(pts_local[k], up_local_eff)
                if spill_state and (proj_render > 0.005 or proj_eff > 0.01):
                    released[k] = True
                    v_w_state[k] = v_mug + 0.4*(R_render @ np.array([0, 1, 0])) + 0.3*np.array([0, 0, 1])
            if p_w_state[k][2] < 0.012:
                p_w_state[k][2] = 0.012
                if released[k]:
                    v_w_state[k] = np.zeros(3)
            particles[k].set_pose(sapien.Pose(p_w_state[k].tolist()))

        img = take_rgb(scene, cam)
        lines = [
            (f"tilt = {tilt_deg:.1f}°", "r" if spill_state else "g"),
            (f"clearance = {d_clear*100:+.1f} cm", "r" if collision_state else "g"),
            (f"spilled {int(released.sum())}/{M}", "r" if spill_state else "g"),
        ]
        if collision_state:
            lines.append(("→ COLLISION!", "r"))
        img = annotate_frame(img, method_label, lines, title_bg=color_bg)
        frames.append(img)
    return frames


def main():
    g_info = parse_grasp_info(VU_ROOT / "data/grasp_info.txt")
    T_em = np.load(VU_ROOT / "data/ee_to_mug.npy")
    xhand_qpos = np.load(VU_ROOT / "data/xhand_qpos.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    # Setup matches run_obstacle.py
    p_start = (0.5, -0.3, 0.30); p_goal = (0.5, 0.4, 0.30)
    def ik(p, q_seed):
        T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy()
    q_start = ik(p_start, g_info['qpos'][:6])
    q_goal = ik(p_goal, q_start)

    N = 60; dt = 0.025; theta_max = 18.0
    box_center = (0.55, 0.05, 0.28); box_halfext = (0.10, 0.10, 0.10)
    obs_cfg = ObstacleConfig(box_center=box_center, box_halfext=box_halfext,
                              margin=0.04, T_ee_to_mug=T_em,
                              mug_radius=0.07, wrist_radius=0.05)
    obs = ObstacleCost(obs_cfg, fk)
    spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max,
                                    mug_up_local=MUG_UP_LOCAL, dt=dt), fk)

    print("[plan] min-jerk")
    q_minjerk = minjerk_interp(q_start, q_goal, N)
    print("[plan] CHOMP smooth+obs")
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=600, alpha_smooth=1.0,
                      gamma_spill=0.0, beta_obs=50.0, step_size=0.005)
    res_obs = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill, obs_cost_fn=obs.cost)
    q_chomp_obs = res_obs["q_traj"]
    print("[plan] CHOMP smooth+obs+spill")
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=2.0, beta_obs=50.0, step_size=0.003)
    res_ours = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill, obs_cost_fn=obs.cost)
    q_ours = res_ours["q_traj"]

    methods = [
        ("Min-jerk (naive)",       q_minjerk,    (255, 220, 200)),
        ("CHOMP smooth + obs",      q_chomp_obs,  (220, 220, 240)),
        ("CHOMP + obs + spill",     q_ours,       (220, 240, 220)),
    ]
    per_method = {}
    for label, q_traj, bg in methods:
        print(f"render: {label}")
        frames = render_method(g_info, T_em, fk, xhand_qpos, q_traj, label, bg,
                                box_center, box_halfext, theta_max_deg=theta_max)
        per_method[label] = frames
        out = VU_ROOT / f"figs/video_obstacle_{label.split()[0].lower()}.gif"
        save_gif(frames, out, duration_ms=90)

    # composite
    N_frames = min(len(f) for f in per_method.values())
    composite = []
    keys = [m[0] for m in methods]
    for i in range(N_frames):
        row = np.concatenate([per_method[k][i] for k in keys], axis=1)
        composite.append(row)
    save_gif(composite, VU_ROOT / "figs/video_obstacle_3way.gif", duration_ms=90)


if __name__ == "__main__":
    main()
