"""Grasp comparison video — same transport task with 3 different grasps,
side-by-side. Shows that the affordance choice affects spill robustness.

Grasps:
  A. whole_200 — body wrap (default; best in our analysis)
  B. whole_50  — rim+handle (worst at T=0.5s in our prior result)
  C. 3finger_200 — three-finger side grip

Same task: 70cm lateral transport at T=0.6s (challenging speed).
Same planner: CHOMP+spill.

Output: figs/video_grasp_compare.gif (3-col composite) + per-grasp gifs.
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
from mp.chomp import CHOMPConfig, chomp_optimize
from mp.run_grasp_comparison import extract_ee_to_mug
from mp._video_utils import (
    MUG_UP_LOCAL,
    flip_pose_for_render, mat_to_sapien_pose, sample_water_particles,
    compute_spill_state, render_camera, take_rgb, annotate_frame, save_gif,
    make_floor_cost, ROOT as VU_ROOT,
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


def build_scene(g, n_particles=25):
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


def plan_and_render(grasp_name, label, color_bg, T_budget=0.6, theta_max=18.0):
    g_info = parse_grasp_info(VU_ROOT / "data/grasp_info.txt")  # for global stats; per-grasp loaded below
    T_em, xhand_qpos, ginfo = extract_ee_to_mug(grasp_name)
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    p_start = (0.5, -0.3, 0.35); p_goal = (0.5, 0.4, 0.45)
    def ik(p, q_seed):
        T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy(), info
    q_s, info_s = ik(p_start, ginfo['qpos'][:6])
    if info_s['pos_err'] > 0.03:
        print(f"  {grasp_name}: IK FAIL start (err={info_s['pos_err']:.3f})"); return None
    q_g, info_g = ik(p_goal, q_s)
    if info_g['pos_err'] > 0.03:
        print(f"  {grasp_name}: IK FAIL goal (err={info_g['pos_err']:.3f})"); return None

    N = 50; dt = T_budget / (N - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max,
                                 mug_up_local=MUG_UP_LOCAL, dt=dt), fk)
    floor_cost = make_floor_cost(fk, T_em, z_floor=0.05, margin=0.05)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1000, alpha_smooth=1.0,
                      gamma_spill=5.0, beta_obs=300.0, step_size=0.003)
    res = chomp_optimize(fk, q_s, q_g, cfg, spill=sp, obs_cost_fn=floor_cost)
    q_traj = res["q_traj"]; d = res["spill_diag"]
    print(f"  {grasp_name}: spill_ratio={d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")

    # Render
    scene, robot, mug, particles, pts_local, cam, sapien_order = build_scene(ginfo)
    mug_centers = np.zeros((N, 3)); mug_Rw = np.zeros((N, 3, 3))
    for i in range(N):
        with torch.no_grad():
            T_ee_i = fk(torch.tensor(q_traj[i], dtype=torch.float64)).numpy()
        T_mug_i = T_ee_i @ T_em
        mug_centers[i] = T_mug_i[:3, 3]; mug_Rw[i] = T_mug_i[:3, :3]

    G_W = np.array([0.0, 0.0, -9.81]); M = len(pts_local)
    released = np.zeros(M, dtype=bool)
    p_w_state = np.zeros((M, 3)); v_w_state = np.zeros((M, 3))
    frames = []
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
            T_mug, a_mug, theta_max)
        R_render = T_render[:3, :3]
        for k in range(M):
            if released[k]:
                p_w_state[k] = p_w_state[k] + v_w_state[k]*dt + 0.5*G_W*(dt**2)
                v_w_state[k] = v_w_state[k] + G_W*dt
            else:
                p_w_state[k] = R_render @ pts_local[k] + T_render[:3, 3]
                proj_render = np.dot(pts_local[k], np.array([0.0, 1.0, 0.0]))
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
            (f"grasp = {grasp_name.replace('mug_0_left_', '')}", "k"),
            (f"tilt = {tilt_deg:.1f}°", "r" if spill_state else "g"),
            (f"spilled {int(released.sum())}/{M}", "r" if spill_state else "g"),
        ]
        img = annotate_frame(img, label, lines, title_bg=color_bg)
        frames.append(img)
    return frames


def main():
    grasps = [
        ("mug_0_left_whole_200",   "A. whole_200 (body wrap)",     (220, 240, 220)),
        ("mug_0_left_whole_150",   "B. whole_150 (body side)",      (255, 230, 210)),
        ("mug_0_left_3finger_200", "C. 3finger_200 (side pinch)",  (220, 230, 250)),
    ]
    per_grasp = {}
    for grasp_name, label, bg in grasps:
        print(f"\n=== {label}  ({grasp_name})")
        frames = plan_and_render(grasp_name, label, bg, T_budget=0.6)
        if frames is None:
            continue
        per_grasp[label] = frames
        # Per-grasp gif
        safe = grasp_name.replace("mug_0_left_", "")
        out = VU_ROOT / f"figs/video_grasp_{safe}.gif"
        save_gif(frames, out, duration_ms=90)

    # 3-col composite
    if len(per_grasp) >= 2:
        N_frames = min(len(f) for f in per_grasp.values())
        composite = []
        for i in range(N_frames):
            row = np.concatenate([per_grasp[k][i] for k in per_grasp], axis=1)
            composite.append(row)
        save_gif(composite, VU_ROOT / "figs/video_grasp_compare.gif", duration_ms=90)


if __name__ == "__main__":
    main()
