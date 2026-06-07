"""Pouring task video — robot transports mug upright then tilts to pour.

Time-varying θ_max(t):
  - first 60% of trajectory: θ_max = 10° (transport, strict)
  - middle 30%: sigmoid ramp up to 70° (pouring window allowed)
  - last 10%: stays high (continuing to pour)

Mug renders RIGHT-SIDE UP (uses _video_utils flip convention).
Water particles sit at the bottom of the cup, released when effective tilt
exceeds the current θ_max(t).

Output:
  figs/video_pouring.gif
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

# Correct upright rotation: maps mesh -y (opening) to world +z.
R_UPRIGHT_NEW = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=float)


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT_NEW; T[:3, 3] = np.asarray(p, float)
    return T


def tilted_pose(p, tilt_deg=70.0, axis="x"):
    """Tilt the cup `tilt_deg` around world `axis` starting from upright."""
    base = R_UPRIGHT_NEW
    rot = R.from_euler(axis, tilt_deg, degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = rot @ base; T[:3, 3] = np.asarray(p, float)
    return T


def sigmoid_schedule(N, t_start_frac=0.55, t_end_frac=1.0,
                       theta_low=10.0, theta_high=80.0, sharpness=12.0):
    tau = np.linspace(0, 1, N)
    s_start = 1.0 / (1.0 + np.exp(-sharpness * (tau - t_start_frac)))
    s_end = 1.0 / (1.0 + np.exp(sharpness * (tau - t_end_frac)))
    s = s_start * s_end / max(s_start.max() * s_end.max(), 1e-6)
    return theta_low + (theta_high - theta_low) * s


def build_scene(g, n_particles=35):
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
        pb.add_sphere_visual(radius=0.010, material=mat)
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


def main():
    g_info = parse_grasp_info(VU_ROOT / "data/grasp_info.txt")
    T_em = np.load(VU_ROOT / "data/ee_to_mug.npy")
    xhand_qpos = np.load(VU_ROOT / "data/xhand_qpos.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    # Task — bring upright cup to a tilted pouring posture
    p_start = (0.5, -0.3, 0.30)
    p_goal = (0.5, 0.3, 0.40)
    tilt_goal_deg = 70.0
    T_start = upright_pose(p_start)
    T_goal = tilted_pose(p_goal, tilt_deg=tilt_goal_deg, axis="x")

    def ik(T, q_seed):
        T_ee = T @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy(), info
    q_start, info_s = ik(T_start, g_info['qpos'][:6])
    q_goal, info_g = ik(T_goal, q_start)
    print(f"IK: start err={info_s['pos_err']:.5f}, goal err={info_g['pos_err']:.5f}")

    # Plan with time-varying θ_max
    N = 80; T_budget = 1.6; dt = T_budget / (N - 1)
    schedule = sigmoid_schedule(N, t_start_frac=0.55, t_end_frac=1.0,
                                  theta_low=10.0, theta_high=80.0)
    # NOTE: SpillCost uses the OLD mug_up_local default ([0,1,0]). For physical
    # correctness with our new convention we pass mug_up_local explicitly.
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=schedule,
                                 mug_up_local=MUG_UP_LOCAL, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp)
    q_traj = res["q_traj"]
    d = res["spill_diag"]
    print(f"Pouring CHOMP+spill (timed): spill_ratio={d['spill_ratio']:.1%}  "
           f"max_tilt={d['tilt_deg'].max():.1f}°")

    # ----- Render -----
    scene, robot, mug, particles, pts_local, cam, sapien_order = build_scene(g_info)

    # Pre-compute mug pose history (with FK, no flip yet)
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
    for i in range(N):
        apply_qpos(robot, sapien_order, q_traj[i], xhand_qpos)
        T_mug = np.eye(4); T_mug[:3, :3] = mug_Rw[i]; T_mug[:3, 3] = mug_centers[i]
        # APPLY VISUAL FLIP so cup renders upright
        T_render = flip_pose_for_render(T_mug)
        mug.set_pose(mat_to_sapien_pose(T_render))

        # Mug velocity + accel (central diff)
        if 0 < i < N - 1:
            v_mug = (mug_centers[i+1] - mug_centers[i-1]) / (2*dt)
            a_mug = (mug_centers[i+1] - 2*mug_centers[i] + mug_centers[i-1]) / (dt**2)
        else:
            v_mug = np.zeros(3); a_mug = np.zeros(3)

        spill_state, tilt_deg, g_eff, up_local_eff = compute_spill_state(
            T_mug, a_mug, schedule[i])

        # Particle update (rigidly attached → released under spill)
        # mug_up_render_world is +z when cup is upright
        R_render = T_render[:3, :3]
        for k in range(M):
            if released[k]:
                p_w_state[k] = p_w_state[k] + v_w_state[k]*dt + 0.5*G_W*(dt**2)
                v_w_state[k] = v_w_state[k] + G_W*dt
            else:
                # particle position: use rendered (flipped) frame for visual consistency
                p_w_state[k] = R_render @ pts_local[k] + T_render[:3, 3]
                # Release rule: if spill and particle is on the high side in the
                # rendered frame. "High side" is along +up_local_render = (0,1,0).
                # Tilted up direction (mesh frame, post-flip): up_local_eff
                proj_render = np.dot(pts_local[k], np.array([0.0, +1.0, 0.0]))
                # combine with tilt direction: if cup tilts (up_local_eff has off-axis
                # component), particles on the off-axis side also release
                proj_eff = np.dot(pts_local[k], up_local_eff)
                if spill_state and (proj_render > 0.005 or proj_eff > 0.01):
                    released[k] = True
                    # eject upward (world +z) + mug velocity
                    v_w_state[k] = v_mug + 0.5 * (R_render @ np.array([0, +1, 0])) + 0.3 * np.array([0, 0, 1])
            if p_w_state[k][2] < 0.012:
                p_w_state[k][2] = 0.012
                if released[k]:
                    v_w_state[k] = np.zeros(3)
            particles[k].set_pose(sapien.Pose(p_w_state[k].tolist()))

        img = take_rgb(scene, cam)
        n_rel = int(released.sum())
        lines = [
            (f"t = {i*dt:.2f}s / {T_budget:.1f}s", "k"),
            (f"θ_max(t) = {schedule[i]:.0f}°", "b"),
            (f"effective tilt = {tilt_deg:.1f}°", "r" if spill_state else "g"),
            (f"spilled {n_rel}/{M}", "r" if spill_state else "g"),
        ]
        phase = "TRANSPORT" if i / (N - 1) < 0.55 else "POURING"
        title = f"Pouring task — {phase}"
        title_bg = (220, 240, 220) if phase == "TRANSPORT" else (255, 230, 200)
        img = annotate_frame(img, title, lines, title_bg=title_bg)
        frames.append(img)

    out = VU_ROOT / "figs/video_pouring.gif"
    save_gif(frames, out, duration_ms=80)
    print("video file:", out)


if __name__ == "__main__":
    main()
