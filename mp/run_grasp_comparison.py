"""Grasp robustness ablation — connects to the thesis affordance grasp.

For N different grasps from the thesis dataset (mix of 2finger / 3finger /
whole-hand), measure how spill-resistant the resulting transport trajectory
can be made under a fixed time budget.

Pipeline per grasp:
  1) Load grasp_info → build sapien scene at that qpos.
  2) Extract T_ee_to_mug for this grasp.
  3) Solve IK for upright transport start + goal (note: some grasps may make
     these poses unreachable — log as IK failure).
  4) Run CHOMP+spill at T=0.5 s (challenging) and T=1.0 s.
  5) Record min achievable spill ratio + max tilt.

Output:
  results/grasp_comparison.json
  figs/fig_grasp_comparison.png  — per-grasp bars, colored by taxonomy
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import sapien
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import sys
ROOT = Path(__file__).resolve().parents[1]
THESIS = Path("/home/dongjae/UltraDexGrasp/UltraDexGrasp_jihoon")
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize

# Grasps to evaluate (mix of taxonomies, hand-picked)
GRASP_CANDIDATES = [
    ("2finger", 0),  ("2finger", 113), ("2finger", 200),
    ("3finger", 0),  ("3finger", 100), ("3finger", 200),
    ("whole",   50), ("whole",   100), ("whole",   150), ("whole",   200), ("whole", 280),
]

TAXONOMY_COLOR = {"2finger": "#1f77b4", "3finger": "#ff7f0e", "whole": "#2ca02c"}


def upright_pose(p_world):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p_world, float)
    return T


def pose_to_mat(pose: sapien.Pose) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
    M[:3, 3] = pose.p
    return M


def extract_ee_to_mug(grasp_name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load grasp data + sapien scene → returns (T_ee_to_mug, xhand_qpos, grasp_info_dict)."""
    folder = THESIS / f"data_grasp_pose/mug/mug_0/{grasp_name}"
    grasp_info_path = folder / "grasp_info.txt"
    g = parse_grasp_info(grasp_info_path)

    scene = sapien.Scene(); scene.set_timestep(1/240); scene.add_ground(0)
    loader = scene.create_urdf_loader(); loader.fix_root_link = True
    robot = loader.load(str(URDF))
    robot.set_root_pose(sapien.Pose([0, 0, 0]))

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(str(ROOT / "data/mug_mesh.obj"), scale=[g['scale']] * 3)
    mug = builder.build_kinematic(name="mug")
    mug.set_pose(sapien.Pose(g['obj_pos'].tolist(), g['obj_quat_wxyz'].tolist()))

    JOINT_ORDER = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
        "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
        "left_hand_mid_joint1", "left_hand_mid_joint2",
        "left_hand_ring_joint1", "left_hand_ring_joint2",
        "left_hand_pinky_joint1", "left_hand_pinky_joint2",
    ]
    sapien_order = [j.name for j in robot.get_active_joints()]
    n2i = {n: i for i, n in enumerate(JOINT_ORDER)}
    qpos_sapien = np.array([g['qpos'][n2i[n]] for n in sapien_order])
    robot.set_qpos(qpos_sapien)

    ee_link = next(l for l in robot.get_links() if l.name == "base")
    T_base_to_ee = pose_to_mat(ee_link.pose)
    T_base_to_mug = pose_to_mat(mug.pose)
    T_em = np.linalg.inv(T_base_to_ee) @ T_base_to_mug
    return T_em, g['qpos'][6:], g


def evaluate_one_grasp(fk, T_em, q_seed, p_start, p_goal,
                        T_budget=1.0, N=50, theta_max=18.0,
                        n_chomp_iters=800):
    """Run IK + CHOMP+spill for one grasp, return diagnostics dict.
    Returns None for {start,goal,chomp}_ok = False on failure."""
    out = dict(ik_start_err=None, ik_goal_err=None, ik_ok=False,
                spill_ratio=None, max_tilt_deg=None)
    # IK start
    T_start = upright_pose(p_start)
    T_ee_target = T_start @ np.linalg.inv(T_em)
    q0, info0 = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                          torch.tensor(T_ee_target, dtype=torch.float64),
                          n_iters=2500, lr=0.03)
    out["ik_start_err"] = info0["pos_err"]
    if info0["pos_err"] > 0.01:
        return out
    # IK goal
    T_goal = upright_pose(p_goal)
    T_ee_target = T_goal @ np.linalg.inv(T_em)
    q1, info1 = ik_solve(fk, q0,
                          torch.tensor(T_ee_target, dtype=torch.float64),
                          n_iters=2500, lr=0.03)
    out["ik_goal_err"] = info1["pos_err"]
    if info1["pos_err"] > 0.01:
        return out
    out["ik_ok"] = True

    # CHOMP+spill
    dt = T_budget / (N - 1)
    spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=n_chomp_iters,
                      alpha_smooth=1.0, gamma_spill=2.0, step_size=0.003)
    res = chomp_optimize(fk, q0.numpy(), q1.numpy(), cfg, spill=spill)
    d = res["spill_diag"]
    out["spill_ratio"] = float(d["spill_ratio"])
    out["max_tilt_deg"] = float(d["tilt_deg"].max())
    return out


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    p_start = (0.5, -0.3, 0.25)
    p_goal = (0.5, 0.4, 0.40)

    print(f"Evaluating {len(GRASP_CANDIDATES)} grasps at T = 0.5s and 1.0s\n")
    rows = []
    for tax, idx in GRASP_CANDIDATES:
        grasp_name = f"mug_0_left_{tax}_{idx}"
        print(f"--- {grasp_name}")
        try:
            T_em, xhand_qpos, ginfo = extract_ee_to_mug(grasp_name)
        except Exception as e:
            print(f"  FAIL load: {e}"); continue
        q_seed = ginfo['qpos'][:6]
        rec = dict(grasp_name=grasp_name, taxonomy=tax, idx=idx)
        for T_b in [0.5, 1.0]:
            r = evaluate_one_grasp(fk, T_em, q_seed, p_start, p_goal, T_budget=T_b)
            for k, v in r.items():
                rec[f"T{int(T_b*1000)}ms_{k}"] = v
            print(f"  T={T_b}s  ik_ok={r['ik_ok']}  "
                  f"spill={r['spill_ratio']*100 if r['spill_ratio'] is not None else 'NA':<6}  "
                  f"max_tilt={r['max_tilt_deg']:.1f}°" if r['max_tilt_deg'] is not None else "  max_tilt=NA")
        rows.append(rec)

    # Save JSON
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    with open(out_dir / "grasp_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out_dir/'grasp_comparison.json'}")

    # Plot: spill ratio per grasp, 2 bars (T=0.5 / T=1.0)
    labels = [f"{r['taxonomy']}\n#{r['idx']}" for r in rows]
    colors = [TAXONOMY_COLOR[r['taxonomy']] for r in rows]
    s05 = [r['T500ms_spill_ratio'] * 100 if r.get('T500ms_spill_ratio') is not None else np.nan for r in rows]
    s10 = [r['T1000ms_spill_ratio'] * 100 if r.get('T1000ms_spill_ratio') is not None else np.nan for r in rows]
    ok = [r['T500ms_ik_ok'] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    x = np.arange(len(labels))
    for ax, vals, title in [(axes[0], s05, "T = 0.5 s (challenging)"),
                              (axes[1], s10, "T = 1.0 s")]:
        bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
        for i, (v, o) in enumerate(zip(vals, ok)):
            if np.isnan(v):
                ax.text(i, 2, "IK\nfail", ha="center", fontsize=9, color="red")
            else:
                ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Spill ratio (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    # Legend for taxonomy
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor='black', label=t)
                for t, c in TAXONOMY_COLOR.items()]
    axes[0].legend(handles=handles, fontsize=9, loc="upper right")
    fig.suptitle("Spill robustness across grasps (CHOMP+spill, same transport task)",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_grasp_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {ROOT/'figs/fig_grasp_comparison.png'}")


if __name__ == "__main__":
    main()
