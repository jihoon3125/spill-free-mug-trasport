"""Obstacle scenario: same start/goal as transport ablation, but with a box
obstacle (e.g., a book on the table) directly on the linear path.

Methods (each plans EE trajectory through obstacle):
  Min-jerk (naive)     — straight line through box (collision)
  CHOMP smooth+obs     — avoids box but ignores mug tilt → may spill
  CHOMP smooth+obs+spill (ours) — avoids box AND keeps mug upright

Outputs:
  results/obstacle.npz
  figs/fig_obstacle_3d.png         — mug 3D path of 3 methods + box rendering
  figs/fig_obstacle_metrics.png    — bar (collision + spill ratios)
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.obstacle import ObstacleCost, ObstacleConfig
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp


def upright_pose(p_world):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p_world, float)
    return T


def solve_qpos_for_mug(fk, T_em, T_target, q_init):
    T_ee = T_target @ np.linalg.inv(T_em)
    q_opt, info = ik_solve(fk, torch.tensor(q_init, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2000, lr=0.03)
    return q_opt.numpy(), info


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    # Same path as before but with obstacle in the middle (y=0.05, z=0.30)
    p_start = (0.5, -0.3, 0.30)
    p_goal = (0.5, 0.4, 0.30)
    q_start, _ = solve_qpos_for_mug(fk, T_em, upright_pose(p_start), g['qpos'][:6])
    q_goal, _ = solve_qpos_for_mug(fk, T_em, upright_pose(p_goal), q_start)

    N = 60; dt = 0.025; theta_max = 18.0
    # Larger box so it clearly intersects the joint-space-linear mug path
    obs_cfg = ObstacleConfig(box_center=(0.55, 0.05, 0.28),
                              box_halfext=(0.10, 0.10, 0.10),
                              margin=0.04, T_ee_to_mug=T_em,
                              mug_radius=0.07, wrist_radius=0.05)
    obs = ObstacleCost(obs_cfg, fk)
    spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)

    results: dict[str, dict] = {}

    # ----- 1. Min-jerk -----
    print("[method] min-jerk (naive)")
    q_traj = minjerk_interp(q_start, q_goal, N)
    d_spill = spill.diagnostics(torch.tensor(q_traj, dtype=torch.float64))
    d_obs = obs.diagnostics(torch.tensor(q_traj, dtype=torch.float64))
    results["minjerk"] = dict(q=q_traj, spill=d_spill, obs=d_obs)
    print(f"  spill={d_spill['spill_ratio']:.1%}  obs violations={d_obs['violation_ratio']:.1%}"
          f"  min_clear={d_obs['min_clearance']:.3f}m")

    # ----- 2. CHOMP smooth + obs (no spill) -----
    print("[method] CHOMP smooth+obs")
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=600, alpha_smooth=1.0,
                      gamma_spill=0.0, beta_obs=50.0, step_size=0.005)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill, obs_cost_fn=obs.cost)
    d_spill = res["spill_diag"]; d_obs = obs.diagnostics(torch.tensor(res["q_traj"], dtype=torch.float64))
    results["chomp_obs"] = dict(q=res["q_traj"], spill=d_spill, obs=d_obs)
    print(f"  spill={d_spill['spill_ratio']:.1%}  obs violations={d_obs['violation_ratio']:.1%}"
          f"  min_clear={d_obs['min_clearance']:.3f}m")

    # ----- 3. CHOMP smooth + obs + spill (ours) -----
    print("[method] CHOMP smooth+obs+spill (ours)")
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=2.0, beta_obs=50.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill, obs_cost_fn=obs.cost)
    d_spill = res["spill_diag"]; d_obs = obs.diagnostics(torch.tensor(res["q_traj"], dtype=torch.float64))
    results["chomp_obs_spill"] = dict(q=res["q_traj"], spill=d_spill, obs=d_obs)
    print(f"  spill={d_spill['spill_ratio']:.1%}  obs violations={d_obs['violation_ratio']:.1%}"
          f"  min_clear={d_obs['min_clearance']:.3f}m")

    # Summary
    print("\n=== Summary (obstacle scenario) ===")
    print(f"{'method':<24}  {'spill%':<8} {'collide%':<10} {'min_clear (m)':<12}")
    for name, r in results.items():
        print(f"{name:<24}  {r['spill']['spill_ratio']*100:<8.1f} "
              f"{r['obs']['violation_ratio']*100:<10.1f} "
              f"{r['obs']['min_clearance']:<12.3f}")

    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "obstacle.npz",
             box_center=np.array(obs_cfg.box_center),
             box_halfext=np.array(obs_cfg.box_halfext),
             margin=obs_cfg.margin, theta_max_deg=theta_max,
             **{f"{k}_q": v["q"] for k, v in results.items()},
             **{f"{k}_tilt": v["spill"]["tilt_deg"] for k, v in results.items()},
             **{f"{k}_pmug": v["spill"]["p_mug"] for k, v in results.items()},
             **{f"{k}_minclear": np.array([v["obs"]["min_clearance"]])
                for k, v in results.items()},
             **{f"{k}_violratio": np.array([v["obs"]["violation_ratio"]])
                for k, v in results.items()},
             **{f"{k}_spillratio": np.array([v["spill"]["spill_ratio"]])
                for k, v in results.items()},
             )

    # ----- Figure 1: 3D mug path with box -----
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    methods = ["minjerk", "chomp_obs", "chomp_obs_spill"]
    palette = {"minjerk": ("#d95f02", "Min-jerk (naive)"),
               "chomp_obs": ("#7570b3", "CHOMP smooth+obs"),
               "chomp_obs_spill": ("#1b9e77", "CHOMP+obs+spill (ours)")}
    for m in methods:
        c, lbl = palette[m]
        pm = results[m]["spill"]["p_mug"]
        ax.plot(pm[:, 0], pm[:, 1], pm[:, 2], color=c, label=lbl, lw=2)
    # box visualization
    cx, cy, cz = obs_cfg.box_center; hx, hy, hz = obs_cfg.box_halfext
    verts = np.array([
        [cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz], [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz], [cx - hx, cy + hy, cz + hz],
    ])
    faces = [[verts[i] for i in face] for face in
              [[0,1,2,3], [4,5,6,7], [0,1,5,4], [2,3,7,6], [1,2,6,5], [0,3,7,4]]]
    poly = Poly3DCollection(faces, alpha=0.25, facecolor="grey", edgecolor="k")
    ax.add_collection3d(poly)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Mug center path around obstacle (grey box)")
    ax.view_init(elev=20, azim=-65)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_obstacle_3d.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {ROOT/'figs/fig_obstacle_3d.png'}")

    # ----- Figure 2: metrics bar -----
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [palette[m][1] for m in methods]
    spill_vals = [results[m]["spill"]["spill_ratio"] * 100 for m in methods]
    obs_vals = [results[m]["obs"]["violation_ratio"] * 100 for m in methods]
    x = np.arange(len(methods)); w = 0.35
    ax.bar(x - w/2, spill_vals, w, label="Spill ratio (%)",
            color=[palette[m][0] for m in methods], alpha=0.95)
    ax.bar(x + w/2, obs_vals, w, label="Collision ratio (%)",
            color=[palette[m][0] for m in methods], hatch="//", alpha=0.55, edgecolor="black")
    for i, (s, o) in enumerate(zip(spill_vals, obs_vals)):
        ax.text(i - w/2, s + 1.5, f"{s:.0f}%", ha="center", fontsize=10)
        ax.text(i + w/2, o + 1.5, f"{o:.0f}%", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("% of trajectory")
    ax.set_title("Spill + collision rates (obstacle scenario, lower = better)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_obstacle_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {ROOT/'figs/fig_obstacle_metrics.png'}")


if __name__ == "__main__":
    main()
