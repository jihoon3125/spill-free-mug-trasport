"""Hyperparameter sensitivity: γ_spill sweep.

For γ_spill ∈ {0, 0.1, 0.5, 1, 2, 5, 10, 50} (with α_smooth = 1.0 fixed)
on the same T=0.5s transport task (challenging), measure:
  - spill ratio (% of waypoints exceeding θ_max)
  - jerk integral (joint-space smoothness)
  - max tilt
  - mug center trajectory length

Output:
  results/gamma_sweep.json
  figs/fig_gamma_sweep.png  — Pareto curve: smoothness vs spill
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize


def upright_pose(p_world):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p_world, float)
    return T


def jerk_integral(q_traj, dt):
    a = (q_traj[2:] - 2 * q_traj[1:-1] + q_traj[:-2]) / (dt ** 2)
    j = (a[1:] - a[:-1]) / dt
    return float(np.sum(np.linalg.norm(j, axis=1) ** 2))


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)
    # IK once
    def solve(p_target, q_init):
        T_t = upright_pose(p_target); T_ee = T_t @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_init, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy(), info
    q_start, _ = solve(p_start, g['qpos'][:6])
    q_goal, _ = solve(p_goal, q_start)

    N = 50; T_budget = 0.5; dt = T_budget / (N - 1); theta_max = 18.0
    spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)

    gammas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    rows = []
    for gamma in gammas:
        cfg = CHOMPConfig(N=N, dt=dt, n_iters=1000, alpha_smooth=1.0,
                          gamma_spill=gamma, step_size=0.003)
        res = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
        d = res["spill_diag"]; q_t = res["q_traj"]
        rec = dict(gamma=gamma,
                    spill_ratio=float(d['spill_ratio']),
                    max_tilt_deg=float(d['tilt_deg'].max()),
                    mean_tilt_deg=float(d['tilt_deg'].mean()),
                    jerk_int=jerk_integral(q_t, dt))
        rows.append(rec)
        print(f"γ={gamma:>6.2f}  spill={rec['spill_ratio']*100:5.1f}%  "
              f"max_tilt={rec['max_tilt_deg']:5.1f}°  "
              f"jerk_int={rec['jerk_int']:.3e}")

    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    with open(out_dir / "gamma_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ----- Pareto figure -----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    gs = [r['gamma'] for r in rows]
    sp = [r['spill_ratio'] * 100 for r in rows]
    jk = [r['jerk_int'] for r in rows]

    ax = axes[0]
    ax.semilogx([max(g, 0.05) for g in gs], sp, "o-", color="#1b9e77", lw=1.8, markersize=7)
    for x, y, g in zip([max(g, 0.05) for g in gs], sp, gs):
        ax.annotate(f"γ={g}", (x, y), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel(r"$\gamma_{\rm spill}$ (log scale)")
    ax.set_ylabel("Spill ratio (%)")
    ax.set_title("Spill ratio vs γ_spill (T=0.5s, fixed α_smooth=1.0)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(jk, sp, c=range(len(rows)), cmap="viridis", s=80, edgecolor="k")
    for j, s, g in zip(jk, sp, gs):
        ax.annotate(f"γ={g}", (j, s), fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Jerk integral (joint-space)")
    ax.set_ylabel("Spill ratio (%)")
    ax.set_title("Pareto frontier: smoothness vs spill")
    ax.grid(alpha=0.3)

    fig.suptitle(r"Hyperparameter sensitivity: $\gamma_{\rm spill}$ sweep",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    out = ROOT / "figs/fig_gamma_sweep.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
