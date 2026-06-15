"""Trajectory comparison figure: the 5 methods on ONE transport task.
Left: mug-center 3D path. Right: effective tilt over time (spill threshold marked).

Output: figs/fig_trajectories.png
"""
from __future__ import annotations
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
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from mp.stomp import STOMPConfig, stomp_optimize

PS = (0.45, -0.30, 0.35); PG = (0.45, 0.30, 0.35)
T = 1.0; N = 50; THETA = 18.0; GAMMA = 1.0
COL = {"min-jerk": "#999999", "chomp-smooth": "#1f77b4", "stomp-spill": "#9467bd",
       "vanilla-grad": "#ff7f0e", "ours": "#2ca02c"}


def upright(p):
    M = np.eye(4); M[:3, :3] = R_UPRIGHT; M[:3, 3] = np.asarray(p, float); return M


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    dt = T / (N - 1)

    def ik(p, seed):
        T_ee = torch.tensor(upright(p) @ np.linalg.inv(T_em), dtype=torch.float64)
        q, info = ik_solve(fk, torch.tensor(seed, dtype=torch.float64), T_ee, n_iters=2500, lr=0.03)
        return q.numpy(), info
    qs, _ = ik(PS, g['qpos'][:6]); qg, _ = ik(PG, qs)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA, dt=dt), fk)

    trajs = {}
    trajs["min-jerk"] = minjerk_interp(qs, qg, N)
    trajs["chomp-smooth"] = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                            gamma_spill=0.0, step_size=0.003), spill=sp)["q_traj"]
    torch.manual_seed(0)
    trajs["stomp-spill"] = stomp_optimize(fk, qs, qg, STOMPConfig(N=N, dt=dt, K=50, n_iters=300,
                            sigma_init=0.04, sigma_decay=0.995, lam=0.2, gamma_spill=GAMMA), spill=sp)["q_traj"]
    trajs["vanilla-grad"] = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                            gamma_spill=GAMMA, step_size=0.003, use_covariant=False), spill=sp)["q_traj"]
    trajs["ours"] = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                            gamma_spill=GAMMA, step_size=0.003), spill=sp)["q_traj"]

    diag = {m: sp.diagnostics(torch.tensor(q, dtype=torch.float64)) for m, q in trajs.items()}
    t = np.arange(N) * dt

    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for m, q in trajs.items():
        p = diag[m]["p_mug"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=COL[m], lw=2, label=m)
    p0 = diag["ours"]["p_mug"]
    ax.scatter(*p0[0], c="black", marker="o", s=40); ax.text(*p0[0], "  start", fontsize=8)
    ax.scatter(*p0[-1], c="black", marker="*", s=90); ax.text(*p0[-1], "  goal", fontsize=8)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title("Mug-center path"); ax.view_init(elev=20, azim=-60)
    ax.legend(fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2)
    for m in trajs:
        ax2.plot(t, diag[m]["tilt_deg"], color=COL[m], lw=2, label=m)
    ax2.axhline(THETA, color="red", ls="--", lw=1, label=f"spill threshold {THETA:.0f}°")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("effective tilt (°)")
    ax2.set_title(f"Effective tilt over time (T={T}s)"); ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle("Five methods on one transport task", fontsize=13)
    fig.tight_layout()
    out = ROOT / "figs/fig_trajectories.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"saved {out}")
    for m in trajs:
        print(f"  {m:14s} max tilt {diag[m]['tilt_deg'].max():.0f}°  spill {diag[m]['spill_ratio']*100:.0f}%")


if __name__ == "__main__":
    main()
