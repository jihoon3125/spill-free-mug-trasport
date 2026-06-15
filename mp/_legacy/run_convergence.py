"""Convergence diagnostics — CHOMP cost evolution over iterations.

For our 3 main scenarios (T=1s transport, pouring, obstacle), record per-iter:
  total cost J, J_smooth, J_spill, J_obs
and plot.

Validates that the optimizer (Adam + M^-1) actually converges and shows the
trade-off between cost components.

Output:
  figs/fig_convergence.png    multi-panel cost curves
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
from mp.obstacle import ObstacleCost, ObstacleConfig
from mp.chomp import CHOMPConfig, chomp_optimize


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    def ik(p, q_seed):
        T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
        q, _ = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                         torch.tensor(T_ee, dtype=torch.float64),
                         n_iters=2500, lr=0.03)
        return q.numpy()
    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)
    q_start = ik(p_start, g['qpos'][:6])
    q_goal = ik(p_goal, q_start)

    scenarios: dict[str, dict] = {}

    # --- A) Transport only (T=1s) ---
    print("[scenario] transport (T=1s)")
    N = 50; dt = 0.02
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp)
    scenarios["transport_T1s"] = res["loss_hist"]
    print(f"  final J = {res['loss_hist'][-1]['J']:.3e}")

    # --- B) Transport at challenging T=0.5s ---
    print("[scenario] transport (T=0.5s)")
    N = 50; dt = 0.5 / (N - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp)
    scenarios["transport_T05s"] = res["loss_hist"]
    print(f"  final J = {res['loss_hist'][-1]['J']:.3e}")

    # --- C) Obstacle scenario ---
    print("[scenario] obstacle")
    p_start_o = (0.5, -0.3, 0.30); p_goal_o = (0.5, 0.4, 0.30)
    q_start_o = ik(p_start_o, g['qpos'][:6])
    q_goal_o = ik(p_goal_o, q_start_o)
    N = 60; dt = 0.025
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
    obs_cfg = ObstacleConfig(box_center=(0.55, 0.05, 0.28),
                              box_halfext=(0.10, 0.10, 0.10),
                              margin=0.04, T_ee_to_mug=T_em,
                              mug_radius=0.07, wrist_radius=0.05)
    obs = ObstacleCost(obs_cfg, fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                      gamma_spill=2.0, beta_obs=50.0, step_size=0.003)
    res = chomp_optimize(fk, q_start_o, q_goal_o, cfg, spill=sp, obs_cost_fn=obs.cost)
    scenarios["obstacle"] = res["loss_hist"]
    print(f"  final J = {res['loss_hist'][-1]['J']:.3e}")

    # ----- Figure -----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    titles = {"transport_T1s": "Transport @ T=1.0s",
              "transport_T05s": "Transport @ T=0.5s (challenging)",
              "obstacle": "Obstacle + spill"}
    for ax, (k, hist) in zip(axes, scenarios.items()):
        iters = np.arange(len(hist))
        J = np.array([h['J'] for h in hist])
        J_s = np.array([h['J_smooth'] for h in hist])
        J_sp = np.array([h['J_spill'] for h in hist])
        J_o = np.array([h['J_obs'] for h in hist])
        ax.semilogy(iters, np.maximum(J, 1e-10), color="k", lw=1.6, label="J total")
        ax.semilogy(iters, np.maximum(J_s, 1e-10), color="#7570b3", lw=1.0, alpha=0.8, label="J_smooth")
        ax.semilogy(iters, np.maximum(J_sp, 1e-10), color="#1b9e77", lw=1.0, alpha=0.8, label="J_spill")
        if J_o.max() > 0:
            ax.semilogy(iters, np.maximum(J_o, 1e-10), color="#d95f02", lw=1.0, alpha=0.8, label="J_obs")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Cost (log)")
        ax.set_title(titles[k]); ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle("CHOMP convergence — cost components vs iteration", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs/fig_convergence.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
