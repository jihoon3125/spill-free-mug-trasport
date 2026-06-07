"""Joint velocity limits — check + enforce.

UR5e spec: max |dq/dt| ≈ π rad/s per joint (some joints 2π).
We check existing CHOMP+spill trajectories at various T budgets, and
add a soft velocity-limit penalty if needed.

Output:
  results/vellimits.json
  figs/fig_vellimits.png
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


UR5E_VLIM = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])     # rad/s


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def check_vel(q_traj, dt):
    dq = (q_traj[1:] - q_traj[:-1]) / dt
    max_abs = np.abs(dq).max(axis=0)        # (D,) max |dq/dt| per joint
    over = max_abs > UR5E_VLIM[:len(max_abs)]
    return max_abs, over


def velocity_cost_fn_factory(dt: float, v_lim: np.ndarray):
    """Returns a callable cost that penalizes |dq/dt| > v_lim per joint."""
    v_lim_t = torch.tensor(v_lim, dtype=torch.float64)
    def cost(q_traj: torch.Tensor) -> torch.Tensor:
        dq = (q_traj[1:] - q_traj[:-1]) / dt
        excess = torch.clamp(dq.abs() - v_lim_t, min=0.0)
        return (excess ** 2).sum()
    return cost


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)
    def ik(p, q_seed):
        T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy()
    q_start = ik(p_start, g['qpos'][:6])
    q_goal = ik(p_goal, q_start)

    N = 50
    Ts = [0.3, 0.5, 0.8, 1.0, 1.5]
    rows = []
    for T_budget in Ts:
        dt = T_budget / (N - 1)
        spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)

        # Without velocity penalty
        cfg = CHOMPConfig(N=N, dt=dt, n_iters=1000, alpha_smooth=1.0,
                          gamma_spill=5.0, step_size=0.003)
        res = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
        q_t = res["q_traj"]
        max_v, over = check_vel(q_t, dt)
        d = res["spill_diag"]
        rec_a = dict(T=T_budget, mode="no_vlim",
                      spill_ratio=float(d['spill_ratio']),
                      max_v=float(max_v.max()),
                      n_joints_over=int(over.sum()),
                      per_joint_max=max_v.tolist())

        # With velocity penalty
        vel_cost = velocity_cost_fn_factory(dt, UR5E_VLIM)
        cfg2 = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                            gamma_spill=5.0, beta_obs=200.0, step_size=0.003)
        res2 = chomp_optimize(fk, q_start, q_goal, cfg2, spill=spill,
                              obs_cost_fn=vel_cost)
        q_t2 = res2["q_traj"]
        max_v2, over2 = check_vel(q_t2, dt)
        d2 = res2["spill_diag"]
        rec_b = dict(T=T_budget, mode="with_vlim",
                      spill_ratio=float(d2['spill_ratio']),
                      max_v=float(max_v2.max()),
                      n_joints_over=int(over2.sum()),
                      per_joint_max=max_v2.tolist())
        rows.append(rec_a); rows.append(rec_b)

        print(f"T={T_budget}s  (no_vlim)  max|dq/dt|={max_v.max():.2f}  "
               f"limit_ok={not over.any()}  spill={d['spill_ratio']*100:.1f}%")
        print(f"T={T_budget}s  (w_vlim)   max|dq/dt|={max_v2.max():.2f}  "
               f"limit_ok={not over2.any()}  spill={d2['spill_ratio']*100:.1f}%")

    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    with open(out_dir / "vellimits.json", "w") as f:
        json.dump(rows, f, indent=2)

    # ----- Figure -----
    Ts_arr = np.array(Ts)
    novel = np.array([r["max_v"] for r in rows if r["mode"] == "no_vlim"])
    wvel = np.array([r["max_v"] for r in rows if r["mode"] == "with_vlim"])
    sp_no = np.array([r["spill_ratio"] for r in rows if r["mode"] == "no_vlim"]) * 100
    sp_w = np.array([r["spill_ratio"] for r in rows if r["mode"] == "with_vlim"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(Ts_arr, novel, "o-", color="#d95f02", label="CHOMP+spill (no vlim)", lw=1.8)
    ax.plot(Ts_arr, wvel, "s-", color="#1b9e77", label="CHOMP+spill+vlim", lw=1.8)
    ax.axhline(np.pi, ls="--", color="r", lw=1, label=f"UR5e v_max = π ≈ {np.pi:.2f} rad/s")
    ax.set_xlabel("Trajectory time T (s)")
    ax.set_ylabel("Max joint velocity (rad/s)")
    ax.set_title("Joint velocity vs T (UR5e limit)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Ts_arr, sp_no, "o-", color="#d95f02", label="no vlim", lw=1.8)
    ax.plot(Ts_arr, sp_w, "s-", color="#1b9e77", label="with vlim", lw=1.8)
    ax.set_xlabel("Trajectory time T (s)")
    ax.set_ylabel("Spill ratio (%)")
    ax.set_title("Spill cost stability when adding vlim")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("Joint velocity limit constraint impact", y=1.02)
    fig.tight_layout()
    out = ROOT / "figs/fig_vellimits.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
