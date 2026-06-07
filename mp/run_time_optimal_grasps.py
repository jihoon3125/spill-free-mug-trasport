"""Time-optimal transport per grasp.

For each grasp (from the prior grasp comparison set), find the MINIMUM trajectory
time T_min at which CHOMP+spill achieves 0% spill on the standard transport task.

Binary search over T ∈ [0.3, 2.5] seconds.
A grasp is 'better' if T_min is smaller.

Output:
  results/time_optimal_grasps.json
  figs/fig_time_optimal_grasps.png
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
from mp.run_grasp_comparison import extract_ee_to_mug, GRASP_CANDIDATES, TAXONOMY_COLOR


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def evaluate_T(fk, T_em, q_start, q_goal, T_budget, theta_max=18.0):
    N = 50; dt = T_budget / (N - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp)
    return float(res["spill_diag"]["spill_ratio"])


def find_T_min(fk, T_em, q_start, q_goal, T_lo=0.3, T_hi=2.5, tol=0.05,
                theta_max=18.0, max_iter=8):
    """Binary search: find smallest T where spill ratio == 0.
    Returns T_min (or T_hi+ if even T_hi has spill)."""
    # Check feasibility at T_hi
    sp_hi = evaluate_T(fk, T_em, q_start, q_goal, T_hi, theta_max)
    if sp_hi > 0.02:
        return None       # even at slow T, can't reach 0
    sp_lo = evaluate_T(fk, T_em, q_start, q_goal, T_lo, theta_max)
    if sp_lo <= 0.02:
        return T_lo
    for _ in range(max_iter):
        if T_hi - T_lo < tol:
            break
        T_mid = 0.5 * (T_lo + T_hi)
        sp = evaluate_T(fk, T_em, q_start, q_goal, T_mid, theta_max)
        if sp <= 0.02:
            T_hi = T_mid
        else:
            T_lo = T_mid
    return T_hi


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)

    rows = []
    for tax, idx in GRASP_CANDIDATES:
        grasp_name = f"mug_0_left_{tax}_{idx}"
        print(f"--- {grasp_name}")
        try:
            T_em, _, ginfo = extract_ee_to_mug(grasp_name)
        except Exception as e:
            print(f"  load fail: {e}"); continue
        # IK
        def ik(p, q_seed):
            T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
            q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                                torch.tensor(T_ee, dtype=torch.float64),
                                n_iters=2500, lr=0.03)
            return q.numpy(), info
        q_s, info_s = ik(p_start, ginfo['qpos'][:6])
        if info_s['pos_err'] > 0.01:
            print(f"  IK fail start (err={info_s['pos_err']:.3f})")
            rows.append(dict(grasp_name=grasp_name, taxonomy=tax, idx=idx, T_min=None, ik_fail=True))
            continue
        q_g, info_g = ik(p_goal, q_s)
        if info_g['pos_err'] > 0.01:
            print(f"  IK fail goal (err={info_g['pos_err']:.3f})")
            rows.append(dict(grasp_name=grasp_name, taxonomy=tax, idx=idx, T_min=None, ik_fail=True))
            continue
        T_min = find_T_min(fk, T_em, q_s, q_g, T_lo=0.3, T_hi=2.5)
        rows.append(dict(grasp_name=grasp_name, taxonomy=tax, idx=idx,
                          T_min=T_min, ik_fail=False))
        print(f"  T_min = {T_min:.3f}s" if T_min else "  even T=2.5s fails")

    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    with open(out_dir / "time_optimal_grasps.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Plot
    labels = [f"{r['taxonomy']}\n#{r['idx']}" for r in rows]
    colors = [TAXONOMY_COLOR[r['taxonomy']] for r in rows]
    vals = [r['T_min'] if (r['T_min'] is not None) else np.nan for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="black")
    for i, (v, r) in enumerate(zip(vals, rows)):
        if np.isnan(v):
            txt = "IK\nfail" if r["ik_fail"] else "infeasible"
            ax.text(i, 0.05, txt, ha="center", fontsize=8, color="red")
        else:
            ax.text(i, v + 0.03, f"{v:.2f}s", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$T_{\min}$ (s) — fastest spill-free transport")
    ax.set_title("Time-optimal transport per grasp (lower is better)")
    ax.set_ylim(0, max(2.6, max([v for v in vals if not np.isnan(v)], default=2.5) * 1.1))
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor='black', label=t)
                for t, c in TAXONOMY_COLOR.items()]
    ax.legend(handles=handles, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = ROOT / "figs/fig_time_optimal_grasps.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
