"""Feasible grasp-conditioned analysis (honest replacement for T_min).

At gamma_spill=1.0 (no degenerate high-accel "cheat"), for each reachable grasp
sweep the time budget T and record spill and joint-limit feasibility. Define
  T* = fastest budget that is BOTH spill-free (<=2%) AND within joint limits.
Check monotonicity of spill in T (so T* is well-defined), and whether T* varies
across grasps and correlates with grasp offset.

Output: results/tstar.json, figs/fig_tstar.png, figs/fig_tstar_offset.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.scene import URDF
from mp.kinematics import URDFForwardKinematics
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize
from mp.run_grasp_comparison import extract_ee_to_mug
from mp.run_method_comparison import ik_multiseed

PS = (0.5, -0.3, 0.25); PG = (0.5, 0.4, 0.40)
N = 50; GAMMA = 1.0
TS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
V_MAX = 3.3; A_MAX = 20.0
COL = {"2finger": "#1f77b4", "3finger": "#ff7f0e", "whole": "#2ca02c"}


def evaluate(fk, T_em, qs, qg, T):
    dt = T / (N - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
    q = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                       gamma_spill=GAMMA, step_size=0.003), spill=sp)["q_traj"]
    spill = sp.diagnostics(torch.tensor(q, dtype=torch.float64))["spill_ratio"]
    v = np.abs(np.diff(q, axis=0) / dt).max()
    a = np.abs(np.diff(np.diff(q, axis=0), axis=0) / dt ** 2).max()
    return float(spill), float(v), float(a)


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    tm = {r["grasp_name"]: r for r in json.load(open(ROOT / "results/transport_map.json"))}
    names = [g["grasp_name"] for g in json.load(open(ROOT / "results/method_comparison.json"))["per_grasp"]]
    rows = []
    for gi, name in enumerate(names):
        T_em, _, g = extract_ee_to_mug(name)
        qs, i_s = ik_multiseed(fk, T_em, PS, g['qpos'][:6])
        qg, i_g = ik_multiseed(fk, T_em, PG, qs)
        if i_s['pos_err'] > 0.012 or i_g['pos_err'] > 0.012:
            continue
        spills, feas = [], []
        for T in TS:
            s, v, a = evaluate(fk, T_em, qs, qg, T)
            spills.append(s); feas.append(v < V_MAX and a < A_MAX)
        # T* = smallest T with spill<=0.02 and feasible
        tstar = None
        for T, s, f in zip(TS, spills, feas):
            if s <= 0.02 and f:
                tstar = T; break
        rows.append(dict(grasp_name=name, taxonomy=name.split("_left_")[1].split("_")[0],
                         offset_r=tm[name]["offset_r"] * 100, spills=spills, feas=feas, tstar=tstar))
        print(f"[{gi+1}/{len(names)}] {name} T*={tstar} spills={[round(s,2) for s in spills]}", flush=True)

    # monotonicity: fraction of grasps where spill is (weakly) non-increasing in T (tol 0.05)
    mono = 0
    for r in rows:
        s = r["spills"]
        if all(s[i+1] <= s[i] + 0.05 for i in range(len(s) - 1)):
            mono += 1
    json.dump(dict(TS=TS, gamma=GAMMA, rows=rows, monotonic_frac=mono / max(1, len(rows))),
              open(ROOT / "results/tstar.json", "w"), indent=2)

    tstars = [r["tstar"] for r in rows if r["tstar"] is not None]
    print(f"\nn={len(rows)}  monotonic spill(T): {mono}/{len(rows)} "
          f"({100*mono/len(rows):.0f}%)")
    print(f"T* defined for {len(tstars)}/{len(rows)} grasps; "
          f"range {min(tstars):.1f}-{max(tstars):.1f}s, median {np.median(tstars):.1f}s")
    off = [r["offset_r"] for r in rows if r["tstar"] is not None]
    if len(tstars) > 2:
        print(f"corr(offset, T*) = {np.corrcoef(off, tstars)[0,1]:.2f}")

    # Fig 1: T* per grasp sorted
    rr = sorted([r for r in rows if r["tstar"] is not None], key=lambda r: r["tstar"])
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(range(len(rr)), [r["tstar"] for r in rr],
           color=[COL[r["taxonomy"]] for r in rr], edgecolor="black", lw=0.3)
    ax.set_xlabel("grasp (sorted)"); ax.set_ylabel("T* (s)  fastest feasible spill-free")
    ax.set_title("Fastest feasible spill-free transport time varies by grasp")
    ax.legend(handles=[Patch(facecolor=c, edgecolor='k', label=t) for t, c in COL.items()], fontsize=8)
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_tstar.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # Fig 2: T* vs offset
    fig, ax = plt.subplots(figsize=(5, 4))
    for t in COL:
        m = [r for r in rows if r["taxonomy"] == t and r["tstar"] is not None]
        ax.scatter([r["offset_r"] for r in m], [r["tstar"] for r in m], c=COL[t],
                   s=36, edgecolor="black", label=t)
    rcoef = np.corrcoef(off, tstars)[0, 1] if len(tstars) > 2 else float("nan")
    ax.set_xlabel("EE-mug offset r (cm)"); ax.set_ylabel("T* (s)")
    ax.set_title(f"T* not predicted by offset (r={rcoef:+.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_tstar_offset.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved figs/fig_tstar.png + fig_tstar_offset.png")


if __name__ == "__main__":
    main()
