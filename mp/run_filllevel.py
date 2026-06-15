"""Experiment B2 — spill-robustness vs spill threshold (fill level).
theta_max maps to fill level: theta_max ~= atan((rim - liquid)/radius), so a
fuller cup => smaller theta_max => harder. At T=1.0s, gamma=1.0, plan ours per
grasp and evaluate the fraction of grasps kept spill-free as theta_max varies.
Output: results/filllevel.json, figs/fig_filllevel.png
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
from mp.kinematics import URDFForwardKinematics
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize
from sim.scene import URDF
from mp.run_grasp_comparison import extract_ee_to_mug
from mp.run_method_comparison import ik_multiseed

PS = (0.5, -0.3, 0.25); PG = (0.5, 0.4, 0.40)
N = 50; T = 1.0; dt = T / (N - 1)
THETAS = [10, 15, 18, 22, 26, 30, 35]
N_GRASPS = 20
CUP_R = 3.5; CUP_DEPTH = 8.5      # cm, approx mug interior


def fill_for_theta(th):
    """Approx fill fraction giving this theta_max (headroom = R*tan(theta))."""
    headroom = CUP_R * np.tan(np.radians(th))
    return max(0.0, min(1.0, 1.0 - headroom / CUP_DEPTH))


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    names = [r["grasp_name"] for r in json.load(open(ROOT / "results/transport_map.json"))
             if r["reachable"]]
    sel = [names[i] for i in np.linspace(0, len(names) - 1, min(N_GRASPS, len(names))).round().astype(int)]
    sel = sorted(set(sel))
    tilts = []
    for name in sel:
        T_em, _, g = extract_ee_to_mug(name)
        qs, i_s = ik_multiseed(fk, T_em, PS, g['qpos'][:6]); qg, i_g = ik_multiseed(fk, T_em, PG, qs)
        if i_s['pos_err'] > 0.012 or i_g['pos_err'] > 0.012:
            continue
        sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
        q = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                           gamma_spill=1.0, step_size=0.003), spill=sp)["q_traj"]
        tilts.append(sp.diagnostics(torch.tensor(q, dtype=torch.float64))["tilt_deg"])
    print(f"fill-level over {len(tilts)} grasps", flush=True)

    mean_spill = []; frac_free = []; fills = []
    for th in THETAS:
        sr = [float((t > th).mean()) * 100 for t in tilts]
        mean_spill.append(float(np.mean(sr)))
        frac_free.append(float(np.mean([s <= 2 for s in sr]) * 100))
        fills.append(round(fill_for_theta(th) * 100))
    res = {"thetas": THETAS, "fill_pct": fills, "mean_spill": mean_spill, "frac_spillfree": frac_free}
    json.dump(res, open(ROOT / "results/filllevel.json", "w"), indent=2)
    for th, f, ms, ff in zip(THETAS, fills, mean_spill, frac_free):
        print(f"  theta={th} (~{f}% full): mean spill={ms:.0f}%  grasps spill-free={ff:.0f}%", flush=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(THETAS, frac_free, "o-", color="#2ca02c", label="% grasps kept spill-free")
    ax.plot(THETAS, mean_spill, "s--", color="#d62728", label="mean spill (%)")
    ax.set_xlabel("spill threshold theta_max (deg)"); ax.set_ylabel("%")
    ax.set_title("Spill-robustness vs fill level (T=1.0 s, ours)")
    # secondary fill-level ticks
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(THETAS)
    ax2.set_xticklabels([f"{f}%" for f in fills], fontsize=7)
    ax2.set_xlabel("approx fill level", fontsize=8)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig_filllevel.png", dpi=160, bbox_inches="tight")
    print("saved figs/fig_filllevel.png", flush=True)


if __name__ == "__main__":
    main()
