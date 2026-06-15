"""Experiment B1 — spill / feasibility trade-off vs the spill weight gamma.
At T=1.0s, sweep gamma; report mean spill and joint-limit feasibility.
Output: results/gamma_pareto.json, figs/fig_gamma_pareto.png
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
GAMMAS = [0.0, 0.3, 0.7, 1.0, 2.0, 3.0, 5.0]
V_MAX = 3.3; A_MAX = 20.0
N_GRASPS = 15


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    names = [r["grasp_name"] for r in json.load(open(ROOT / "results/transport_map.json"))
             if r["reachable"]]
    sel = [names[i] for i in np.linspace(0, len(names) - 1, min(N_GRASPS, len(names))).round().astype(int)]
    sel = sorted(set(sel))
    qcache = {}
    for name in sel:
        T_em, _, g = extract_ee_to_mug(name)
        qs, i_s = ik_multiseed(fk, T_em, PS, g['qpos'][:6]); qg, i_g = ik_multiseed(fk, T_em, PG, qs)
        if i_s['pos_err'] <= 0.012 and i_g['pos_err'] <= 0.012:
            qcache[name] = (T_em, qs, qg)
    print(f"gamma sweep over {len(qcache)} grasps", flush=True)

    res = {"gammas": GAMMAS, "spill": [], "feasible": []}
    for gam in GAMMAS:
        sp_list, fe_list = [], []
        for name, (T_em, qs, qg) in qcache.items():
            sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
            q = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                               gamma_spill=gam, step_size=0.003), spill=sp)["q_traj"]
            sp_list.append(sp.diagnostics(torch.tensor(q, dtype=torch.float64))["spill_ratio"] * 100)
            v = np.abs(np.diff(q, axis=0) / dt).max(); a = np.abs(np.diff(np.diff(q, axis=0), axis=0) / dt**2).max()
            fe_list.append(v < V_MAX and a < A_MAX)
        res["spill"].append(float(np.mean(sp_list)))
        res["feasible"].append(float(np.mean(fe_list) * 100))
        print(f"  gamma={gam}: spill={np.mean(sp_list):.0f}%  feasible={np.mean(fe_list)*100:.0f}%", flush=True)
    json.dump(res, open(ROOT / "results/gamma_pareto.json", "w"), indent=2)

    fig, ax1 = plt.subplots(figsize=(5.2, 3.6))
    ax1.plot(GAMMAS, res["spill"], "o-", color="#d62728", label="spill")
    ax1.set_xlabel("spill weight gamma"); ax1.set_ylabel("mean spill (%)", color="#d62728")
    ax2 = ax1.twinx(); ax2.plot(GAMMAS, res["feasible"], "s--", color="#2ca02c", label="feasible")
    ax2.set_ylabel("feasible (%)", color="#2ca02c"); ax2.set_ylim(-5, 105)
    ax1.set_title("Spill vs feasibility trade-off (T=1.0 s)"); ax1.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig_gamma_pareto.png", dpi=160, bbox_inches="tight")
    print("saved figs/fig_gamma_pareto.png", flush=True)


if __name__ == "__main__":
    main()
