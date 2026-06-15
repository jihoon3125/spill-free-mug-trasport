"""Experiment A — robustness of the method ranking to (i) time budget T and
(ii) the spill-angle threshold theta_max.

Same task and grasp set as the method comparison (Exp C). For a subset of
reachable grasps, plan 4 methods (min-jerk / chomp-smooth / vanilla-grad / ours)
at several T; evaluate spill at several theta thresholds (post-processing the
per-waypoint effective tilt, so a single plan covers all thresholds).

(STOMP omitted: it diverges and is ~4x slower; see method-comparison box plot.)

Output:
  results/robustness.json
  figs/fig_robust_T.png       mean spill vs T (theta=18)
  figs/fig_robust_theta.png   mean spill vs theta_max (T=0.5)
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
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from sim.scene import URDF
from mp.run_grasp_comparison import extract_ee_to_mug
from mp.run_method_comparison import ik_multiseed

PS = (0.5, -0.3, 0.25); PG = (0.5, 0.4, 0.40)        # same task as Exp C
N = 50
TS = [0.4, 0.5, 0.6, 0.8, 1.0]
THETAS = [10, 15, 18, 22, 28, 35]
N_GRASPS = 12
COL = {"min-jerk": "#999999", "chomp-smooth": "#1f77b4",
       "vanilla-grad": "#ff7f0e", "ours": "#2ca02c"}


def plan_all(fk, T_em, qs, qg, T):
    dt = T / (N - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
    out = {}
    out["min-jerk"] = minjerk_interp(qs, qg, N)
    out["chomp-smooth"] = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                          gamma_spill=0.0, step_size=0.003), spill=sp)["q_traj"]
    out["vanilla-grad"] = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                          gamma_spill=5.0, step_size=0.003, use_covariant=False), spill=sp)["q_traj"]
    out["ours"] = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                  gamma_spill=5.0, step_size=0.003), spill=sp)["q_traj"]
    tilts = {m: sp.diagnostics(torch.tensor(q, dtype=torch.float64))["tilt_deg"] for m, q in out.items()}
    return tilts


def upright(p):
    M = np.eye(4); M[:3, :3] = R_UPRIGHT; M[:3, 3] = np.asarray(p, float); return M


def spill_ratio(tilt, theta):
    return float((tilt > theta).mean())


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    names = [g["grasp_name"] for g in json.load(open(ROOT / "results/method_comparison.json"))["per_grasp"]]
    sel = np.linspace(0, len(names) - 1, N_GRASPS).round().astype(int)
    grasps = [names[i] for i in sorted(set(sel.tolist()))]
    print(f"robustness over {len(grasps)} grasps\n")

    # data[T][method] = list of tilt arrays (one per grasp)
    data = {T: {m: [] for m in COL} for T in TS}
    for gi, name in enumerate(grasps):
        T_em, _, ginfo = extract_ee_to_mug(name)
        qs, i_s = ik_multiseed(fk, T_em, PS, ginfo['qpos'][:6])
        qg, i_g = ik_multiseed(fk, T_em, PG, qs)
        if i_s['pos_err'] > 0.012 or i_g['pos_err'] > 0.012:
            print(f"  [{gi}] {name} IK-fail, skip"); continue
        for T in TS:
            tilts = plan_all(fk, T_em, qs, qg, T)
            for m in COL:
                data[T][m].append(tilts[m])
        print(f"  [{gi+1}/{len(grasps)}] {name} done", flush=True)

    # aggregate
    def mean_spill(T, m, theta):
        return float(np.mean([spill_ratio(t, theta) for t in data[T][m]])) * 100

    res = {"grasps": grasps, "TS": TS, "THETAS": THETAS,
           "spill_vs_T_theta18": {m: [mean_spill(T, m, 18) for T in TS] for m in COL},
           "spill_vs_theta_T0.5": {m: [mean_spill(0.5, m, th) for th in THETAS] for m in COL}}
    json.dump(res, open(ROOT / "results/robustness.json", "w"), indent=2)

    print("\n=== mean spill vs T (theta=18) ===")
    print("T      " + "  ".join(f"{m:>13s}" for m in COL))
    for i, T in enumerate(TS):
        print(f"{T:<6}" + "  ".join(f"{res['spill_vs_T_theta18'][m][i]:12.0f}%" for m in COL))
    print("\n=== mean spill vs theta_max (T=0.5) ===")
    print("theta  " + "  ".join(f"{m:>13s}" for m in COL))
    for i, th in enumerate(THETAS):
        print(f"{th:<6}" + "  ".join(f"{res['spill_vs_theta_T0.5'][m][i]:12.0f}%" for m in COL))

    # figures
    for fname, key, xs, xlabel, title in [
        ("fig_robust_T.png", "spill_vs_T_theta18", TS, "time budget T (s)",
         "Spill vs time budget (threshold 18 deg)"),
        ("fig_robust_theta.png", "spill_vs_theta_T0.5", THETAS, "spill threshold theta_max (deg)",
         "Spill vs spill threshold (T = 0.5 s)")]:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for m in COL:
            ax.plot(xs, res[key][m], "o-", color=COL[m], label=m, lw=2)
        ax.set_xlabel(xlabel); ax.set_ylabel("mean spill ratio (%)")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(ROOT / "figs" / fname, dpi=160, bbox_inches="tight")
        print(f"saved figs/{fname}")


if __name__ == "__main__":
    main()
