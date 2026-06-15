"""Experiment C — Method comparison across many grasps (fixed task, fixed T).

Fixes the N=1 weakness of run_ablation.py: instead of one grasp, evaluate every
REACHABLE grasp from the transportability map (Exp A) on the SAME transport task
at a SINGLE challenging time budget, and report the per-method spill DISTRIBUTION.

Methods (each row changes exactly one ingredient — controlled ablation, no strawman):
  min-jerk        : no spill term, analytic baseline
  chomp-smooth    : gradient + smoothness metric, no spill term
  stomp-spill     : spill term, gradient-FREE sampling
  vanilla-grad    : spill term, gradient, NO CHOMP M^-1 preconditioning
  ours (chomp+spill): spill term, gradient, WITH CHOMP M^-1 preconditioning

Plan time is recorded per method so the sampling-vs-gradient comparison is
transparent (STOMP gets >= as many cost evaluations as CHOMP).

Usage:
  python mp/run_method_comparison.py --T 0.6 --max-grasps 40

Output:
  results/method_comparison.json
  figs/fig_method_comparison.png   — box/strip plot of spill ratio per method
"""
from __future__ import annotations
import argparse, json, time
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
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from mp.stomp import STOMPConfig, stomp_optimize
from sim.scene import URDF
from mp.run_grasp_comparison import extract_ee_to_mug
from mp.run_ablation import trajectory_length, jerk_integral

N_WP = 50
THETA_MAX = 18.0
GAMMA = 1.0          # feasible weight (gamma=5 produced joint-limit-violating "cheats")
V_MAX = 3.3; A_MAX = 20.0

METHOD_ORDER = ["min-jerk", "chomp-smooth", "stomp-spill", "vanilla-grad", "ours"]
METHOD_COLOR = {"min-jerk": "#999999", "chomp-smooth": "#1f77b4",
                "stomp-spill": "#9467bd", "vanilla-grad": "#ff7f0e", "ours": "#2ca02c"}


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def ik_multiseed(fk, T_em, p, q_seed, n_restarts=4):
    T = upright_pose(p); T_ee = torch.tensor(T @ np.linalg.inv(T_em), dtype=torch.float64)
    rng = np.random.default_rng(0)
    best_q, best_info = None, None
    for k in range(n_restarts):
        seed = np.asarray(q_seed, float) if k == 0 else \
            np.asarray(q_seed, float) + rng.normal(0, 0.3, size=len(q_seed))
        q, info = ik_solve(fk, torch.tensor(seed, dtype=torch.float64), T_ee,
                           n_iters=2500, lr=0.03)
        if best_info is None or info['pos_err'] < best_info['pos_err']:
            best_q, best_info = q.numpy(), info
        if best_info['pos_err'] <= 0.01:
            break
    return best_q, best_info


def run_methods(fk, T_em, q_start, q_goal, T_budget):
    dt = T_budget / (N_WP - 1)
    spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA_MAX, dt=dt), fk)
    res = {}

    # min-jerk
    t0 = time.time()
    q = minjerk_interp(q_start, q_goal, N_WP)
    d = spill.diagnostics(torch.tensor(q, dtype=torch.float64))
    res["min-jerk"] = _pack(q, d, dt, time.time() - t0)

    # chomp-smooth (no spill)
    cfg = CHOMPConfig(N=N_WP, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=0.0, step_size=0.003, use_covariant=True)
    t0 = time.time(); r = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
    res["chomp-smooth"] = _pack(r["q_traj"], r["spill_diag"], dt, time.time() - t0)

    # stomp-spill (gradient-free)
    cfg = STOMPConfig(N=N_WP, dt=dt, K=50, n_iters=300, sigma_init=0.04,
                      sigma_decay=0.995, lam=0.2, gamma_spill=GAMMA)
    torch.manual_seed(0)
    t0 = time.time(); r = stomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
    res["stomp-spill"] = _pack(r["q_traj"], r["spill_diag"], dt, time.time() - t0)

    # vanilla-grad (spill, gradient, no M^-1)
    cfg = CHOMPConfig(N=N_WP, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=GAMMA, step_size=0.003, use_covariant=False)
    t0 = time.time(); r = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
    res["vanilla-grad"] = _pack(r["q_traj"], r["spill_diag"], dt, time.time() - t0)

    # ours (chomp+spill)
    cfg = CHOMPConfig(N=N_WP, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=GAMMA, step_size=0.003, use_covariant=True)
    t0 = time.time(); r = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
    res["ours"] = _pack(r["q_traj"], r["spill_diag"], dt, time.time() - t0)
    return res


def _pack(q, d, dt, plan_time):
    v = np.abs(np.diff(q, axis=0) / dt)
    a = np.abs(np.diff(np.diff(q, axis=0), axis=0) / dt ** 2)
    return dict(spill_ratio=float(d["spill_ratio"]),
                max_tilt=float(d["tilt_deg"].max()),
                jerk=jerk_integral(q, dt),
                traj_len=trajectory_length(q),
                max_qvel=float(v.max()), max_qacc=float(a.max()),
                feasible=bool(v.max() < V_MAX and a.max() < A_MAX),
                plan_time=plan_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=0.6, help="fixed time budget (s)")
    ap.add_argument("--max-grasps", type=int, default=40)
    ap.add_argument("--map", default="results/transport_map.json")
    args = ap.parse_args()

    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)

    tmap = json.load(open(ROOT / args.map))
    reachable = [r for r in tmap if r.get("reachable")]
    if len(reachable) > args.max_grasps:
        sel = np.linspace(0, len(reachable) - 1, args.max_grasps).round().astype(int)
        reachable = [reachable[i] for i in sorted(set(sel.tolist()))]
    print(f"Method comparison on {len(reachable)} reachable grasps @ T={args.T}s\n")

    per_grasp = []
    for gr in reachable:
        name = gr["grasp_name"]
        try:
            T_em, _, ginfo = extract_ee_to_mug(name)
        except Exception as e:
            print(f"--- {name} load fail: {e}"); continue
        q_s, info_s = ik_multiseed(fk, T_em, p_start, ginfo['qpos'][:6])
        if info_s['pos_err'] > 0.01:
            print(f"--- {name} IK-fail(start) — skip"); continue
        q_g, info_g = ik_multiseed(fk, T_em, p_goal, q_s)
        if info_g['pos_err'] > 0.01:
            print(f"--- {name} IK-fail(goal) — skip"); continue
        res = run_methods(fk, T_em, q_s, q_g, args.T)
        per_grasp.append(dict(grasp_name=name, taxonomy=gr["taxonomy"], methods=res))
        s = "  ".join(f"{m}={res[m]['spill_ratio']*100:.0f}%" for m in METHOD_ORDER)
        print(f"--- {name}  {s}")

    out = dict(T=args.T, n_grasps=len(per_grasp), per_grasp=per_grasp)
    out_path = ROOT / "results/method_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nsaved {out_path}")

    # aggregate
    print(f"\n=== aggregate spill ratio over {len(per_grasp)} grasps (T={args.T}s) ===")
    print(f"{'method':14s} {'mean':>7s} {'median':>7s} {'%@0spill':>9s} {'%feasible':>9s} {'plan_s':>7s}")
    agg = {}
    for m in METHOD_ORDER:
        sr = np.array([g["methods"][m]["spill_ratio"] for g in per_grasp])
        feas = np.mean([g["methods"][m]["feasible"] for g in per_grasp]) * 100
        pt = np.mean([g["methods"][m]["plan_time"] for g in per_grasp])
        agg[m] = sr
        print(f"{m:14s} {sr.mean()*100:6.1f}% {np.median(sr)*100:6.1f}% "
              f"{(sr<=0.02).mean()*100:7.0f}% {feas:8.0f}%  {pt:6.2f}")

    make_figure(agg, args.T, len(per_grasp))


def make_figure(agg, T, n):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [agg[m] * 100 for m in METHOD_ORDER]
    bp = ax.boxplot(data, labels=METHOD_ORDER, patch_artist=True, showmeans=True,
                    medianprops=dict(color="black"))
    for patch, m in zip(bp["boxes"], METHOD_ORDER):
        patch.set_facecolor(METHOD_COLOR[m]); patch.set_alpha(0.6)
    # overlay individual grasps
    rng = np.random.default_rng(0)
    for i, m in enumerate(METHOD_ORDER):
        x = rng.normal(i + 1, 0.05, size=len(agg[m]))
        ax.scatter(x, agg[m] * 100, s=12, c="black", alpha=0.4, zorder=3)
    ax.set_ylabel("Spill ratio (%)")
    ax.set_title(f"Method comparison across {n} grasps "
                 f"(fixed transport task, T={T}s)\nlower is better")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figs/fig_method_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
