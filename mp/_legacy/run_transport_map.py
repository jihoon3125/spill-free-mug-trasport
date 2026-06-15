"""Experiment A — Grasp Transportability Map.

For a systematically sampled set of mug_0 grasps, evaluate each grasp on two axes
for one FIXED transport task (upright, 70 cm lateral + 15 cm lift):

  axis 1 — reachability : can IK reach the upright transport start AND goal?
  axis 2 — spill margin  : T_min, the minimum trajectory time at which CHOMP+spill
                           achieves spill-free (<=2%) transport. Lower = more robust.

Also records, per grasp:
  offset_r   = || translation(T_ee_to_mug) ||      (EE -> mug lever arm, for Exp B)
  spill_at   = spill ratio at a fixed reference T   (continuous differentiator)

Sampling is deterministic (evenly spaced idx per taxonomy) — NOT hand-picked.

Usage:
  python mp/run_transport_map.py --per-tax 5      # smoke (15 grasps)
  python mp/run_transport_map.py --per-tax 20     # full (60 grasps)

Output:
  results/transport_map.json
  figs/fig_transport_map.png        — T_min per grasp, grey X for unreachable
  figs/fig_transport_offset.png     — Exp B scatter: T_min vs offset_r
"""
from __future__ import annotations
import argparse, json
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
from mp.chomp import CHOMPConfig, chomp_optimize
from sim.scene import URDF
from mp.run_grasp_comparison import extract_ee_to_mug, TAXONOMY_COLOR

TAXONOMIES = ["2finger", "3finger", "whole"]
N_WP = 50
THETA_MAX = 18.0
SPILL_TOL = 0.02
REF_T = 0.5          # fixed reference time budget for spill_at metric


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def evaluate_T(fk, T_em, q_start, q_goal, T_budget):
    dt = T_budget / (N_WP - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA_MAX, dt=dt), fk)
    cfg = CHOMPConfig(N=N_WP, dt=dt, n_iters=1200, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp)
    return float(res["spill_diag"]["spill_ratio"])


def find_T_min(fk, T_em, q_start, q_goal, T_lo=0.3, T_hi=2.5, tol=0.05, max_iter=8):
    """Smallest T with spill-free transport. None if even T_hi spills."""
    if evaluate_T(fk, T_em, q_start, q_goal, T_hi) > SPILL_TOL:
        return None
    if evaluate_T(fk, T_em, q_start, q_goal, T_lo) <= SPILL_TOL:
        return T_lo
    for _ in range(max_iter):
        if T_hi - T_lo < tol:
            break
        T_mid = 0.5 * (T_lo + T_hi)
        if evaluate_T(fk, T_em, q_start, q_goal, T_mid) <= SPILL_TOL:
            T_hi = T_mid
        else:
            T_lo = T_mid
    return T_hi


def sample_indices(per_tax: int) -> list[int]:
    return sorted(set(np.linspace(0, 299, per_tax).round().astype(int).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tax", type=int, default=5, help="grasps per taxonomy")
    ap.add_argument("--out", default="results/transport_map.json")
    args = ap.parse_args()

    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)
    idxs = sample_indices(args.per_tax)
    print(f"Sampling {len(idxs)} idx/taxonomy = {len(idxs)*3} grasps: {idxs}\n")

    def ik(p, q_seed, n_restarts=4):
        """Multi-seed IK: retry from perturbed seeds so an IK-fail means the pose
        is genuinely unreachable, not an Adam local minimum. Returns best result."""
        T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
        T_ee_t = torch.tensor(T_ee, dtype=torch.float64)
        rng = np.random.default_rng(0)
        best_q, best_info = None, None
        for k in range(n_restarts):
            seed = np.asarray(q_seed, float) if k == 0 else \
                np.asarray(q_seed, float) + rng.normal(0, 0.3, size=len(q_seed))
            q, info = ik_solve(fk, torch.tensor(seed, dtype=torch.float64),
                               T_ee_t, n_iters=2500, lr=0.03)
            if best_info is None or info['pos_err'] < best_info['pos_err']:
                best_q, best_info = q.numpy(), info
            if best_info['pos_err'] <= 0.01:
                break
        return best_q, best_info

    rows = []
    total = len(idxs) * len(TAXONOMIES)
    done = 0
    import time as _time
    t0 = _time.time()
    for tax in TAXONOMIES:
        for idx in idxs:
            done += 1
            elapsed = _time.time() - t0
            eta = (elapsed / done * (total - done)) if done > 1 else 0
            prog = f"[{done:3d}/{total} {done*100//total:3d}%  ETA {eta/60:4.1f}m]"
            name = f"mug_0_left_{tax}_{idx}"
            try:
                T_em, _, ginfo = extract_ee_to_mug(name)
            except Exception as e:
                print(f"{prog} {name}  LOAD FAIL: {e}", flush=True); continue
            offset_r = float(np.linalg.norm(T_em[:3, 3]))
            rec = dict(grasp_name=name, taxonomy=tax, idx=int(idx),
                       offset_r=offset_r, reachable=False,
                       T_min=None, spill_at=None)
            q_s, info_s = ik(p_start, ginfo['qpos'][:6])
            if info_s['pos_err'] > 0.01:
                print(f"{prog} {name}  r={offset_r*100:5.1f}cm  IK-fail(start)", flush=True)
                rows.append(rec); continue
            q_g, info_g = ik(p_goal, q_s)
            if info_g['pos_err'] > 0.01:
                print(f"{prog} {name}  r={offset_r*100:5.1f}cm  IK-fail(goal)", flush=True)
                rows.append(rec); continue
            rec["reachable"] = True
            rec["spill_at"] = evaluate_T(fk, T_em, q_s, q_g, REF_T)
            rec["T_min"] = find_T_min(fk, T_em, q_s, q_g)
            tmin_s = f"{rec['T_min']:.2f}s" if rec["T_min"] else ">2.5s"
            print(f"{prog} {name}  r={offset_r*100:5.1f}cm  "
                  f"spill@{REF_T}s={rec['spill_at']*100:4.0f}%  T_min={tmin_s}", flush=True)
            rows.append(rec)

    out_path = ROOT / args.out
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out_path}")

    # ---- summary stats ----
    reach = [r for r in rows if r["reachable"]]
    print(f"\n=== summary ({len(rows)} grasps) ===")
    for tax in TAXONOMIES:
        tr = [r for r in rows if r["taxonomy"] == tax]
        ok = [r for r in tr if r["reachable"]]
        tmins = [r["T_min"] for r in ok if r["T_min"] is not None]
        print(f"  {tax:8s} reachable {len(ok)}/{len(tr)}  "
              f"T_min[min/med/max]="
              f"{min(tmins) if tmins else float('nan'):.2f}/"
              f"{np.median(tmins) if tmins else float('nan'):.2f}/"
              f"{max(tmins) if tmins else float('nan'):.2f}")

    make_figures(rows)


def make_figures(rows):
    # Fig 1: transportability map (T_min per grasp)
    rows_sorted = sorted(rows, key=lambda r: (r["taxonomy"], r["idx"]))
    labels = [f"{r['taxonomy'][:2]}#{r['idx']}" for r in rows_sorted]
    colors = [TAXONOMY_COLOR[r["taxonomy"]] for r in rows_sorted]
    vals = [r["T_min"] if r["T_min"] is not None else np.nan for r in rows_sorted]
    fig, ax = plt.subplots(figsize=(max(10, len(rows_sorted) * 0.32), 4))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="black", lw=0.4)
    for i, (v, r) in enumerate(zip(vals, rows_sorted)):
        if np.isnan(v):
            txt = "unreach" if not r["reachable"] else ">2.5s"
            ax.text(i, 0.05, txt, ha="center", fontsize=6, color="red", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel(r"$T_{\min}$ (s) — fastest spill-free transport")
    ax.set_title("Grasp transportability map (lower $T_{\\min}$ = more spill-robust)")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, edgecolor='black', label=t)
                       for t, c in TAXONOMY_COLOR.items()], fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_transport_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Fig 2 (Exp B): T_min vs offset_r scatter
    fig, ax = plt.subplots(figsize=(6, 5))
    for tax in TAXONOMIES:
        tr = [r for r in rows if r["taxonomy"] == tax and r["reachable"]
              and r["T_min"] is not None]
        ax.scatter([r["offset_r"] * 100 for r in tr], [r["T_min"] for r in tr],
                   c=TAXONOMY_COLOR[tax], label=tax, edgecolor="black", s=40)
    ax.set_xlabel("EE→mug offset $r$ (cm)")
    ax.set_ylabel(r"$T_{\min}$ (s)")
    ax.set_title("Spill robustness vs grasp offset")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_transport_offset.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved figs/fig_transport_map.png + fig_transport_offset.png")


if __name__ == "__main__":
    main()
