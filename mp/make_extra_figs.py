"""Extra figures for the report:
  fig_cone.png   - apparent-gravity rim-cone schematic (safe vs spill)
  fig_accel.png  - mug-center acceleration magnitude over time (mechanism)
  fig_reach.png  - reachability by taxonomy
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrow

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp


# ---------- Fig: cone schematic ----------
def fig_cone():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (ax_label, a_vec, title) in zip(axes, [
            ("safe", np.array([0.30, 0]), "Safe: −g_eff inside rim cone"),
            ("spill", np.array([0.95, 0]), "Spill: −g_eff outside rim cone")]):
        th = 18.0
        # cup (simple trapezoid, opening up), opening axis n = +y
        ax.plot([-0.32, -0.22], [-0.6, 0.2], "k", lw=2)
        ax.plot([0.32, 0.22], [-0.6, 0.2], "k", lw=2)
        ax.plot([-0.32, 0.32], [-0.6, -0.6], "k", lw=2)
        # rim cone around n=+y, half-angle th
        ax.add_patch(Wedge((0, 0), 0.9, 90 - th, 90 + th, color="#2ca02c", alpha=0.18))
        for s in (-1, 1):
            ang = np.radians(90 + s * th)
            ax.plot([0, 0.9 * np.cos(ang)], [0, 0.9 * np.sin(ang)], "--", color="#2ca02c", lw=1)
        # n axis
        ax.annotate("", xy=(0, 0.85), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
        ax.text(0.03, 0.8, "n (cup up)", fontsize=9)
        # gravity g (down) and accel a, effective up = -(g-a) = a - g
        g = np.array([0, -1.0])
        geff = g - a_vec
        up_app = -geff / np.linalg.norm(geff) * 0.85
        ax.annotate("", xy=(0, -0.95), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="gray", lw=2))
        ax.text(0.03, -0.9, "g", fontsize=9, color="gray")
        ax.annotate("", xy=(a_vec[0], 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#1f77b4", lw=2))
        ax.text(a_vec[0] * 0.5, 0.05, "a", fontsize=9, color="#1f77b4")
        col = "#2ca02c" if ax_label == "safe" else "#d62728"
        ax.annotate("", xy=(up_app[0], up_app[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.text(up_app[0] + 0.02, up_app[1], "−g_eff", fontsize=9, color=col)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)
    fig.suptitle("Apparent-gravity rim-cone spill condition", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_cone.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved figs/fig_cone.png")


# ---------- Fig: acceleration profile ----------
def fig_accel():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    g = parse_grasp_info(ROOT / "data/grasp_info.txt"); T_em = np.load(ROOT / "data/ee_to_mug.npy")
    PS = (0.5, -0.3, 0.25); PG = (0.5, 0.4, 0.40); T = 1.0; N = 50; dt = T / (N - 1)

    def up(p):
        M = np.eye(4); M[:3, :3] = R_UPRIGHT; M[:3, 3] = np.array(p); return M

    def ik(p, seed):
        Tee = torch.tensor(up(p) @ np.linalg.inv(T_em), dtype=torch.float64)
        q, _ = ik_solve(fk, torch.tensor(seed, dtype=torch.float64), Tee, n_iters=2500, lr=0.03)
        return q.numpy()
    qs = ik(PS, g['qpos'][:6]); qg = ik(PG, qs)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=18.0, dt=dt), fk)
    trajs = {"min-jerk": minjerk_interp(qs, qg, N),
             "chomp-smooth": chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                             gamma_spill=0.0, step_size=0.003), spill=sp)["q_traj"],
             "ours": chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200,
                     gamma_spill=1.0, step_size=0.003), spill=sp)["q_traj"]}
    col = {"min-jerk": "#999999", "chomp-smooth": "#1f77b4", "ours": "#2ca02c"}
    t = np.arange(N) * dt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for m, q in trajs.items():
        p = np.array([(fk(torch.tensor(q[i], dtype=torch.float64)).detach().numpy() @ T_em)[:3, 3] for i in range(N)])
        a = np.zeros(N); a[1:-1] = np.linalg.norm((p[2:] - 2 * p[1:-1] + p[:-2]) / dt**2, axis=1)
        a[0] = a[1]; a[-1] = a[-2]
        ax.plot(t, a, color=col[m], lw=2, label=f"{m} (peak {a.max():.0f})")
        print(f"  {m}: peak |a| = {a.max():.1f} m/s^2")
    ax.set_xlabel("time (s)"); ax.set_ylabel("mug acceleration |a| (m/s²)")
    ax.set_title("Ours lowers peak mug acceleration"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig_accel.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved figs/fig_accel.png")


# ---------- Fig: reachability by taxonomy ----------
def fig_reach():
    rows = json.load(open(ROOT / "results/transport_map.json"))
    taxa = ["2finger", "3finger", "whole"]
    reach = [sum(1 for r in rows if r["taxonomy"] == t and r["reachable"]) for t in taxa]
    tot = [sum(1 for r in rows if r["taxonomy"] == t) for t in taxa]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    x = np.arange(len(taxa))
    ax.bar(x, tot, color="#dddddd", edgecolor="black", label="total")
    ax.bar(x, reach, color="#2ca02c", edgecolor="black", label="reachable")
    for i, (r, tt) in enumerate(zip(reach, tot)):
        ax.text(i, tt + 0.3, f"{r}/{tt}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(taxa); ax.set_ylabel("# grasps")
    ax.set_title(f"Reachability by taxonomy ({sum(reach)}/{sum(tot)} overall)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_reach.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved figs/fig_reach.png")


# ---------- Fig: feasibility (joint-limit compliance) per method ----------
def fig_feasibility():
    mc = json.load(open(ROOT / "results/method_comparison.json"))["per_grasp"]
    methods = ["min-jerk", "chomp-smooth", "stomp-spill", "vanilla-grad", "ours"]
    col = {"min-jerk": "#999999", "chomp-smooth": "#1f77b4", "stomp-spill": "#9467bd",
           "vanilla-grad": "#ff7f0e", "ours": "#2ca02c"}
    accs = {m: [g["methods"][m]["max_qacc"] for g in mc] for m in methods}
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.boxplot([accs[m] for m in methods], labels=methods, patch_artist=True,
               medianprops=dict(color="black"), showfliers=False)
    for patch, m in zip(ax.patches if hasattr(ax, "patches") else [], methods):
        pass
    ax.axhline(20, color="red", ls="--", lw=1.2, label="joint-accel limit")
    ax.set_yscale("log"); ax.set_ylabel("max joint acceleration (rad/s², log)")
    ax.set_title("Feasibility: only ours and chomp-smooth stay within limits (T=1.0 s)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figs/fig_feasibility.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("saved figs/fig_feasibility.png")


if __name__ == "__main__":
    fig_cone()
    fig_accel()
    fig_reach()
    fig_feasibility()
