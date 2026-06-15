"""Pouring scenario — time-varying θ_max(t).

Task: transport mug from start (upright at A) to pouring posture at B
(70° tilt for pouring into receptacle).

Time-varying spill threshold:
  θ_max(t) = 10° for t/T < 0.7   (transport phase — strict)
  θ_max(t) ramps up to 90°       (pouring phase — allow tilt)
  Smooth sigmoid transition.

The challenge for CHOMP+spill: stay upright during transport (first 70% of
trajectory), then smoothly tilt to reach the pouring goal.

Compare:
  - Min-jerk (naive)         : tilts gradually from start → "spills" early
  - CHOMP+const θ_max=70°    : permissive, may tilt early (drinking from
                                 spilled cup)
  - CHOMP+timed θ_max (ours) : respects timed window — upright then tilt
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
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from scipy.spatial.transform import Rotation as R


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def tilted_pose(p, tilt_deg=70.0, axis="x"):
    """Mug local +y tilted by `tilt_deg` around `axis` from world +z."""
    base = R_UPRIGHT
    rot = R.from_euler(axis, tilt_deg, degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = rot @ base; T[:3, 3] = np.asarray(p, float)
    return T


def sigmoid_schedule(N, t_start_frac=0.5, t_end_frac=0.9,
                       theta_low=10.0, theta_high=90.0, sharpness=10.0):
    """Sigmoid θ_max(t): low for early phase, ramps up to high for pouring phase."""
    tau = np.linspace(0, 1, N)
    s_start = 1.0 / (1.0 + np.exp(-sharpness * (tau - t_start_frac)))
    s_end = 1.0 / (1.0 + np.exp(sharpness * (tau - t_end_frac)))
    # Combine: rises around t_start, stays high until t_end
    s = s_start * s_end / max(s_start.max() * s_end.max(), 1e-6)
    return theta_low + (theta_high - theta_low) * s


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    p_start = (0.5, -0.3, 0.30)
    p_goal = (0.5, 0.3, 0.40)
    tilt_deg_goal = 70.0
    T_start_pose = upright_pose(p_start)
    T_goal_pose = tilted_pose(p_goal, tilt_deg=tilt_deg_goal, axis="x")

    def ik(T_target, q_seed):
        T_ee = T_target @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy(), info
    q_start, _ = ik(T_start_pose, g['qpos'][:6])
    q_goal, info_g = ik(T_goal_pose, q_start)
    print(f"goal IK err: pos={info_g['pos_err']:.4f} rot={info_g['rot_err']:.4f}")

    N = 80; T_budget = 1.5; dt = T_budget / (N - 1)
    schedule = sigmoid_schedule(N, t_start_frac=0.6, t_end_frac=0.95,
                                  theta_low=10.0, theta_high=90.0)
    print(f"θ_max schedule: {schedule.min():.1f}° → {schedule.max():.1f}°")

    methods: dict[str, dict] = {}

    # 1) Min-jerk
    print("[method] min-jerk (naive)")
    q_traj = minjerk_interp(q_start, q_goal, N)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=schedule, dt=dt), fk)
    d = sp.diagnostics(torch.tensor(q_traj, dtype=torch.float64))
    methods["minjerk"] = dict(q=q_traj, diag=d)
    print(f"  spill (vs timed θ_max) = {d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")

    # 2) CHOMP + constant θ_max=70° (permissive)
    print("[method] CHOMP + const θ_max=70°")
    sp_const = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=70.0, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=800, alpha_smooth=1.0,
                      gamma_spill=2.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp_const)
    # evaluate against TIMED schedule
    d_eval = sp.diagnostics(torch.tensor(res["q_traj"], dtype=torch.float64))
    methods["chomp_const"] = dict(q=res["q_traj"], diag=d_eval)
    print(f"  spill (vs timed θ_max) = {d_eval['spill_ratio']:.1%}  max_tilt={d_eval['tilt_deg'].max():.1f}°")

    # 3) CHOMP + timed θ_max(t) (ours)
    print("[method] CHOMP + timed θ_max(t) (ours)")
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp)
    d = sp.diagnostics(torch.tensor(res["q_traj"], dtype=torch.float64))
    methods["chomp_timed"] = dict(q=res["q_traj"], diag=d)
    print(f"  spill (vs timed θ_max) = {d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")

    # Save
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "pouring.npz",
             schedule=schedule, dt=dt, T_budget=T_budget,
             **{f"{k}_q": v["q"] for k, v in methods.items()},
             **{f"{k}_tilt": v["diag"]["tilt_deg"] for k, v in methods.items()},
             **{f"{k}_pmug": v["diag"]["p_mug"] for k, v in methods.items()})

    # ----- Figure: tilt vs schedule -----
    t = np.arange(N) * dt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"minjerk": "#d95f02", "chomp_const": "#7570b3", "chomp_timed": "#1b9e77"}
    labels = {"minjerk": "Min-jerk (naive)",
              "chomp_const": "CHOMP + const θ_max=70°",
              "chomp_timed": "CHOMP + timed θ_max(t) (ours)"}
    for m in methods:
        ax.plot(t, methods[m]["diag"]["tilt_deg"], color=colors[m], label=labels[m], lw=1.7)
    ax.fill_between(t, 0, schedule, color="grey", alpha=0.15, label="allowed tilt zone (θ_max(t))")
    ax.plot(t, schedule, color="red", lw=1.5, ls="--", label="θ_max(t) schedule")
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Mug effective tilt (deg)")
    ax.set_title("Pouring task — tilt vs time-varying θ_max")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ROOT / "figs/fig_pouring.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
