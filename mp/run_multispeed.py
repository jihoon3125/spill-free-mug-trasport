"""Multi-speed ablation: same start/goal, 3 time budgets.

For T ∈ {0.5, 1.0, 1.5} seconds, run all 4 methods.
Show that:
  - Naive baselines collapse as T shrinks (high accel → high tilt)
  - CHOMP+spill stays at 0% spill across all T (or fails only at extreme T)

Outputs:
  results/multispeed.npz
  figs/fig_multispeed.png — line plot: T vs spill_ratio per method
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from mp.stomp import STOMPConfig, stomp_optimize


def upright_pose(p_world):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p_world, float)
    return T


def solve_qpos_for_mug(fk, T_ee_to_mug, T_mug_target, q_init):
    T_ee_target = T_mug_target @ np.linalg.inv(T_ee_to_mug)
    q_opt, info = ik_solve(fk, torch.tensor(q_init, dtype=torch.float64),
                            torch.tensor(T_ee_target, dtype=torch.float64),
                            n_iters=2000, lr=0.03)
    return q_opt.numpy(), info


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    p_start = (0.5, -0.3, 0.25)
    p_goal = (0.5, 0.4, 0.40)
    q_init_seed = g['qpos'][:6]
    q_start, _ = solve_qpos_for_mug(fk, T_em, upright_pose(p_start), q_init_seed)
    q_goal, _ = solve_qpos_for_mug(fk, T_em, upright_pose(p_goal), q_start)
    print(f"start q = {q_start}")
    print(f"goal  q = {q_goal}")

    theta_max = 18.0
    N = 50
    Ts = [0.5, 0.8, 1.0, 1.5, 2.0]      # seconds total
    METHODS = ["minjerk", "chomp", "stomp_spill", "chomp_spill"]
    spill_results = {m: [] for m in METHODS}
    max_tilt_results = {m: [] for m in METHODS}

    for T in Ts:
        dt = T / (N - 1)
        spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)
        print(f"\n=== T={T}s  dt={dt:.4f}s ===")

        # Min-jerk
        q_traj = minjerk_interp(q_start, q_goal, N)
        d_mj = spill.diagnostics(torch.tensor(q_traj, dtype=torch.float64))
        print(f"  minjerk     spill={d_mj['spill_ratio']:.1%}  max_tilt={d_mj['tilt_deg'].max():.1f}°")
        spill_results["minjerk"].append(d_mj['spill_ratio'])
        max_tilt_results["minjerk"].append(d_mj['tilt_deg'].max())

        # CHOMP smooth only
        cfg = CHOMPConfig(N=N, dt=dt, n_iters=300, alpha_smooth=1.0,
                          gamma_spill=0.0, step_size=0.005)
        res = chomp_optimize(fk, q_start, q_goal, cfg, spill=spill)
        d = res["spill_diag"]
        print(f"  chomp       spill={d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")
        spill_results["chomp"].append(d['spill_ratio'])
        max_tilt_results["chomp"].append(d['tilt_deg'].max())

        # STOMP + spill
        torch.manual_seed(0)
        cfg_st = STOMPConfig(N=N, dt=dt, K=50, n_iters=200, sigma_init=0.04,
                              sigma_decay=0.995, lam=0.2, gamma_spill=5.0)
        res_st = stomp_optimize(fk, q_start, q_goal, cfg_st, spill=spill)
        d = res_st["spill_diag"]
        print(f"  stomp+spill spill={d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")
        spill_results["stomp_spill"].append(d['spill_ratio'])
        max_tilt_results["stomp_spill"].append(d['tilt_deg'].max())

        # CHOMP + spill (ours)
        cfg_cs = CHOMPConfig(N=N, dt=dt, n_iters=1000, alpha_smooth=1.0,
                             gamma_spill=2.0, step_size=0.003)
        res_cs = chomp_optimize(fk, q_start, q_goal, cfg_cs, spill=spill)
        d = res_cs["spill_diag"]
        print(f"  chomp+spill spill={d['spill_ratio']:.1%}  max_tilt={d['tilt_deg'].max():.1f}°")
        spill_results["chomp_spill"].append(d['spill_ratio'])
        max_tilt_results["chomp_spill"].append(d['tilt_deg'].max())

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "multispeed.npz",
             Ts=np.array(Ts), theta_max=theta_max,
             **{f"{m}_spill": np.array(spill_results[m]) for m in METHODS},
             **{f"{m}_maxtilt": np.array(max_tilt_results[m]) for m in METHODS})
    print(f"\nsaved {out_dir/'multispeed.npz'}")


if __name__ == "__main__":
    main()
