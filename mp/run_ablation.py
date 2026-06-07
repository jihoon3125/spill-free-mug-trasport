"""3-way ablation: linear interp / CHOMP / CHOMP+spill.

Task: transport mug from (0.5, 0.0, 0.25) to (0.5, 0.4, 0.35), keeping it upright.
Goal pose: same orientation (upright), translated 40cm in +y, raised 10cm in z.

For each method:
  - generate joint trajectory q(t), shape (N, 6) UR5e DOF
  - compute spill diagnostics (per-waypoint tilt, spill_ratio)
  - compute trajectory length & smoothness (jerk integral)
  - record plan time

Outputs:
  results/ablation.npz   q_traj + diagnostics per method
  results/ablation.csv   summary metrics
"""
from __future__ import annotations
import time, csv
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
from mp.chomp import CHOMPConfig, chomp_optimize, linear_interp, minjerk_interp
from mp.stomp import STOMPConfig, stomp_optimize


def upright_pose(p_world):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p_world, float)
    return T


def solve_qpos_for_mug(fk, T_ee_to_mug, T_mug_target, q_init):
    T_ee_target = T_mug_target @ np.linalg.inv(T_ee_to_mug)
    q_init_t = torch.tensor(q_init, dtype=torch.float64)
    T_target_t = torch.tensor(T_ee_target, dtype=torch.float64)
    q_opt, info = ik_solve(fk, q_init_t, T_target_t, n_iters=2000, lr=0.03)
    return q_opt.numpy(), info


def trajectory_length(q_traj):
    return float(np.sum(np.linalg.norm(np.diff(q_traj, axis=0), axis=1)))


def jerk_integral(q_traj, dt):
    """Sum of squared jerk (3rd derivative) — proxy for smoothness."""
    a = (q_traj[2:] - 2 * q_traj[1:-1] + q_traj[:-2]) / (dt ** 2)
    j = (a[1:] - a[:-1]) / dt
    return float(np.sum(np.linalg.norm(j, axis=1) ** 2))


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    # Targets — challenging 70 cm lateral + 15 cm lift in ~1 sec
    p_start = (0.5, -0.3, 0.25)
    p_goal = (0.5, 0.4, 0.40)
    q_init_seed = g['qpos'][:6]

    # Solve IK for start
    print("[ik] solving start ...")
    q_start, info_s = solve_qpos_for_mug(fk, T_em, upright_pose(p_start), q_init_seed)
    print(f"  pos_err={info_s['pos_err']:.5f} rot_err={info_s['rot_err']:.5f}")
    # Solve IK for goal — seed with q_start
    print("[ik] solving goal ...")
    q_goal, info_g = solve_qpos_for_mug(fk, T_em, upright_pose(p_goal), q_start)
    print(f"  pos_err={info_g['pos_err']:.5f} rot_err={info_g['rot_err']:.5f}")

    # Common config — short time to force high accel
    N = 50
    dt = 0.02            # 1-second trajectory
    theta_max = 18.0     # spill tolerance (deg, tight enough to differentiate)

    spill = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)

    methods: dict[str, dict] = {}

    # ----- 1. Min-jerk (rest-to-rest naive baseline) -----
    print("\n[method] min-jerk (naive)")
    t0 = time.time()
    q_lin = minjerk_interp(q_start, q_goal, N)
    t_plan = time.time() - t0
    diag = spill.diagnostics(torch.tensor(q_lin, dtype=torch.float64))
    methods["minjerk"] = dict(q_traj=q_lin, diag=diag, plan_time=t_plan)
    print(f"  spill_ratio={diag['spill_ratio']:.1%}  max_tilt={diag['tilt_deg'].max():.1f}deg")

    # ----- 2. CHOMP (smooth only) -----
    print("\n[method] CHOMP (smooth only)")
    cfg_chomp = CHOMPConfig(N=N, dt=dt, n_iters=300, alpha_smooth=1.0,
                            gamma_spill=0.0, beta_obs=0.0, step_size=0.005, verbose=False)
    t0 = time.time()
    res_c = chomp_optimize(fk, q_start, q_goal, cfg_chomp, spill=spill)
    t_plan_c = time.time() - t0
    methods["chomp"] = dict(q_traj=res_c["q_traj"], diag=res_c["spill_diag"],
                            plan_time=t_plan_c, loss_hist=res_c["loss_hist"])
    print(f"  spill_ratio={res_c['spill_diag']['spill_ratio']:.1%}  "
          f"max_tilt={res_c['spill_diag']['tilt_deg'].max():.1f}deg")

    # ----- 3. CHOMP + spill -----
    print("\n[method] CHOMP + spill")
    cfg_cs = CHOMPConfig(N=N, dt=dt, n_iters=1000, alpha_smooth=1.0,
                         gamma_spill=2.0, beta_obs=0.0, step_size=0.003, verbose=False)
    t0 = time.time()
    res_cs = chomp_optimize(fk, q_start, q_goal, cfg_cs, spill=spill)
    t_plan_cs = time.time() - t0
    methods["chomp_spill"] = dict(q_traj=res_cs["q_traj"], diag=res_cs["spill_diag"],
                                  plan_time=t_plan_cs, loss_hist=res_cs["loss_hist"])
    print(f"  spill_ratio={res_cs['spill_diag']['spill_ratio']:.1%}  "
          f"max_tilt={res_cs['spill_diag']['tilt_deg'].max():.1f}deg")

    # ----- 4. STOMP + spill -----
    print("\n[method] STOMP + spill (gradient-free)")
    cfg_st = STOMPConfig(N=N, dt=dt, K=50, n_iters=300, sigma_init=0.04,
                          sigma_decay=0.995, lam=0.2, gamma_spill=5.0, verbose=False)
    torch.manual_seed(0)
    t0 = time.time()
    res_st = stomp_optimize(fk, q_start, q_goal, cfg_st, spill=spill)
    t_plan_st = time.time() - t0
    methods["stomp_spill"] = dict(q_traj=res_st["q_traj"], diag=res_st["spill_diag"],
                                   plan_time=t_plan_st, loss_hist=res_st["loss_hist"])
    print(f"  spill_ratio={res_st['spill_diag']['spill_ratio']:.1%}  "
          f"max_tilt={res_st['spill_diag']['tilt_deg'].max():.1f}deg")

    # ----- Summary -----
    rows = []
    for name, m in methods.items():
        q = m['q_traj']; d = m['diag']
        rows.append(dict(
            method=name,
            spill_ratio=f"{d['spill_ratio']:.3f}",
            max_tilt_deg=f"{d['tilt_deg'].max():.2f}",
            mean_tilt_deg=f"{d['tilt_deg'].mean():.2f}",
            traj_len_rad=f"{trajectory_length(q):.3f}",
            jerk_int=f"{jerk_integral(q, dt):.3e}",
            plan_time_s=f"{m['plan_time']:.2f}",
        ))
    print("\n=== Summary ===")
    keys = list(rows[0].keys())
    print(" | ".join(f"{k:>15s}" for k in keys))
    for r in rows:
        print(" | ".join(f"{r[k]:>15s}" for k in keys))

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "ablation.npz",
             q_start=q_start, q_goal=q_goal, dt=dt, theta_max_deg=theta_max,
             T_ee_to_mug=T_em,
             minjerk_q=methods["minjerk"]["q_traj"],
             chomp_q=methods["chomp"]["q_traj"],
             chomp_spill_q=methods["chomp_spill"]["q_traj"],
             stomp_spill_q=methods["stomp_spill"]["q_traj"],
             minjerk_tilt=methods["minjerk"]["diag"]["tilt_deg"],
             chomp_tilt=methods["chomp"]["diag"]["tilt_deg"],
             chomp_spill_tilt=methods["chomp_spill"]["diag"]["tilt_deg"],
             stomp_spill_tilt=methods["stomp_spill"]["diag"]["tilt_deg"],
             minjerk_pmug=methods["minjerk"]["diag"]["p_mug"],
             chomp_pmug=methods["chomp"]["diag"]["p_mug"],
             chomp_spill_pmug=methods["chomp_spill"]["diag"]["p_mug"],
             stomp_spill_pmug=methods["stomp_spill"]["diag"]["p_mug"],
             )
    with open(out_dir / "ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"\nsaved {out_dir/'ablation.npz'} + ablation.csv")


if __name__ == "__main__":
    main()
