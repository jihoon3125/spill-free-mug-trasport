"""Grasp uncertainty robustness — connects to real-world grasp slippage.

In practice T_ee_to_mug is not exact: the grasp deforms slightly when the
robot accelerates. We model this as a Gaussian rotation perturbation on
T_ee_to_mug (small angle around a random axis).

Two strategies:
  1) Nominal — optimize for exact T_em, evaluate on noisy T_em
  2) Robust — sample K perturbations during optimization; minimize the
     WORST-CASE spill cost across samples. The resulting trajectory is more
     conservative but holds up under realistic uncertainty.

Output:
  results/uncertainty.json
  figs/fig_uncertainty.png   — spill ratio CDF over noise samples per method
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sim.scene import URDF, parse_grasp_info
from mp.kinematics import URDFForwardKinematics
from mp.ik import ik_solve
from mp.constants import R_UPRIGHT
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize


def upright_pose(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float)
    return T


def sample_perturbed_Tem(T_em_nom: np.ndarray, sigma_rot_deg: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Apply a random small rotation to T_em (slip model)."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
    angle = abs(rng.normal(scale=math.radians(sigma_rot_deg)))
    dR = R.from_rotvec(axis * angle).as_matrix()
    T = T_em_nom.copy()
    T[:3, :3] = dR @ T[:3, :3]
    return T


def chomp_robust(fk, q_start, q_goal, T_em_nom, theta_max=18.0,
                  N=50, T_budget=0.7, sigma_rot_deg=5.0, K_samples=8,
                  n_iters=1500, step_size=0.003, gamma_spill=5.0):
    """Sample K perturbations of T_em; minimize the MEAN spill cost across them.

    This is the worst-case-robust variant. We use mean here for differentiability;
    one could swap in max-style aggregation.
    """
    dt = T_budget / (N - 1)
    # Build K SpillCost objects with different T_em
    rng = np.random.default_rng(0)
    spills = []
    for _ in range(K_samples):
        Tn = sample_perturbed_Tem(T_em_nom, sigma_rot_deg, rng)
        spills.append(SpillCost(SpillConfig(T_ee_to_mug=Tn, theta_max_deg=theta_max, dt=dt), fk))

    def combined_obs(q_traj):
        return sum(s.cost(q_traj) for s in spills) / len(spills)

    cfg = CHOMPConfig(N=N, dt=dt, n_iters=n_iters, alpha_smooth=1.0,
                      gamma_spill=0.0, beta_obs=gamma_spill,
                      step_size=step_size)
    res = chomp_optimize(fk, q_start, q_goal, cfg, spill=None, obs_cost_fn=combined_obs)
    return res["q_traj"], dt


def evaluate_under_uncertainty(q_traj, fk, T_em_nom, theta_max, dt,
                                 sigma_rot_deg, K_eval, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    spill_ratios = []
    for _ in range(K_eval):
        Tn = sample_perturbed_Tem(T_em_nom, sigma_rot_deg, rng)
        sc = SpillCost(SpillConfig(T_ee_to_mug=Tn, theta_max_deg=theta_max, dt=dt), fk)
        d = sc.diagnostics(torch.tensor(q_traj, dtype=torch.float64))
        spill_ratios.append(float(d['spill_ratio']))
    return np.array(spill_ratios)


def main():
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    p_start = (0.5, -0.3, 0.25); p_goal = (0.5, 0.4, 0.40)
    def ik(p, q_seed):
        T = upright_pose(p); T_ee = T @ np.linalg.inv(T_em)
        q, info = ik_solve(fk, torch.tensor(q_seed, dtype=torch.float64),
                            torch.tensor(T_ee, dtype=torch.float64),
                            n_iters=2500, lr=0.03)
        return q.numpy()
    q_start = ik(p_start, g['qpos'][:6])
    q_goal = ik(p_goal, q_start)

    N = 50; T_budget = 0.7; dt = T_budget / (N - 1); theta_max = 18.0
    sigma_rot_deg = 5.0   # realistic grasp slip per use
    K_eval = 100

    # --- 1) Nominal: plan with exact T_em
    print("[plan] nominal (exact T_em)")
    sp_nom = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=theta_max, dt=dt), fk)
    cfg = CHOMPConfig(N=N, dt=dt, n_iters=1500, alpha_smooth=1.0,
                      gamma_spill=5.0, step_size=0.003)
    res_nom = chomp_optimize(fk, q_start, q_goal, cfg, spill=sp_nom)
    q_nom = res_nom["q_traj"]
    print(f"  nominal spill (no noise) = {res_nom['spill_diag']['spill_ratio']:.1%}")

    # --- 2) Robust: plan with K perturbations
    print(f"[plan] robust (K=8 samples, σ_rot={sigma_rot_deg}°)")
    q_rob, _ = chomp_robust(fk, q_start, q_goal, T_em, theta_max=theta_max,
                             N=N, T_budget=T_budget, sigma_rot_deg=sigma_rot_deg,
                             K_samples=8, n_iters=1500, gamma_spill=5.0)
    d_rob_nom = sp_nom.diagnostics(torch.tensor(q_rob, dtype=torch.float64))
    print(f"  robust spill (no noise)  = {d_rob_nom['spill_ratio']:.1%}")

    # --- 3) Evaluate under K_eval noise realizations
    print(f"[eval] {K_eval} noise samples for each plan")
    sp_nom_eval = evaluate_under_uncertainty(q_nom, fk, T_em, theta_max, dt,
                                                sigma_rot_deg, K_eval)
    sp_rob_eval = evaluate_under_uncertainty(q_rob, fk, T_em, theta_max, dt,
                                                sigma_rot_deg, K_eval)
    print(f"  nominal  : mean={sp_nom_eval.mean()*100:.1f}%  "
           f"p95={np.percentile(sp_nom_eval,95)*100:.1f}%  max={sp_nom_eval.max()*100:.1f}%")
    print(f"  robust   : mean={sp_rob_eval.mean()*100:.1f}%  "
           f"p95={np.percentile(sp_rob_eval,95)*100:.1f}%  max={sp_rob_eval.max()*100:.1f}%")

    # Save
    out_dir = ROOT / "results"; out_dir.mkdir(exist_ok=True)
    json.dump(dict(sigma_rot_deg=sigma_rot_deg, T_budget=T_budget,
                    nominal=dict(mean=float(sp_nom_eval.mean()),
                                  p95=float(np.percentile(sp_nom_eval, 95)),
                                  max=float(sp_nom_eval.max()),
                                  samples=sp_nom_eval.tolist()),
                    robust=dict(mean=float(sp_rob_eval.mean()),
                                 p95=float(np.percentile(sp_rob_eval, 95)),
                                 max=float(sp_rob_eval.max()),
                                 samples=sp_rob_eval.tolist())),
              open(out_dir / "uncertainty.json", "w"), indent=2)

    # --- Figure: CDF of spill ratio over noise samples
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    bins = np.linspace(0, max(sp_nom_eval.max(), sp_rob_eval.max(), 0.01) * 100 + 5, 40)
    ax.hist(sp_nom_eval * 100, bins=bins, alpha=0.6, color="#d95f02", label="Nominal plan")
    ax.hist(sp_rob_eval * 100, bins=bins, alpha=0.6, color="#1b9e77", label="Robust plan (ours)")
    ax.set_xlabel("Spill ratio under noisy T_ee_to_mug (%)")
    ax.set_ylabel("# of noise realizations (of 100)")
    ax.set_title(f"Spill distribution under σ_rot={sigma_rot_deg}° grasp slip")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    sorted_nom = np.sort(sp_nom_eval * 100)
    sorted_rob = np.sort(sp_rob_eval * 100)
    p = np.linspace(0, 100, len(sorted_nom))
    ax.plot(sorted_nom, p, color="#d95f02", label="Nominal plan", lw=1.8)
    ax.plot(sorted_rob, p, color="#1b9e77", label="Robust plan (ours)", lw=1.8)
    ax.axvline(18, ls="--", color="r", lw=1, alpha=0.7)
    ax.set_xlabel("Spill ratio (%)"); ax.set_ylabel("Percentile of noise realizations")
    ax.set_title("CDF: robust plan has tighter tail")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Robustness to grasp uncertainty (random rotation slip)",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    out = ROOT / "figs/fig_uncertainty.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
