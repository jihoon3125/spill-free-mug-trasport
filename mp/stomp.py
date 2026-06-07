"""STOMP — Stochastic Trajectory Optimization for Motion Planning.

Gradient-free counterpart of CHOMP. At each iteration:
  1. Sample K noisy perturbations of the current trajectory.
     Noise is *correlated* via M^{-1/2} so it respects the smoothness metric.
  2. Evaluate cost J(traj + noise_k) for each candidate.
  3. Importance-weight (softmin): w_k ∝ exp(−(J_k − min J) / λ).
  4. Update trajectory with weighted average of noise.

Same cost as our CHOMP: J = α J_smooth + γ J_spill (+ optional obs).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from .kinematics import URDFForwardKinematics
from .spill_cost import SpillCost
from .chomp import make_smoothness_metric, minjerk_interp


@dataclass
class STOMPConfig:
    N: int = 50
    dt: float = 0.02
    K: int = 30                  # number of noisy samples per iter
    n_iters: int = 80
    sigma_init: float = 0.10     # noise std (joint-radians scale)
    sigma_decay: float = 0.98
    lam: float = 0.3             # softmin temperature
    alpha_smooth: float = 1.0
    gamma_spill: float = 5.0
    verbose: bool = False


def stomp_optimize(
    fk: URDFForwardKinematics,
    q_init: np.ndarray,
    q_goal: np.ndarray,
    cfg: STOMPConfig,
    spill: SpillCost | None = None,
) -> dict:
    D = q_init.shape[0]
    N = cfg.N
    dtype = fk.dtype
    device = fk.device

    # Initial trajectory (min-jerk)
    q_t = torch.tensor(minjerk_interp(q_init, q_goal, N), dtype=dtype, device=device)
    q_start = q_t[0:1]; q_end = q_t[-1:]
    xi = q_t[1:-1].clone()                                  # (N-2, D)

    # Build noise covariance from smoothness metric: Σ = (M + εI)^-1
    M = make_smoothness_metric(N, dtype=dtype, device=device)        # (N-2, N-2)
    Sigma = torch.linalg.inv(M)                                       # (N-2, N-2)
    # Cholesky for sampling: noise = L @ z, z ~ N(0, I)
    # Σ is symmetric positive definite (M was)
    L = torch.linalg.cholesky(Sigma + 1e-6 * torch.eye(N - 2, dtype=dtype, device=device))

    def eval_cost(traj: torch.Tensor) -> torch.Tensor:
        sd = traj[2:] - 2 * traj[1:-1] + traj[:-2]
        J_s = (sd ** 2).sum()
        J_sp = torch.tensor(0.0, dtype=dtype, device=device)
        if spill is not None and cfg.gamma_spill > 0:
            J_sp = spill.cost(traj)
        return cfg.alpha_smooth * J_s + cfg.gamma_spill * J_sp

    sigma = cfg.sigma_init
    loss_hist = []
    for it in range(cfg.n_iters):
        # Sample K noisy perturbations of xi
        z = torch.randn(cfg.K, N - 2, D, dtype=dtype, device=device) * sigma
        # Apply smoothness correlation per-dim: noise = L @ z (over time axis)
        noise = torch.einsum("ij,kjd->kid", L, z)            # (K, N-2, D)

        costs = torch.zeros(cfg.K, dtype=dtype, device=device)
        with torch.no_grad():
            for k in range(cfg.K):
                xi_k = xi + noise[k]
                traj_k = torch.cat([q_start, xi_k, q_end], dim=0)
                costs[k] = eval_cost(traj_k)

        # Softmin importance weights
        c_min = costs.min()
        w = torch.exp(-(costs - c_min) / cfg.lam)
        w = w / w.sum()                                       # (K,)

        # Weighted noise → update
        delta = torch.einsum("k,kid->id", w, noise)           # (N-2, D)
        xi = xi + delta

        with torch.no_grad():
            traj_final = torch.cat([q_start, xi, q_end], dim=0)
            J_cur = eval_cost(traj_final).item()
        loss_hist.append(dict(J=J_cur, sigma=sigma, best_cost=c_min.item()))
        if cfg.verbose and (it % 10 == 0 or it == cfg.n_iters - 1):
            print(f"  stomp[{it:3d}]  J={J_cur:.4e}  best_sample={c_min.item():.4e}  σ={sigma:.4f}")

        sigma *= cfg.sigma_decay

    out = dict(q_traj=traj_final.detach().cpu().numpy(),
               loss_hist=loss_hist,
               final_loss=loss_hist[-1])
    if spill is not None:
        out["spill_diag"] = spill.diagnostics(traj_final)
    return out
