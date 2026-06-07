"""Joint-space CHOMP trajectory optimization with optional spill cost.

Trajectory parameterization:
  ξ = (q_0, q_1, ..., q_{N-1}) ∈ R^{N × D}     waypoints
  q_0 and q_{N-1} are fixed (start and goal)
  the free variable is the inner waypoints ξ_inner = (q_1, ..., q_{N-2})

Cost:
  J(ξ) = α · J_smooth(ξ) + β · J_obs(ξ) + γ · J_spill(ξ)
       (J_obs is left as a hook — disabled by default in this 2-day project)

CHOMP update:
  ξ ← ξ − η · M^{-1} · ∇J(ξ)
  where M is the discrete acceleration metric:  M = A^T A,
  A is the (N-2) × N finite-difference 2nd-derivative operator (in time).

For the inner waypoints we restrict M^{-1} to the inner block, computed once.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import math
import numpy as np
import torch
from .kinematics import URDFForwardKinematics
from .spill_cost import SpillCost


def make_smoothness_metric(N: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Build the (N-2)x(N-2) preconditioner for inner waypoints.

    A is (N-2) × N second-difference: row t (for t in 0..N-3) maps to
        ξ[t] − 2 ξ[t+1] + ξ[t+2].
    We optimize only the inner waypoints (indices 1..N-2). So we split A into
    [A_left | A_inner | A_right] and use M_inner = A_inner^T A_inner as the
    smoothness metric on the inner block.
    """
    A = torch.zeros(N - 2, N, dtype=dtype, device=device)
    for t in range(N - 2):
        A[t, t] = 1.0; A[t, t + 1] = -2.0; A[t, t + 2] = 1.0
    A_inner = A[:, 1:-1]                    # (N-2, N-2)
    M = A_inner.T @ A_inner                 # (N-2, N-2)
    # Add small regularization for numerical stability (CHOMP paper does this)
    M = M + 1e-3 * torch.eye(N - 2, dtype=dtype, device=device)
    return M


@dataclass
class CHOMPConfig:
    N: int = 60                             # number of waypoints
    dt: float = 0.05                        # time step (s)  → T_total = (N-1)*dt
    n_iters: int = 300
    step_size: float = 0.5
    alpha_smooth: float = 1.0
    beta_obs: float = 0.0                   # off by default
    gamma_spill: float = 50.0
    use_covariant: bool = True              # CHOMP M^-1 preconditioning
    verbose: bool = False


def linear_interp(q_init: np.ndarray, q_goal: np.ndarray, N: int) -> np.ndarray:
    """Constant-velocity linear interp (rest-not-respected)."""
    alphas = np.linspace(0.0, 1.0, N)[:, None]
    return (1 - alphas) * q_init[None, :] + alphas * q_goal[None, :]


def minjerk_interp(q_init: np.ndarray, q_goal: np.ndarray, N: int) -> np.ndarray:
    """Quintic minimum-jerk profile, rest-to-rest. Smooth velocity + accel,
    nontrivial accel in the interior — a realistic 'naive' baseline."""
    tau = np.linspace(0.0, 1.0, N)[:, None]
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5    # ∈ [0,1], with s'(0)=s'(1)=0, s''(0)=s''(1)=0
    return (1 - s) * q_init[None, :] + s * q_goal[None, :]


def chomp_optimize(
    fk: URDFForwardKinematics,
    q_init: np.ndarray,
    q_goal: np.ndarray,
    cfg: CHOMPConfig,
    spill: SpillCost | None = None,
    obs_cost_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    q_init_traj: np.ndarray | None = None,
) -> dict:
    """Return dict with q_traj (N, D) numpy + per-iter loss + final diagnostics."""
    D = q_init.shape[0]
    N = cfg.N
    dtype = fk.dtype
    device = fk.device

    # Initial trajectory — minjerk (smooth, rest-to-rest) unless overridden
    if q_init_traj is None:
        q_init_traj = minjerk_interp(q_init, q_goal, N)
    q_init_t = torch.tensor(q_init_traj, dtype=dtype, device=device)
    xi_inner = q_init_t[1:-1].clone().detach().requires_grad_(True)        # (N-2, D)
    q_start = q_init_t[0:1].detach()
    q_end = q_init_t[-1:].detach()

    M = make_smoothness_metric(N, dtype=dtype, device=device)
    M_inv = torch.linalg.inv(M)

    # Adam-based update (CHOMP covariant step inside the gradient when use_covariant=True)
    opt = torch.optim.Adam([xi_inner], lr=cfg.step_size)

    loss_hist = []
    for it in range(cfg.n_iters):
        opt.zero_grad()
        q_traj = torch.cat([q_start, xi_inner, q_end], dim=0)               # (N, D)

        sd = q_traj[2:] - 2 * q_traj[1:-1] + q_traj[:-2]
        J_smooth = (sd ** 2).sum()

        J_obs = torch.tensor(0.0, dtype=dtype, device=device)
        if obs_cost_fn is not None and cfg.beta_obs > 0:
            J_obs = obs_cost_fn(q_traj)

        J_spill = torch.tensor(0.0, dtype=dtype, device=device)
        if spill is not None and cfg.gamma_spill > 0:
            J_spill = spill.cost(q_traj)

        J = cfg.alpha_smooth * J_smooth + cfg.beta_obs * J_obs + cfg.gamma_spill * J_spill
        J.backward()

        if cfg.use_covariant:
            with torch.no_grad():
                # Apply M^-1 preconditioning to the gradient before Adam sees it.
                xi_inner.grad = M_inv @ xi_inner.grad

        # Clip gradient norm to avoid divergence on hard scenarios
        torch.nn.utils.clip_grad_norm_([xi_inner], max_norm=10.0)

        opt.step()

        if cfg.verbose and (it % 50 == 0 or it == cfg.n_iters - 1):
            print(f"  chomp[{it:4d}]  J={J.item():.4e}  J_smooth={J_smooth.item():.4e}  "
                  f"J_spill={J_spill.item():.4e}  J_obs={J_obs.item():.4e}")
        loss_hist.append(dict(J=J.item(), J_smooth=J_smooth.item(),
                              J_spill=J_spill.item(), J_obs=J_obs.item()))

    with torch.no_grad():
        q_final = torch.cat([q_start, xi_inner, q_end], dim=0)
    out = dict(q_traj=q_final.detach().cpu().numpy(),
               loss_hist=loss_hist,
               final_loss=loss_hist[-1])
    if spill is not None:
        out["spill_diag"] = spill.diagnostics(q_final)
    return out


def run_baseline_linear(q_init: np.ndarray, q_goal: np.ndarray, N: int) -> np.ndarray:
    return linear_interp(q_init, q_goal, N)
