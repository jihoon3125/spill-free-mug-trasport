"""Differentiable IK on top of URDFForwardKinematics.

Adam-based gradient descent on a pose loss:
    L(q) = ||p(q) − p_target||² + λ · (3 − tr(R(q) R_target^T))

Robust enough for our use (single-arm 6-DoF, single target).
"""
from __future__ import annotations
import torch
from .kinematics import URDFForwardKinematics


def ik_solve(
    fk: URDFForwardKinematics,
    q_init: torch.Tensor,
    T_target: torch.Tensor,
    n_iters: int = 800,
    lr: float = 0.05,
    lambda_rot: float = 0.5,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-4,
    verbose: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Return q_opt and an info dict (final loss / errors).

    q_init: (n_active,) torch tensor (will be cloned and detached).
    T_target: (4,4) torch tensor.
    """
    q = q_init.clone().detach().to(dtype=fk.dtype, device=fk.device).requires_grad_(True)
    T_target = T_target.to(dtype=fk.dtype, device=fk.device)
    opt = torch.optim.Adam([q], lr=lr)
    last_pos = last_rot = None
    for it in range(n_iters):
        opt.zero_grad()
        T = fk(q)
        e_p = T[:3, 3] - T_target[:3, 3]
        loss_p = (e_p ** 2).sum()
        # 3 − trace(R @ R_target^T)
        loss_r = 3.0 - torch.einsum("ij,ij->", T[:3, :3], T_target[:3, :3])
        loss = loss_p + lambda_rot * loss_r
        loss.backward()
        opt.step()
        last_pos = loss_p.item() ** 0.5
        last_rot = loss_r.item()
        if verbose and it % 100 == 0:
            print(f"  ik[{it:4d}]  pos_err={last_pos:.5f}  rot_err={last_rot:.5f}")
        if last_pos < tol_pos and last_rot < tol_rot:
            break
    info = dict(iters=it + 1, pos_err=last_pos, rot_err=last_rot)
    return q.detach(), info
