"""Spill cost: quasi-static rim cone constraint on the mug.

For each trajectory waypoint q_t, we compute:
    p_mug(t)      = T_world_ee(q_t) · T_ee_to_mug · [0,0,0,1]   ; mug center
    mug_up_w(t)   = T_world_ee(q_t)[:3,:3] · T_ee_to_mug[:3,:3] · MUG_UP_LOCAL
    a_mug(t)      = (p_mug(t+1) − 2 p_mug(t) + p_mug(t−1)) / dt²
    g_eff(t)      = g_world − a_mug(t)              ; effective gravity in EE/world frame
    align(t)      = −normalize(g_eff(t)) · mug_up_w(t)

`align(t) = 1` when the effective gravity points straight into the mug opening
(no tilt, no horizontal accel). `align(t) = cos(θ)` for an effective tilt of θ.

Spill condition:  align(t) < cos(θ_max)
Penalty:          c_spill(t) = max(0, cos(θ_max) − align(t))²

Returned cost is the sum over waypoints (no normalization — units are radians²
roughly, depending on dt and trajectory length).
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from .kinematics import URDFForwardKinematics


@dataclass
class SpillConfig:
    T_ee_to_mug: np.ndarray            # (4,4)
    mug_up_local: np.ndarray = None    # default = [0,1,0] (mesh +y is up)
    dt: float = 0.05                   # seconds per waypoint
    theta_max_deg: float = 25.0        # scalar OR np.ndarray (N,) for time-varying
    g: float = 9.81

    def __post_init__(self):
        if self.mug_up_local is None:
            self.mug_up_local = np.array([0.0, 1.0, 0.0])


class SpillCost:
    def __init__(self, cfg: SpillConfig, fk: URDFForwardKinematics):
        self.cfg = cfg
        self.fk = fk
        self.dtype = fk.dtype
        self.device = fk.device
        self.T_em = torch.as_tensor(cfg.T_ee_to_mug, dtype=self.dtype, device=self.device)
        self.up_local = torch.as_tensor(cfg.mug_up_local, dtype=self.dtype, device=self.device)
        self.g_world = torch.tensor([0.0, 0.0, -cfg.g], dtype=self.dtype, device=self.device)
        # Support scalar OR per-waypoint theta_max
        if np.isscalar(cfg.theta_max_deg):
            self.cos_thmax = torch.tensor(math.cos(math.radians(cfg.theta_max_deg)),
                                           dtype=self.dtype, device=self.device)
            self.theta_max_per_wp = None
        else:
            arr = np.asarray(cfg.theta_max_deg, dtype=float)
            self.cos_thmax = torch.tensor(np.cos(np.radians(arr)),
                                           dtype=self.dtype, device=self.device)
            self.theta_max_per_wp = arr
        self.dt = cfg.dt

    def _per_waypoint(self, q: torch.Tensor):
        """q: (N, n_dof) → (mug_pos: (N,3), mug_up: (N,3))."""
        T = self.fk(q)                                  # (N, 4, 4)
        R_ee = T[:, :3, 3 - 3:3]                        # (N, 3, 3)  same as T[:, :3, :3]
        R_ee = T[:, :3, :3]
        t_ee = T[:, :3, 3]                              # (N, 3)
        # mug center in world = R_ee @ t_em + t_ee
        t_em = self.T_em[:3, 3]
        R_em = self.T_em[:3, :3]
        p_mug = (R_ee @ t_em).squeeze(-1) + t_ee \
            if t_em.dim() == 2 else (R_ee @ t_em) + t_ee
        # Simpler: matrix-vector
        p_mug = torch.einsum("nij,j->ni", R_ee, t_em) + t_ee
        up_w = torch.einsum("nij,jk,k->ni", R_ee, R_em, self.up_local)
        # normalize up just in case (should be unit anyway)
        up_w = up_w / (up_w.norm(dim=-1, keepdim=True) + 1e-12)
        return p_mug, up_w

    def cost(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, n_dof). Returns scalar cost = sum_t penalty(t)."""
        p_mug, up_w = self._per_waypoint(q)
        N = q.shape[0]
        # central finite difference acceleration; pad endpoints with same a as nearest interior
        a = torch.zeros_like(p_mug)
        if N >= 3:
            a[1:-1] = (p_mug[2:] - 2 * p_mug[1:-1] + p_mug[:-2]) / (self.dt ** 2)
            a[0] = a[1]; a[-1] = a[-2]
        g_eff = self.g_world.unsqueeze(0) - a                       # (N, 3)
        g_hat = g_eff / (g_eff.norm(dim=-1, keepdim=True) + 1e-9)
        align = -(g_hat * up_w).sum(dim=-1)                          # (N,)
        # cos_thmax is either scalar tensor or (N,) tensor
        cos_thmax = self.cos_thmax if self.cos_thmax.dim() == 0 else self.cos_thmax[:align.shape[0]]
        penalty = torch.clamp(cos_thmax - align, min=0.0) ** 2  # (N,)
        return penalty.sum()

    def diagnostics(self, q: torch.Tensor) -> dict:
        """Per-waypoint tilt angles + spill flag. For visualization / metric."""
        with torch.no_grad():
            p_mug, up_w = self._per_waypoint(q)
            N = q.shape[0]
            a = torch.zeros_like(p_mug)
            if N >= 3:
                a[1:-1] = (p_mug[2:] - 2 * p_mug[1:-1] + p_mug[:-2]) / (self.dt ** 2)
                a[0] = a[1]; a[-1] = a[-2]
            g_eff = self.g_world.unsqueeze(0) - a
            g_hat = g_eff / (g_eff.norm(dim=-1, keepdim=True) + 1e-9)
            align = -(g_hat * up_w).sum(dim=-1)
            theta = torch.acos(align.clamp(-1, 1))
            if np.isscalar(self.cfg.theta_max_deg):
                thmax_arr = np.full(theta.shape[0], self.cfg.theta_max_deg)
            else:
                thmax_arr = np.asarray(self.cfg.theta_max_deg, dtype=float)[:theta.shape[0]]
            thmax_rad = torch.tensor(np.radians(thmax_arr), dtype=self.dtype, device=self.device)
            spill_flag = (theta > thmax_rad)
            return dict(
                p_mug=p_mug.cpu().numpy(),
                up_w=up_w.cpu().numpy(),
                accel=a.cpu().numpy(),
                tilt_rad=theta.cpu().numpy(),
                tilt_deg=torch.rad2deg(theta).cpu().numpy(),
                spill_flag=spill_flag.cpu().numpy(),
                spill_ratio=float(spill_flag.float().mean()),
                theta_max_per_wp=thmax_arr,
            )
