"""Obstacle cost: SDF for a single axis-aligned box, plus a sphere-based robot
collision model. Differentiable in PyTorch (works as a CHOMP cost term).

We approximate the UR5e + xhand by N collision spheres on the wrist/hand links
(simplified — we just check the EE 'base' link and a few wrist links).

For our simple demo:
  - obstacle = axis-aligned box at world (cx, cy, cz) with half-extents (hx, hy, hz)
  - SDF box: d(p) = sphere_dist_to_box(p)
  - cost per sphere: max(0, margin − d(p))²
  - sum over collision spheres + over trajectory waypoints

The mug is included as one extra sphere (centered at mug center) since we want
the mug to also avoid the obstacle.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import torch
from .kinematics import URDFForwardKinematics


@dataclass
class ObstacleConfig:
    box_center: tuple = (0.5, 0.05, 0.30)
    box_halfext: tuple = (0.05, 0.04, 0.10)
    margin: float = 0.04
    T_ee_to_mug: np.ndarray = None
    mug_radius: float = 0.07          # bounding-sphere radius for mug
    wrist_link_offsets: Sequence[tuple] = field(default_factory=lambda: [
        # In EE/base frame: offsets to check for arm self
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -0.05),
        (0.0, 0.0, -0.10),
    ])
    wrist_radius: float = 0.05


class ObstacleCost:
    def __init__(self, cfg: ObstacleConfig, fk: URDFForwardKinematics):
        self.cfg = cfg
        self.fk = fk
        self.dtype = fk.dtype; self.device = fk.device
        self.box_c = torch.as_tensor(cfg.box_center, dtype=self.dtype, device=self.device)
        self.box_h = torch.as_tensor(cfg.box_halfext, dtype=self.dtype, device=self.device)
        if cfg.T_ee_to_mug is not None:
            self.T_em = torch.as_tensor(cfg.T_ee_to_mug, dtype=self.dtype, device=self.device)
        else:
            self.T_em = None
        self.wrist_offsets = torch.tensor(cfg.wrist_link_offsets,
                                           dtype=self.dtype, device=self.device)  # (M, 3)

    def sdf_box(self, p: torch.Tensor) -> torch.Tensor:
        """Signed distance to axis-aligned box. p: (..., 3) → (...)."""
        q = (p - self.box_c).abs() - self.box_h          # (..., 3)
        outside = torch.clamp(q, min=0.0).norm(dim=-1)
        inside = torch.clamp(q.max(dim=-1).values, max=0.0)
        return outside + inside                           # signed: <0 inside, >0 outside

    def cost(self, q: torch.Tensor) -> torch.Tensor:
        """q: (N, dof). Returns scalar cost summed over waypoints + collision points."""
        T = self.fk(q)                                   # (N, 4, 4)
        R_ee = T[:, :3, :3]; t_ee = T[:, :3, 3]          # (N,3,3), (N,3)

        total = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # 1) wrist / hand collision spheres (offsets in EE frame)
        for off in self.wrist_offsets:
            p_w = torch.einsum("nij,j->ni", R_ee, off) + t_ee   # (N, 3)
            d = self.sdf_box(p_w) - self.cfg.wrist_radius
            total = total + torch.clamp(self.cfg.margin - d, min=0.0).pow(2).sum()

        # 2) mug-bounding sphere
        if self.T_em is not None:
            t_em = self.T_em[:3, 3]
            p_mug = torch.einsum("nij,j->ni", R_ee, t_em) + t_ee
            d = self.sdf_box(p_mug) - self.cfg.mug_radius
            total = total + torch.clamp(self.cfg.margin - d, min=0.0).pow(2).sum()
        return total

    def diagnostics(self, q: torch.Tensor) -> dict:
        with torch.no_grad():
            T = self.fk(q)
            R_ee = T[:, :3, :3]; t_ee = T[:, :3, 3]
            t_em = self.T_em[:3, 3] if self.T_em is not None else None
            mug_pos = torch.einsum("nij,j->ni", R_ee, t_em) + t_ee if t_em is not None else None
            d = self.sdf_box(mug_pos) - self.cfg.mug_radius if mug_pos is not None else None
            n_violations = int((d < 0).sum().item()) if d is not None else 0
            return dict(min_clearance=float(d.min().item()) if d is not None else None,
                        mean_clearance=float(d.mean().item()) if d is not None else None,
                        n_violations=n_violations,
                        violation_ratio=n_violations / q.shape[0])
