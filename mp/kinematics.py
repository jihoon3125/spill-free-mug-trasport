"""URDF-based forward kinematics in PyTorch (autograd-friendly).

Parses a URDF, builds the kinematic chain from base_link to ee_link, and
provides a differentiable FK on a batch of joint configurations.

Used by:
  - spill cost (need EE rotation as a function of q)
  - obstacle cost (need link transforms for collision spheres)
  - IK (gradient through FK)

For our project: chain base_link → ... → wrist_3_link → arm_hand_joint → base
(xhand base treated as "EE" — what the planning sim attaches mug to).
"""
from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Sequence
from pathlib import Path
import torch


@dataclass
class JointInfo:
    name: str
    type: str        # revolute | prismatic | fixed | continuous
    parent: str
    child: str
    origin_xyz: list  # (3,)
    origin_rpy: list  # (3,)
    axis: list       # (3,) — meaningful only if type in {revolute, prismatic, continuous}


def _parse_urdf(urdf_path: Path) -> dict[str, JointInfo]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out = {}
    for j in root.findall("joint"):
        name = j.get("name")
        jtype = j.get("type")
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        if origin is not None:
            xyz = list(map(float, (origin.get("xyz") or "0 0 0").split()))
            rpy = list(map(float, (origin.get("rpy") or "0 0 0").split()))
        else:
            xyz, rpy = [0, 0, 0], [0, 0, 0]
        axis_el = j.find("axis")
        axis = list(map(float, ((axis_el.get("xyz") if axis_el is not None else "1 0 0") or "1 0 0").split()))
        out[name] = JointInfo(name, jtype, parent, child, xyz, rpy, axis)
    return out


def build_chain(urdf_path: Path, base_link: str, ee_link: str) -> list[JointInfo]:
    """Return list of JointInfo along the chain base_link → ee_link (in order)."""
    joints = _parse_urdf(urdf_path)
    # parent map: child_link → (joint_name, parent_link)
    parent_of = {j.child: j.name for j in joints.values()}
    # walk back from ee_link to base_link
    chain_rev = []
    cur = ee_link
    while cur != base_link:
        if cur not in parent_of:
            raise ValueError(f"link {cur!r} has no parent joint (chain not reachable from {base_link!r})")
        jname = parent_of[cur]
        j = joints[jname]
        chain_rev.append(j)
        cur = j.parent
    return list(reversed(chain_rev))


def rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """rpy: (..., 3) → R: (..., 3, 3). Roll-pitch-yaw = X-Y-Z intrinsic."""
    r, p, y = rpy.unbind(-1)
    cr, sr = torch.cos(r), torch.sin(r)
    cp, sp = torch.cos(p), torch.sin(p)
    cy, sy = torch.cos(y), torch.sin(y)
    zero = torch.zeros_like(r)
    one = torch.ones_like(r)
    Rx = torch.stack([
        torch.stack([one, zero, zero], -1),
        torch.stack([zero, cr, -sr], -1),
        torch.stack([zero, sr, cr], -1)], -2)
    Ry = torch.stack([
        torch.stack([cp, zero, sp], -1),
        torch.stack([zero, one, zero], -1),
        torch.stack([-sp, zero, cp], -1)], -2)
    Rz = torch.stack([
        torch.stack([cy, -sy, zero], -1),
        torch.stack([sy, cy, zero], -1),
        torch.stack([zero, zero, one], -1)], -2)
    return Rz @ Ry @ Rx


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis: (..., 3) unit, angle: (...) → R: (..., 3, 3)."""
    a = axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)
    ax, ay, az = a.unbind(-1)
    c = torch.cos(angle); s = torch.sin(angle); C = 1 - c
    R00 = c + ax*ax*C; R01 = ax*ay*C - az*s; R02 = ax*az*C + ay*s
    R10 = ay*ax*C + az*s; R11 = c + ay*ay*C; R12 = ay*az*C - ax*s
    R20 = az*ax*C - ay*s; R21 = az*ay*C + ax*s; R22 = c + az*az*C
    R = torch.stack([
        torch.stack([R00, R01, R02], -1),
        torch.stack([R10, R11, R12], -1),
        torch.stack([R20, R21, R22], -1)], -2)
    return R


def make_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """R: (...,3,3), t: (...,3) → T: (...,4,4) homogeneous."""
    *batch, _, _ = R.shape
    T = torch.zeros(*batch, 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


class URDFForwardKinematics:
    """Differentiable FK along a single URDF chain.

    Each call:
      input q of shape (B, n_active_dof) → output T of shape (B, 4, 4)
      (or (4,4) if input is (n_active_dof,))
    """

    def __init__(self, urdf_path: str | Path, base_link: str, ee_link: str,
                 dtype=torch.float64, device="cpu"):
        self.chain = build_chain(Path(urdf_path), base_link, ee_link)
        self.dtype = dtype
        self.device = device
        # Precompute fixed origin transforms (constant) and per-joint info
        self._origins_R: list[torch.Tensor] = []
        self._origins_t: list[torch.Tensor] = []
        self._axes: list[torch.Tensor] = []
        self._is_active: list[bool] = []
        for j in self.chain:
            rpy = torch.tensor(j.origin_rpy, dtype=dtype, device=device)
            xyz = torch.tensor(j.origin_xyz, dtype=dtype, device=device)
            R = rpy_to_matrix(rpy)
            self._origins_R.append(R)
            self._origins_t.append(xyz)
            self._axes.append(torch.tensor(j.axis, dtype=dtype, device=device))
            self._is_active.append(j.type in {"revolute", "prismatic", "continuous"})
        self.n_active = sum(self._is_active)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute T_base_to_ee for each row of q."""
        if q.dim() == 1:
            return self.forward(q.unsqueeze(0)).squeeze(0)
        B = q.shape[0]
        assert q.shape[1] == self.n_active, f"expected {self.n_active} active dof, got {q.shape[1]}"
        eye_R = torch.eye(3, dtype=self.dtype, device=self.device).expand(B, 3, 3).clone()
        eye_t = torch.zeros(B, 3, dtype=self.dtype, device=self.device)
        T = make_T(eye_R, eye_t)
        q_idx = 0
        for k, j in enumerate(self.chain):
            R0 = self._origins_R[k].expand(B, 3, 3)
            t0 = self._origins_t[k].expand(B, 3)
            T_origin = make_T(R0, t0)
            if self._is_active[k]:
                ang = q[:, q_idx]
                q_idx += 1
                if j.type == "prismatic":
                    R_j = torch.eye(3, dtype=self.dtype, device=self.device).expand(B, 3, 3)
                    t_j = self._axes[k].expand(B, 3) * ang.unsqueeze(-1)
                    T_joint = make_T(R_j, t_j)
                else:  # revolute or continuous
                    axis_b = self._axes[k].expand(B, 3)
                    R_j = axis_angle_to_matrix(axis_b, ang)
                    T_joint = make_T(R_j, torch.zeros(B, 3, dtype=self.dtype, device=self.device))
                T = T @ T_origin @ T_joint
            else:  # fixed
                T = T @ T_origin
        return T

    __call__ = forward
