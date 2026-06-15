"""3D mug trajectory figure for the report.

Loads results/ablation.npz and draws, for naive (min-jerk) vs ours (CHOMP+spill):
  - the mug center-of-mass path A -> B in 3D
  - each waypoint colored by EFFECTIVE tilt (g - a basis), not raw orientation,
    so "upright but accelerating => spilling" is visible
  - mug up-vector arrows (quiver) at sampled waypoints, red where spilling
  - start / goal markers

Light compute: one FK pass over the saved trajectories (no optimization).

Output: figs/fig_traj_3d.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp.kinematics import URDFForwardKinematics
from sim.scene import URDF
from mp.constants import MUG_UP_AXIS_LOCAL

PANELS = [("minjerk", "Naive (min-jerk)"), ("chomp_spill", "Ours (CHOMP + spill)")]


def up_vectors(fk, q_traj, T_em):
    """Mug up-vector in world frame at each waypoint."""
    q = torch.tensor(q_traj, dtype=torch.float64)
    T = fk(q).detach().numpy()                       # (N,4,4)
    R_em = T_em[:3, :3]
    up_local = np.asarray(MUG_UP_AXIS_LOCAL)
    up_w = np.einsum("nij,jk,k->ni", T[:, :3, :3], R_em, up_local)
    return up_w / np.linalg.norm(up_w, axis=1, keepdims=True)


def main():
    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d["theta_max_deg"])
    T_em = d["T_ee_to_mug"]
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    vmax = max(d[f"{k}_tilt"].max() for k, _ in PANELS)
    norm = Normalize(0, vmax)
    cmap = plt.get_cmap("RdYlBu_r")

    fig = plt.figure(figsize=(13, 5.5))
    for i, (key, title) in enumerate(PANELS):
        p = d[f"{key}_pmug"]                          # (N,3)
        tilt = d[f"{key}_tilt"]                       # (N,)
        up_w = up_vectors(fk, d[f"{key}_q"], T_em)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # path colored by effective tilt
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=tilt, cmap=cmap, norm=norm, s=16, depthshade=False)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="gray", lw=0.8, alpha=0.5)
        # up-vector arrows at sampled waypoints
        step = max(1, len(p) // 9)
        L = 0.06
        for j in range(0, len(p), step):
            spill = tilt[j] > theta_max
            ax.quiver(p[j, 0], p[j, 1], p[j, 2],
                      up_w[j, 0] * L, up_w[j, 1] * L, up_w[j, 2] * L,
                      color="red" if spill else "green", lw=1.6, arrow_length_ratio=0.3)
        ax.scatter(*p[0], c="black", marker="o", s=60)
        ax.text(*p[0], "  start", fontsize=8)
        ax.scatter(*p[-1], c="black", marker="*", s=110)
        ax.text(*p[-1], "  goal", fontsize=8)
        n_spill = int((tilt > theta_max).sum())
        ax.set_title(f"{title}\nspill {n_spill}/{len(p)} wp, max tilt {tilt.max():.0f}°", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=18, azim=-60)

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.025, pad=0.04)
    cb.set_label(f"effective tilt (°)  —  spill threshold {theta_max:.0f}°")
    cb.ax.axhline(theta_max / norm.vmax, color="black", lw=1.2)
    fig.suptitle("Mug trajectory in 3D — arrows = cup up-vector (green safe / red spilling)",
                 fontsize=12)
    out = ROOT / "figs/fig_traj_3d.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
