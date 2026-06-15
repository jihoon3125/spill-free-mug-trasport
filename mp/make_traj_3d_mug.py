"""3D mug-trajectory figure WITH the actual mug mesh drawn along the path.

Like make_traj_3d.py, but at sampled waypoints it renders the real mug mesh
(data/mug_mesh.obj) at the cup's true world pose, face-colored green (safe) or
red (spilling). This makes it read as a cup being carried & tilting along A->B.

Light compute: one FK pass per method (no optimization).

Usage:
  python mp/make_traj_3d_mug.py [--mug-scale 1.0] [--n-mugs 5]
Output: figs/fig_traj_3d_mug.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp.kinematics import URDFForwardKinematics
from sim.scene import URDF

PANELS = [("minjerk", "Naive (min-jerk)"), ("chomp_spill", "Ours (CHOMP + spill)")]
MUG_MESH_SCALE = 0.12          # grasp_info Object Scale


def mug_world_transforms(fk, q_traj, T_em):
    q = torch.tensor(q_traj, dtype=torch.float64)
    T_ee = fk(q).detach().numpy()                    # (N,4,4)
    return T_ee @ T_em                               # (N,4,4) world<-mug


def set_equal_3d(ax, pts):
    mid = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2
    ax.set_xlim(mid[0] - r, mid[0] + r)
    ax.set_ylim(mid[1] - r, mid[1] + r)
    ax.set_zlim(mid[2] - r, mid[2] + r)
    ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mug-scale", type=float, default=1.0, help="visual size multiplier")
    ap.add_argument("--n-mugs", type=int, default=5)
    args = ap.parse_args()

    d = np.load(ROOT / "results/ablation.npz")
    theta_max = float(d["theta_max_deg"])
    T_em = d["T_ee_to_mug"]
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")

    mesh = trimesh.load(ROOT / "data/mug_mesh.obj", force="mesh")
    # Convention is now correct (R_UPRIGHT maps mesh -y opening -> world up), so the
    # mug renders right-side up directly under the trajectory pose — no flip needed.
    V = np.asarray(mesh.vertices) * MUG_MESH_SCALE * args.mug_scale  # (Nv,3)
    F = np.asarray(mesh.faces)

    fig = plt.figure(figsize=(13, 5.5))
    for i, (key, title) in enumerate(PANELS):
        p = d[f"{key}_pmug"]; tilt = d[f"{key}_tilt"]
        Tw = mug_world_transforms(fk, d[f"{key}_q"], T_em)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        # trajectory: dark guide line + waypoints colored by effective tilt
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#333333", lw=1.4, alpha=0.9, zorder=5)
        sc = ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=tilt, cmap="RdYlBu_r",
                        vmin=0, vmax=max(d['minjerk_tilt'].max(), d['chomp_spill_tilt'].max()),
                        s=14, depthshade=False, zorder=6)

        all_pts = [p]
        samples = np.linspace(0, len(p) - 1, args.n_mugs).round().astype(int)
        for j in samples:
            R = Tw[j, :3, :3]; t = Tw[j, :3, 3]
            Vw = V @ R.T + t                          # mesh verts -> world
            all_pts.append(Vw)
            spill = tilt[j] > theta_max
            tris = Vw[F]
            col = Poly3DCollection(tris, alpha=0.55, linewidths=0.1)
            col.set_facecolor("#d62728" if spill else "#2ca02c")
            col.set_edgecolor((0, 0, 0, 0.15))
            ax.add_collection3d(col)
        ax.scatter(*p[0], c="black", marker="o", s=50)
        ax.text(*p[0], "  start", fontsize=8)
        ax.scatter(*p[-1], c="black", marker="*", s=110)
        ax.text(*p[-1], "  goal", fontsize=8)

        set_equal_3d(ax, np.vstack(all_pts))
        n_spill = int((tilt > theta_max).sum())
        ax.set_title(f"{title}\nspill {n_spill}/{len(p)} wp, max tilt {tilt.max():.0f}°", fontsize=10)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=16, azim=-72)

    fig.suptitle("Mug carried along the trajectory  (green = safe, red = spilling)",
                 fontsize=12)
    out = ROOT / "figs/fig_traj_3d_mug.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (n_mugs={args.n_mugs}, mug_scale x{args.mug_scale})")


if __name__ == "__main__":
    main()
