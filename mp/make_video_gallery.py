"""Capability gallery: ours (CHOMP+spill) generates spill-free transport across
diverse tasks. One grasp, several start→goal scenarios, 2x2 grid.

Message: across varied motions (lateral, lift, diagonal, reach-across) our planner
keeps the water in — slosh is visible but stays below the rim.

Usage: python mp/make_video_gallery.py
Output: figs/video_gallery.gif
"""
from __future__ import annotations
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
from mp.chomp import CHOMPConfig, chomp_optimize
from mp.make_video_transport import build_scene, simulate
import PIL.Image

SCENARIOS = [
    ("Long lateral", (0.50, -0.30, 0.25), (0.50, 0.40, 0.40)),
    ("Vertical lift", (0.50, 0.05, 0.20), (0.50, 0.05, 0.48)),
    ("Diagonal swing", (0.45, -0.25, 0.25), (0.60, 0.30, 0.45)),
    ("Reach across", (0.60, 0.05, 0.32), (0.38, 0.35, 0.28)),
]
T_VID = 0.8
N_WP = 40
THETA_MAX = 18.0


def upright(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float); return T


def ik(fk, T_em, p, seed):
    T_ee = torch.tensor(upright(p) @ np.linalg.inv(T_em), dtype=torch.float64)
    q, info = ik_solve(fk, torch.tensor(seed, dtype=torch.float64), T_ee,
                       n_iters=2500, lr=0.03)
    return q.numpy(), info


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    dt = T_VID / (N_WP - 1)
    scene, robot, mug, particles, cam, sapien_order = build_scene(g)

    panels = []
    for label, ps, pg in SCENARIOS:
        q_s, i_s = ik(fk, T_em, ps, g['qpos'][:6])
        q_g, i_g = ik(fk, T_em, pg, q_s)
        if i_s['pos_err'] > 0.012 or i_g['pos_err'] > 0.012:
            print(f"[skip] {label}: IK err {i_s['pos_err']:.3f}/{i_g['pos_err']:.3f}", flush=True)
            continue
        sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA_MAX, dt=dt), fk)
        res = chomp_optimize(fk, q_s, q_g, CHOMPConfig(N=N_WP, dt=dt, n_iters=1200,
                             gamma_spill=5.0, step_size=0.003), spill=sp)
        tilt = res["spill_diag"]["tilt_deg"]
        print(f"[ok] {label}: max tilt {tilt.max():.0f}deg  spill {(tilt>THETA_MAX).sum()}/{N_WP}",
              flush=True)
        frames = simulate(scene, robot, mug, particles, cam, sapien_order,
                          res["q_traj"], xhand, T_em, fk, tilt, THETA_MAX, label, dt)
        panels.append(frames)

    N = min(len(p) for p in panels)
    # 2x2 grid (pad to 4 by repeating last if needed)
    while len(panels) < 4:
        panels.append(panels[-1])
    grid = []
    for i in range(N):
        top = np.concatenate([panels[0][i], panels[1][i]], axis=1)
        bot = np.concatenate([panels[2][i], panels[3][i]], axis=1)
        grid.append(np.concatenate([top, bot], axis=0))
    out = ROOT / "figs/video_gallery.gif"
    pil = [PIL.Image.fromarray(f) for f in grid]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=80, loop=0)
    print(f"saved {out}  ({N} frames)", flush=True)


if __name__ == "__main__":
    main()
