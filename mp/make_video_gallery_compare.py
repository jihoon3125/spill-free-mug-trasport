"""Capability + value gallery: same tasks, Naive vs Ours.
Grid rows = method (top: Naive min-jerk → spills; bottom: Ours → stays in),
cols = scenarios. Shows across diverse tasks the baseline spills but ours doesn't.

Usage: python mp/make_video_gallery_compare.py
Output: figs/video_gallery_compare.gif
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
from mp.chomp import CHOMPConfig, chomp_optimize, minjerk_interp
from mp._video_utils import make_floor_cost
from mp.make_video_transport import build_scene, simulate
import PIL.Image

SCENARIOS = [
    ("Lateral + lift", (0.50, -0.30, 0.25), (0.50, 0.40, 0.40)),
    ("Lateral", (0.45, -0.30, 0.35), (0.45, 0.30, 0.35)),
    ("Reach across", (0.60, 0.05, 0.32), (0.38, 0.35, 0.28)),
]
T_VID = 0.5
N_WP = 40
THETA_MAX = 18.0
BETA_OBS = 3.0


def upright(p):
    T = np.eye(4); T[:3, :3] = R_UPRIGHT; T[:3, 3] = np.asarray(p, float); return T


def ik(fk, T_em, p, seed):
    T_ee = torch.tensor(upright(p) @ np.linalg.inv(T_em), dtype=torch.float64)
    q, info = ik_solve(fk, torch.tensor(seed, dtype=torch.float64), T_ee, n_iters=2500, lr=0.03)
    return q.numpy(), info


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    dt = T_VID / (N_WP - 1)
    scene, robot, mug, particles, cam, sapien_order = build_scene(g)

    naive_row, ours_row = [], []
    for label, ps, pg in SCENARIOS:
        q_s, i_s = ik(fk, T_em, ps, g['qpos'][:6])
        q_g, i_g = ik(fk, T_em, pg, q_s)
        if i_s['pos_err'] > 0.012 or i_g['pos_err'] > 0.012:
            print(f"[skip] {label}", flush=True); continue
        sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA_MAX, dt=dt), fk)
        # naive
        qn = minjerk_interp(q_s, q_g, N_WP)
        tn = sp.diagnostics(torch.tensor(qn, dtype=torch.float64))["tilt_deg"]
        # ours (with light floor cost to keep the hand above the table)
        floor = make_floor_cost(fk, T_em, z_floor=0.05, margin=0.04)
        ro = chomp_optimize(fk, q_s, q_g, CHOMPConfig(N=N_WP, dt=dt, n_iters=1200,
                            gamma_spill=5.0, beta_obs=BETA_OBS, step_size=0.003),
                            spill=sp, obs_cost_fn=floor)
        to = ro["spill_diag"]["tilt_deg"]
        print(f"[{label}] naive spill {(tn>THETA_MAX).sum()}/{N_WP}  "
              f"ours {(to>THETA_MAX).sum()}/{N_WP}", flush=True)
        naive_row.append(simulate(scene, robot, mug, particles, cam, sapien_order,
                                  qn, xhand, T_em, fk, tn, THETA_MAX,
                                  f"Min-jerk · {label}", dt))
        ours_row.append(simulate(scene, robot, mug, particles, cam, sapien_order,
                                 ro["q_traj"], xhand, T_em, fk, to, THETA_MAX,
                                 f"Ours · {label}", dt))

    N = min(len(f) for f in naive_row + ours_row)
    grid = []
    for i in range(N):
        top = np.concatenate([r[i] for r in naive_row], axis=1)
        bot = np.concatenate([r[i] for r in ours_row], axis=1)
        grid.append(np.concatenate([top, bot], axis=0))
    out = ROOT / "figs/video_gallery_compare.gif"
    pil = [PIL.Image.fromarray(f) for f in grid]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=80, loop=0)
    print(f"saved {out}  ({N} frames)", flush=True)


if __name__ == "__main__":
    main()
