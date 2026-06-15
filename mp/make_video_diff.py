"""Focused comparison on ONE task where Min-jerk and Ours take clearly different
paths (same start/goal, same time budget). Min-jerk spills; ours reshapes the
trajectory to stay spill-free. Path overlay + synced time make the difference clear.

Usage: python mp/make_video_diff.py
Output: figs/video_diff.gif
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

PS = (0.45, -0.30, 0.35); PG = (0.45, 0.30, 0.35)   # 60cm lateral, constant height
T = 0.5; N = 50; THETA = 18.0; BETA_OBS = 3.0


def upright(p):
    M = np.eye(4); M[:3, :3] = R_UPRIGHT; M[:3, 3] = np.asarray(p, float); return M


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    g = parse_grasp_info(ROOT / "data/grasp_info.txt")
    xhand = np.load(ROOT / "data/xhand_qpos.npy")
    T_em = np.load(ROOT / "data/ee_to_mug.npy")
    dt = T / (N - 1)
    scene, robot, mug, particles, cam, so = build_scene(g)

    def ik(p, seed):
        T_ee = torch.tensor(upright(p) @ np.linalg.inv(T_em), dtype=torch.float64)
        q, info = ik_solve(fk, torch.tensor(seed, dtype=torch.float64), T_ee, n_iters=2500, lr=0.03)
        return q.numpy(), info
    qs, _ = ik(PS, g['qpos'][:6]); qg, _ = ik(PG, qs)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA, dt=dt), fk)

    qn = minjerk_interp(qs, qg, N)
    tn = sp.diagnostics(torch.tensor(qn, dtype=torch.float64))["tilt_deg"]
    floor = make_floor_cost(fk, T_em, z_floor=0.05, margin=0.04)
    ro = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200, gamma_spill=5.0,
                        beta_obs=BETA_OBS, step_size=0.003), spill=sp, obs_cost_fn=floor)
    to = ro["spill_diag"]["tilt_deg"]
    print(f"min-jerk spill {(tn>THETA).sum()}/{N}   ours spill {(to>THETA).sum()}/{N}", flush=True)

    left = simulate(scene, robot, mug, particles, cam, so, qn, xhand, T_em, fk, tn, THETA,
                    "Min-jerk", dt)
    right = simulate(scene, robot, mug, particles, cam, so, ro["q_traj"], xhand, T_em, fk, to,
                     THETA, "Ours (CHOMP+spill)", dt)
    grid = [np.concatenate([left[i], right[i]], axis=1) for i in range(min(len(left), len(right)))]
    out = ROOT / "figs/video_diff.gif"
    pil = [PIL.Image.fromarray(f) for f in grid]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=80, loop=0)
    print(f"saved {out}  ({len(grid)} frames)", flush=True)


if __name__ == "__main__":
    main()
