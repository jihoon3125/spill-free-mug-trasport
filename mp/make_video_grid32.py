"""3x2 grid: 3 clean tasks (rows) x {Min-jerk, Ours} (cols), same time budget.
Each row compares the two methods on one task; path overlay + synced time make the
trajectory difference (and spill) clear. Output: figs/video_grid32.gif
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

TASKS = [
    ("Side-to-side", (0.45, -0.30, 0.35), (0.45, 0.30, 0.35)),
    ("Across-forward", (0.40, -0.25, 0.38), (0.58, 0.25, 0.38)),
    ("Pull-back", (0.58, -0.22, 0.36), (0.40, 0.28, 0.36)),
]
T = 0.5; N = 40; THETA = 18.0; BETA = 3.0


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

    rows_frames = []
    for label, ps, pg in TASKS:
        qs, _ = ik(ps, g['qpos'][:6]); qg, _ = ik(pg, qs)
        sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA, dt=dt), fk)
        floor = make_floor_cost(fk, T_em, z_floor=0.05, margin=0.04)
        qn = minjerk_interp(qs, qg, N)
        tn = sp.diagnostics(torch.tensor(qn, dtype=torch.float64))["tilt_deg"]
        ro = chomp_optimize(fk, qs, qg, CHOMPConfig(N=N, dt=dt, n_iters=1200, gamma_spill=5.0,
                            beta_obs=BETA, step_size=0.003), spill=sp, obs_cost_fn=floor)
        to = ro["spill_diag"]["tilt_deg"]
        print(f"[{label}] min-jerk {(tn>THETA).sum()}/{N}  ours {(to>THETA).sum()}/{N}", flush=True)
        left = simulate(scene, robot, mug, particles, cam, so, qn, xhand, T_em, fk, tn, THETA,
                        f"Min-jerk · {label}", dt)
        right = simulate(scene, robot, mug, particles, cam, so, ro["q_traj"], xhand, T_em, fk, to,
                         THETA, f"Ours · {label}", dt)
        rows_frames.append((left, right))

    Nf = min(min(len(l), len(r)) for l, r in rows_frames)
    grid = []
    for i in range(Nf):
        top = np.concatenate([l[i] for l, r in rows_frames], axis=1)   # 3 min-jerk
        bot = np.concatenate([r[i] for l, r in rows_frames], axis=1)   # 3 ours
        grid.append(np.concatenate([top, bot], axis=0))
    out = ROOT / "figs/video_grid23.gif"
    pil = [PIL.Image.fromarray(f) for f in grid]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=80, loop=0)
    print(f"saved {out}  ({Nf} frames)", flush=True)


if __name__ == "__main__":
    main()
