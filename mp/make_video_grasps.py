"""Thesis-headline video: SAME task, SAME planner (CHOMP+spill), SAME time budget
— only the GRASP changes. Shows that spill-robustness is a property of the grasp.

3 grasps side by side, each planned on the fly at a fixed (challenging) T; the
kinematic water illustration shows which grasp stays dry and which spills.

Usage: python mp/make_video_grasps.py
Output: figs/video_grasps.gif
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.scene import URDF
from mp.kinematics import URDFForwardKinematics
from mp.spill_cost import SpillCost, SpillConfig
from mp.chomp import CHOMPConfig, chomp_optimize
from mp.run_grasp_comparison import extract_ee_to_mug
from mp.run_method_comparison import ik_multiseed, upright_pose
from mp.make_video_transport import build_scene, simulate
import PIL.Image

# (grasp_name, descriptor)
GRASPS = [
    ("mug_0_left_3finger_268", "Spill-robust grasp"),
    ("mug_0_left_2finger_126", "Spill-prone grasp"),
]
T_VID = 0.45
N_WP = 50
THETA_MAX = 18.0
P_START = (0.5, -0.3, 0.25)
P_GOAL = (0.5, 0.4, 0.40)


def plan(fk, T_em, q_seed):
    q_s, i_s = ik_multiseed(fk, T_em, P_START, q_seed)
    q_g, i_g = ik_multiseed(fk, T_em, P_GOAL, q_s)
    if i_s['pos_err'] > 0.01 or i_g['pos_err'] > 0.01:
        return None
    dt = T_VID / (N_WP - 1)
    sp = SpillCost(SpillConfig(T_ee_to_mug=T_em, theta_max_deg=THETA_MAX, dt=dt), fk)
    res = chomp_optimize(fk, q_s, q_g,
                         CHOMPConfig(N=N_WP, dt=dt, n_iters=1200, gamma_spill=5.0,
                                     step_size=0.003), spill=sp)
    return res["q_traj"], res["spill_diag"]["tilt_deg"], dt


def main():
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    panels = []
    for name, desc in GRASPS:
        print(f"[plan+render] {name}  ({desc})", flush=True)
        T_em, xhand, ginfo = extract_ee_to_mug(name)
        out = plan(fk, T_em, ginfo['qpos'][:6])
        if out is None:
            print(f"  IK fail — skip {name}"); continue
        q_traj, tilt, dt = out
        print(f"  max tilt {tilt.max():.0f}deg  spill frames {(tilt>THETA_MAX).sum()}/{N_WP}",
              flush=True)
        scene, robot, mug, particles, cam, sapien_order = build_scene(ginfo)
        frames = simulate(scene, robot, mug, particles, cam, sapien_order,
                          q_traj, xhand, T_em, fk, tilt, THETA_MAX,
                          f"{desc}", dt)
        panels.append(frames)

    N = min(len(p) for p in panels)
    combined = [np.concatenate([p[i] for p in panels], axis=1) for i in range(N)]
    out = ROOT / "figs/video_grasps.gif"
    pil = [PIL.Image.fromarray(f) for f in combined]
    pil[0].save(out, save_all=True, append_images=pil[1:], duration=80, loop=0)
    print(f"saved {out}  ({N} frames, {len(panels)} grasps)", flush=True)


if __name__ == "__main__":
    main()
