"""Experiment B — mechanism analysis: which grasp property predicts spill?

Beyond the EE->mug offset magnitude (shown ~uncorrelated), correlate several
grasp-geometry features with ours' spill@T=0.5s across grasps. If none is
predictive, that reinforces "spill-robustness must be measured, not assumed."

Features (from T_ee_to_mug):
  r       : |translation|                  (offset magnitude)
  r_xy    : horizontal offset              (lever arm in the table plane)
  r_z     : |vertical offset|              (lever arm along gravity)
  tilt_EE : angle of mug-opening axis from the EE z-axis (how tilted in hand)

Output: results/mechanism.json, figs/fig_mechanism.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp.constants import MUG_UP_AXIS_LOCAL
from mp.run_grasp_comparison import extract_ee_to_mug


def features(T_em):
    t = T_em[:3, 3]; R_em = T_em[:3, :3]
    up_ee = R_em @ np.asarray(MUG_UP_AXIS_LOCAL)      # mug opening axis in EE frame
    up_ee = up_ee / (np.linalg.norm(up_ee) + 1e-9)
    tilt_ee = np.degrees(np.arccos(np.clip(abs(up_ee[2]), 0, 1)))
    return dict(r=float(np.linalg.norm(t)) * 100,
                r_xy=float(np.linalg.norm(t[:2])) * 100,
                r_z=float(abs(t[2])) * 100,
                tilt_EE=float(tilt_ee))


def main():
    mc = json.load(open(ROOT / "results/method_comparison.json"))["per_grasp"]
    feats = {k: [] for k in ["r", "r_xy", "r_z", "tilt_EE"]}
    spill = []; tax = []
    for g in mc:
        try:
            T_em, _, _ = extract_ee_to_mug(g["grasp_name"])
        except Exception as e:
            print("skip", g["grasp_name"], e); continue
        f = features(T_em)
        for k in feats:
            feats[k].append(f[k])
        spill.append(g["methods"]["ours"]["spill_ratio"] * 100)
        tax.append(g["taxonomy"])
    spill = np.array(spill)

    corr = {k: float(np.corrcoef(np.array(v), spill)[0, 1]) for k, v in feats.items()}
    print(f"\nn={len(spill)}   correlation of grasp feature vs ours spill@0.5s:")
    for k, c in sorted(corr.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {k:8s}  r = {c:+.2f}  (r^2 = {c*c:.2f})")
    json.dump({"corr": corr, "n": len(spill)}, open(ROOT / "results/mechanism.json", "w"), indent=2)

    # figure: scatter of each feature vs spill, with r in title
    col = {"2finger": "#1f77b4", "3finger": "#ff7f0e", "whole": "#2ca02c"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axes, ["r", "r_xy", "r_z", "tilt_EE"]):
        for t in col:
            idx = [i for i, tt in enumerate(tax) if tt == t]
            ax.scatter([feats[k][i] for i in idx], [spill[i] for i in idx],
                       c=col[t], s=28, edgecolor="black", label=t)
        ax.set_xlabel(k); ax.set_ylabel("spill@0.5s (%)")
        ax.set_title(f"{k}: r = {corr[k]:+.2f}"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("No single grasp feature predicts spill-robustness", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "figs/fig_mechanism.png", dpi=160, bbox_inches="tight")
    print("saved figs/fig_mechanism.png")


if __name__ == "__main__":
    main()
