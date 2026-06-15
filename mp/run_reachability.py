"""Scaled reachability + offset pass (light: IK only, no T_min bisection).
Writes results/transport_map.json compatible with downstream scripts.
Usage: python mp/run_reachability.py --per-tax 40
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp.kinematics import URDFForwardKinematics
from sim.scene import URDF
from mp.run_grasp_comparison import extract_ee_to_mug
from mp.run_method_comparison import ik_multiseed

PS = (0.5, -0.3, 0.25); PG = (0.5, 0.4, 0.40)
TAX = ["2finger", "3finger", "whole"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-tax", type=int, default=40)
    args = ap.parse_args()
    fk = URDFForwardKinematics(URDF, base_link="base_link", ee_link="base")
    idxs = sorted(set(np.linspace(0, 299, args.per_tax).round().astype(int).tolist()))
    rows = []; total = len(idxs) * 3; done = 0
    for tax in TAX:
        for idx in idxs:
            done += 1; name = f"mug_0_left_{tax}_{idx}"
            try:
                T_em, _, g = extract_ee_to_mug(name)
            except Exception as e:
                print(f"[{done}/{total}] {name} LOAD FAIL {e}", flush=True); continue
            offset = float(np.linalg.norm(T_em[:3, 3]))
            rec = dict(grasp_name=name, taxonomy=tax, idx=int(idx), offset_r=offset,
                       reachable=False, T_min=None, spill_at=None)
            qs, i_s = ik_multiseed(fk, T_em, PS, g['qpos'][:6])
            if i_s['pos_err'] <= 0.01:
                qg, i_g = ik_multiseed(fk, T_em, PG, qs)
                rec["reachable"] = bool(i_g['pos_err'] <= 0.01)
            rows.append(rec)
            if done % 10 == 0:
                print(f"[{done}/{total} {done*100//total}%] reachable so far "
                      f"{sum(r['reachable'] for r in rows)}", flush=True)
    json.dump(rows, open(ROOT / "results/transport_map.json", "w"), indent=2)
    nr = sum(r["reachable"] for r in rows)
    print(f"saved transport_map.json: {nr}/{len(rows)} reachable ({nr*100//len(rows)}%)", flush=True)


if __name__ == "__main__":
    main()
