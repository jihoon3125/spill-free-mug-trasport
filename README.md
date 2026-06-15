# Grasp-Conditioned Spill-Free Mug Transport

**Motion Planning Term Project — Jihoon Yoon** (SNU, H. Jin Kim, 2026)

Given a dexterous grasp of a mug, generate a 6-DoF UR5e arm trajectory that
transports it **without spilling**. We embed a differentiable *spill cost*
(derived from the apparent-gravity "waiter" model) into **CHOMP** and evaluate it
per grasp.

## Key results (77 reachable grasps, hardware-feasible budget)

- **Ours is the only method that is both spill-free (1.1% mean spill) and within
  joint limits (94% feasible).** Baselines either spill (chomp-smooth 14%,
  min-jerk 70%) or violate joint limits (vanilla-grad / STOMP, ~0% feasible).
- **Spill-robustness is a property of the grasp:** the fastest feasible
  spill-free transport time ranges 0.7–1.6 s across grasps and is **not** predicted
  by grasp geometry (r ≈ 0); 36% of grasps cannot reach the task at all → it must
  be *measured* by planning, not assumed.
- The spill weight γ trades spill vs. feasibility (γ≈1 is the knee); the method
  stays robust up to a ~half-full mug.

📄 **Report:** [`docs/report.pdf`](docs/report.pdf)  ·  🖥 **Slides:** [`docs/slides_final.pdf`](docs/slides_final.pdf)

## Repository layout

```
mp/      planner + experiments
sim/     SAPIEN scene builder
data/    mug mesh + grasp (copied read-only from the thesis dataset)
results/ experiment outputs (json / npz / csv)
figs/    figures and videos
docs/    report, slides, and their build scripts
```

## Core code

| File | What |
|---|---|
| `mp/kinematics.py` | differentiable URDF forward kinematics |
| `mp/ik.py` | Adam inverse kinematics |
| `mp/spill_cost.py` | apparent-gravity rim-cone spill cost |
| `mp/chomp.py` | CHOMP trajectory optimization (+ spill term) |
| `mp/stomp.py` | sampling-based baseline |

## Experiments

| File | Produces |
|---|---|
| `mp/run_method_comparison.py` | 5-method comparison + feasibility |
| `mp/run_tstar.py` | fastest feasible spill-free time T* per grasp |
| `mp/run_reachability.py` | reachability / offset per grasp |
| `mp/run_gamma_pareto.py` | spill ↔ feasibility trade-off vs γ |
| `mp/run_filllevel.py` | spill-robustness vs fill level (θ_max) |
| `mp/run_obstacle.py` | obstacle + spill coupling |
| `mp/run_overnight.py` | runs the full pipeline end-to-end → report |

## Reproduce

```bash
conda activate ultradexgrasp        # sapien, torch, trimesh, matplotlib
python mp/run_overnight.py          # full pipeline: experiments -> figures -> report
# or a single experiment, e.g.:
python mp/run_method_comparison.py --T 1.0 --max-grasps 80
```

Figures/report are rebuilt by `mp/make_*.py` and `docs/_scripts/build_report.py`
(PDF via LibreOffice `soffice --convert-to pdf`).

## Notes

- Spill is modeled with a reduced **quasi-static apparent-gravity cone** (after
  Muchacho et al., IROS 2022), not full CFD.
- All results use the **corrected mug up-axis convention** and report **joint-limit
  feasibility**; below a feasible budget a trajectory can be nominally spill-free
  but violate joint limits, so we operate at a moderate spill weight and budget.
