# Grasp-Conditioned Spill-Free Mug Transport

**Motion Planning Term Project — Jihoon Yoon**

---

## Abstract

Given a dexterous grasp of a mug, we generate a 6-DoF arm trajectory that
transports it without spilling. We add a differentiable *spill cost* — derived
from the apparent-gravity ("waiter") model — to CHOMP, and evaluate it on every
grasp. Across 40 grasps and four baselines, our spill-aware planner reaches the
lowest spill (11% vs 50–97%) at lower compute than sampling, and this ranking is
robust to both the time budget and the spill-angle threshold. We further show
that spill-robustness is a property of the *grasp*: it varies widely across
grasps (0–34% at a fixed budget; a third of grasps are unreachable) and is **not**
predictable from grasp geometry, so it must be measured by planning. Finally, we
show the cost composes with obstacle avoidance — avoiding an obstacle *alone*
induces spilling (100%), while jointly optimizing both does not (5%).

---

## 1. Introduction

My thesis generates affordance- and taxonomy-aware dexterous grasps: given an
object and a task, it produces a hand pose. A natural next question is what comes
*after* the grasp — can the robot **carry** the grasped object safely? This
project studies the canonical case of transporting a mug of liquid: produce an
arm trajectory that moves the mug from A to B **without spilling**.

The key physical insight is the *waiter problem*: a cup spills not only when it
is geometrically tilted, but whenever it **accelerates**. What the liquid "feels"
is the effective gravity `g_eff = g − a`; spilling occurs when this vector leaves
the cone defined by the cup's rim. Thus a perfectly upright cup spills if moved
too aggressively.

Contributions:

1. A differentiable spill cost integrated into CHOMP that produces spill-free
   transport trajectories, validated against four baselines and shown robust to
   time budget and spill threshold.
2. A grasp-conditioned analysis showing that spill-robustness is a property of
   the grasp and is **not** predictable from simple grasp geometry — it must be
   measured by planning.
3. A demonstration that the spill cost composes with obstacle avoidance, and that
   the two terms must be optimized jointly.

---

## 2. Related Work

Prior work either (a) plans liquid-safe motions for a *fixed* grasp using slosh
models or input shaping, or (b) selects task-appropriate grasps but stops before
transport. To our knowledge, none links *grasp selection* to a physical spill
margin. We use a reduced quasi-static slosh model (apparent-gravity cone), which
prior work (e.g. reduced-order slosh control) has shown to capture the dominant
spilling behavior without full CFD.

---

## 3. Problem Formulation

Let `q(t) ∈ R^6` be the UR5e arm trajectory; the 12 hand joints are frozen at the
grasp pose. The grasp fixes the rigid transform `T_ee→mug` between the hand's
end-effector frame and the mug. Given start/goal mug poses (upright), we seek a
smooth trajectory that keeps the apparent-gravity vector inside the cup's rim
cone for all `t`, while remaining collision-free.

---

## 4. Method

### 4.1 Kinematics

We implement a differentiable forward-kinematics chain walker over the URDF
(`mp/kinematics.py`); it matches SAPIEN to `1e-7`. Inverse kinematics
(`mp/ik.py`) is Adam-based gradient descent on a pose loss
`‖p − p*‖² + λ(3 − tr(R R*ᵀ))`. For each grasp the mug pose in the world is
`T_world_mug = FK(q) · T_ee→mug`.

### 4.2 Spill cost (apparent-gravity cone)

For each waypoint we compute the mug-center world position `p_mug(t)` and its
opening axis `n(t)`. The mug acceleration is the second finite difference of the
**actual mug-center trajectory**:

```
a(t)      = (p_mug(t+1) − 2 p_mug(t) + p_mug(t−1)) / dt²
g_eff(t)  = g − a(t)
align(t)  = − ĝ_eff(t) · n(t)
```

Because `p_mug = FK(q)·T_ee→mug`, the acceleration includes the per-grasp lever
arm exactly (the rotational `α×r` and centripetal `ω×(ω×r)` terms), with no
approximation — different grasps yield different `T_ee→mug` and therefore
different mug accelerations.

No-spill requires `align(t) ≥ cos θ_max`; the penalty is

```
c_spill(t) = max(0, cos θ_max − align(t))²,    J_spill = Σ_t c_spill(t)
```

with `θ_max = 18°` (a roughly half-full mug; we show the conclusions are
threshold-robust in §5.3).

### 4.3 Trajectory optimization

We minimize `J = α J_smooth + β J_obs + γ J_spill` with CHOMP covariant updates
`ξ ← ξ − η M⁻¹ ∇J`, where `M` is the discrete acceleration metric. We use Adam
with `M⁻¹` preconditioning and gradient clipping for stability. Endpoints are
fixed; only inner waypoints are optimized.

### 4.4 Per-grasp evaluation

For a grasp we (i) solve IK for the upright start/goal (multi-seed, so an IK
failure means genuine unreachability), and (ii) optimize the spill-aware
trajectory. The resulting spill ratio at a fixed time budget is our measure of
the grasp's spill-robustness.

---

## 5. Experiments

### 5.1 Setup

Object: a mug (`mug_0`) with 60 grasps from the thesis dataset (2-finger /
3-finger / whole-hand). Transport task unless noted: 70 cm lateral + 15 cm lift,
upright endpoints. Metric: spill ratio = fraction of waypoints with effective
tilt above `θ_max`.

### 5.2 Method comparison (controlled ablation)

We compare five methods, each changing exactly one ingredient, on 40 reachable
grasps at a challenging budget `T = 0.5 s` (Fig. `fig_method_comparison.png`):

| Method | spill term | optimizer | CHOMP metric | mean spill | plan time |
|---|---|---|---|---|---|
| min-jerk | ✗ | analytic | — | 97% | 0.0 s |
| chomp-smooth | ✗ | gradient | ✓ | 93% | 0.2 s |
| stomp-spill | ✓ | sampling | — | 50% | 12.8 s |
| vanilla-grad | ✓ | gradient | ✗ | 66% | 3.0 s |
| **ours** | ✓ | gradient | ✓ | **11%** | 3.0 s |

The spill term is necessary (without it, 93–97%); the CHOMP metric is decisive
(ours 11% vs vanilla-grad 66%); and gradient optimization beats sampling at lower
compute (ours 11% / 3 s vs STOMP 50% / 12.8 s). The geometric cup tilt stays
small (≈2–4°): ours avoids spilling by **shaping the acceleration profile**, not
by tilting the cup.

### 5.3 Robustness (Exp A)

The ranking is robust to both the time budget and the spill threshold (Figs.
`fig_robust_T.png`, `fig_robust_theta.png`):

*Spill vs time budget (`θ = 18°`):*

| T (s) | min-jerk | chomp-smooth | vanilla-grad | ours |
|---|---|---|---|---|
| 0.4 | 100 | 96 | 66 | **9** |
| 0.6 | 92 | 55 | 66 | **6** |
| 1.0 | 75 | 13 | 73 | **2** |

*Spill vs spill threshold (`T = 0.5 s`):*

| θ_max | min-jerk | chomp-smooth | vanilla-grad | ours |
|---|---|---|---|---|
| 10° | 100 | 98 | 91 | **67** |
| 18° | 98 | 95 | 68 | **10** |
| 35° | 88 | 18 | 26 | **1** |

Two observations. (i) Smoothness alone suffices only when slow (chomp-smooth
drops to 13% at 1.0 s but is 96% at 0.4 s) — the spill term matters precisely
when one must move fast. (ii) Without the CHOMP metric, the gradient cannot
exploit a larger time budget (vanilla-grad stays ≈66–73% for all T at equal
optimization budget). Ours is lowest at every setting.

### 5.4 Grasp-conditioned spill-robustness and mechanism (Exp B)

Spill-robustness is a property of the grasp. Of 60 grasps, **34% cannot reach**
the transport workspace at all. Among the reachable ones, the achievable spill at
`T = 0.5 s` ranges from **0 to 34%** (Fig. `fig_spill_by_grasp.png`).

Crucially, this is **not predictable from grasp geometry**. We correlate several
grasp features with the achievable spill (Fig. `fig_mechanism.png`):

| feature | corr. r | r² |
|---|---|---|
| cup–EE tilt in hand | −0.34 | 0.11 |
| vertical lever r_z | +0.13 | 0.02 |
| offset magnitude r | +0.12 | 0.01 |
| horizontal lever r_xy | −0.05 | 0.00 |

No single feature explains more than 11% of the variance. Hence one cannot pick a
spill-robust grasp from a heuristic; it must be **measured by planning**. (Note we
report per-grasp results at a fixed budget rather than a "minimum spill-free time"
because CHOMP's non-convexity makes spill non-monotonic in `T` — see §6.)

### 5.5 Obstacle coupling (Exp C)

We add a box obstacle on the path and an obstacle cost, at an aggressive budget
(`T ≈ 0.6 s`) (Figs. `fig_obstacle_3d.png`, `fig_obstacle_metrics.png`):

| Method | spill | collision |
|---|---|---|
| min-jerk | 93% | 25% |
| CHOMP + obstacle only | **100%** | 0% |
| **ours (obstacle + spill)** | **5%** | 0% |

Avoiding the obstacle *alone* forces a detour that increases acceleration and
spills everything (100%). Only the joint objective achieves both collision-free
and spill-free transport, demonstrating that the terms must be optimized
together.

### 5.6 Qualitative results

Fig. `fig_trajectories.png` shows the five methods' mug-center paths and their
effective tilt over time on one task — ours stays below the spill threshold while
the others spike. Side-by-side videos (`video_grid23.gif`) show min-jerk spilling
and ours keeping the water in across three transport scenarios at the same time
budget, with the mug-center path overlaid. (Water particles are a visualization
of the reduced slosh model, not a fluid simulation.)

---

## 6. Discussion & Limitations

- **Reduced spill model.** We use the quasi-static apparent-gravity cone, not full
  CFD. We also implemented a rigid-body particle simulation as an independent
  check, but rigid spheres behave granularly (not like liquid) and saturate when
  the cup is brim-full, so we did not use it for quantitative claims.
- **Non-monotonic optimization.** CHOMP is non-convex; spill is not monotonic in
  the time budget, so a "minimum spill-free time" metric obtained by bisection is
  unreliable. We therefore report spill at a fixed budget, which is deterministic
  and reproducible.
- **Cost coupling tension.** A floor/obstacle term conflicts with the spill term
  at very aggressive budgets; light weighting keeps the hand above the table
  without harming spill, but the trade-off is real.
- **Scope.** A single object (one mug instance) and 60 grasps. Generalization
  across objects and instances is future work.
- **Process note.** We found and fixed an inverted cup up-axis convention during
  the project; all reported results use the corrected convention.

---

## 7. Conclusion & Future Work

This project achieved two things. First, we built a spill-aware planner that
**reliably generates spill-free transport trajectories**, beating four baselines
and remaining robust across time budgets and spill thresholds. Second, we showed
that **spill margin is a property of the grasp** — it varies widely and is not
predictable from grasp geometry, so it must be measured. This is exactly the kind
of signal a grasp generator could use: future work is to feed the measured
spill-margin back into grasp selection, closing the loop with the thesis, and to
extend the study to more objects and to a learned spill model.
```
