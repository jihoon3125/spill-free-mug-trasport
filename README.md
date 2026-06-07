# Motion Planning Term Project — Spill-Free Mug Transport

SNU Motion Planning (H. Jin Kim) term project, 2026.

## Overview
Given a mug mesh + a task-conditional grasp pose (from upstream affordance-aware grasp generation, my thesis), plan a 6-DoF end-effector trajectory from start to goal such that:
- the cup tilt stays within a no-spill cone
- obstacles are avoided
- the trajectory is smooth

## Method
Trajectory optimization (CHOMP variant) with three cost terms:
- smoothness (CHOMP default)
- obstacle (SDF-based)
- **spill cost** (effective gravity vector inside mug-rim cone)

## Evaluation
3-way ablation: linear interpolation / CHOMP (smooth + obs) / CHOMP + spill cost.
Metrics: spill ratio, trajectory length, smoothness (jerk integral), planning time.

## Folder layout
See `CLAUDE.md` (coding session) and `HANDOFF_TERMPROJECT.md` (materials session).
