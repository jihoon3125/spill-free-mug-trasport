

# Overnight run 2026-06-14 16:57

- [16:57:46] Plan: A (scale grasps) -> B (gamma pareto + fill level) -> figures -> report
- [16:57:46] **START** A1 reachability (120 grasps): `mp/run_reachability.py --per-tax 40`
- [17:15:25] **OK** A1 reachability (120 grasps) (17.6 min)
```
[10/120 8%] reachable so far 6
[20/120 16%] reachable so far 14
[30/120 25%] reachable so far 20
[40/120 33%] reachable so far 28
[50/120 41%] reachable so far 34
[60/120 50%] reachable so far 36
[70/120 58%] reachable so far 43
[80/120 66%] reachable so far 48
[90/120 75%] reachable so far 55
[100/120 83%] reachable so far 64
[110/120 91%] reachable so far 71
[120/120 100%] reachable so far 77
saved transport_map.json: 77/120 reachable (64%)
```
- [17:15:25] **START** A2 method comparison (T=1.0, up to 80 grasps): `mp/run_method_comparison.py --T 1.0 --max-grasps 80`
- [17:43:10] **OK** A2 method comparison (T=1.0, up to 80 grasps) (27.8 min)
```
-jerk=70%  chomp-smooth=0%  stomp-spill=42%  vanilla-grad=68%  ours=0%
--- mug_0_left_whole_207  min-jerk=62%  chomp-smooth=4%  stomp-spill=44%  vanilla-grad=66%  ours=0%
--- mug_0_left_whole_215  min-jerk=82%  chomp-smooth=0%  stomp-spill=30%  vanilla-grad=82%  ours=0%
--- mug_0_left_whole_230  min-jerk=66%  chomp-smooth=0%  stomp-spill=40%  vanilla-grad=56%  ours=0%
--- mug_0_left_whole_238  min-jerk=76%  chomp-smooth=0%  stomp-spill=48%  vanilla-grad=68%  ours=0%
--- mug_0_left_whole_261  min-jerk=80%  chomp-smooth=30%  stomp-spill=34%  vanilla-grad=54%  ours=0%
--- mug_0_left_whole_268  min-jerk=76%  chomp-smooth=0%  stomp-spill=30%  vanilla-grad=66%  ours=0%
--- mug_0_left_whole_291  min-jerk=76%  chomp-smooth=0%  stomp-spill=68%  vanilla-grad=58%  ours=0%
--- mug_0_left_whole_299  min-jerk=78%  chomp-smooth=0%  stomp-spill=32%  vanilla-grad=66%  ours=0%
saved /home/dongjae/motion_planning_termproject/results/method_comparison.json
=== aggregate spill ratio over 77 grasps (T=1.0s) ===
method            mean  median  %@0spill %feasible  plan_s
stomp-spill      50.2%   50.0%       0%        0%   13.05
saved /home/dongjae/motion_planning_termproject/figs/fig_method_comparison.png
```
- [17:43:10] **START** A3 T* sweep: `mp/run_tstar.py`
- [18:17:02] **OK** A3 T* sweep (33.9 min)
```
0.0]
[66/77] mug_0_left_whole_169 T*=0.9 spills=[0.1, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[67/77] mug_0_left_whole_176 T*=0.9 spills=[0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
[68/77] mug_0_left_whole_184 T*=0.8 spills=[0.18, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[69/77] mug_0_left_whole_192 T*=0.7 spills=[0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[70/77] mug_0_left_whole_207 T*=0.9 spills=[0.36, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[71/77] mug_0_left_whole_215 T*=0.8 spills=[0.08, 0.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
[72/77] mug_0_left_whole_230 T*=0.9 spills=[0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[73/77] mug_0_left_whole_238 T*=0.9 spills=[0.08, 0.06, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0]
[74/77] mug_0_left_whole_261 T*=0.9 spills=[0.2, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[75/77] mug_0_left_whole_268 T*=0.9 spills=[0.28, 0.32, 0.18, 0.0, 0.0, 0.0, 0.0, 0.0]
[76/77] mug_0_left_whole_291 T*=0.8 spills=[0.24, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
[77/77] mug_0_left_whole_299 T*=0.8 spills=[0.0, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
n=77  monotonic spill(T): 58/77 (75%)
T* defined for 76/77 grasps; range 0.7-1.6s, median 0.9s
corr(offset, T*) = 0.12
saved figs/fig_tstar.png + fig_tstar_offset.png
```
- [18:17:02] **START** B1 gamma pareto: `mp/run_gamma_pareto.py`
- [18:22:16] **OK** B1 gamma pareto (5.2 min)
```
  gamma=0.0: spill=12%  feasible=100%
  gamma=0.3: spill=8%  feasible=100%
  gamma=0.7: spill=3%  feasible=93%
  gamma=1.0: spill=1%  feasible=100%
  gamma=2.0: spill=0%  feasible=87%
  gamma=3.0: spill=0%  feasible=87%
  gamma=5.0: spill=3%  feasible=73%
saved figs/fig_gamma_pareto.png
```
- [18:22:16] **START** B2 fill level: `mp/run_filllevel.py`
- [18:23:53] **OK** B2 fill level (1.6 min)
```
  theta=10 (~93% full): mean spill=38%  grasps spill-free=30%
  theta=15 (~89% full): mean spill=10%  grasps spill-free=65%
  theta=18 (~87% full): mean spill=0%  grasps spill-free=95%
  theta=22 (~83% full): mean spill=0%  grasps spill-free=100%
  theta=26 (~80% full): mean spill=0%  grasps spill-free=100%
  theta=30 (~76% full): mean spill=0%  grasps spill-free=100%
  theta=35 (~71% full): mean spill=0%  grasps spill-free=100%
saved figs/fig_filllevel.png
```
- [18:23:53] **START** figs extra (cone/accel/reach/feasibility): `mp/make_extra_figs.py`
- [18:23:59] **OK** figs extra (cone/accel/reach/feasibility) (0.1 min)
```
saved figs/fig_cone.png
saved figs/fig_accel.png
saved figs/fig_reach.png
saved figs/fig_feasibility.png
```
- [18:23:59] **START** fig trajectories: `mp/make_fig_trajectories.py`
- [18:24:19] **OK** fig trajectories (0.3 min)
```
saved /home/dongjae/motion_planning_termproject/figs/fig_trajectories.png
  min-jerk       max tilt 22°  spill 50%
  chomp-smooth   max tilt 8°  spill 0%
  stomp-spill    max tilt 50°  spill 70%
  vanilla-grad   max tilt 54°  spill 54%
  ours           max tilt 6°  spill 0%
```
- [18:24:19] **START** build report docx: `docs/_scripts/build_report.py`
- [18:24:19] **OK** build report docx (0.0 min)
```
saved docs/report.docx  (figs=10, n_grasps=77, ours=1.1%/94%feas)
```
- [18:24:19] converting report to PDF ...
- [18:24:21] **DONE** report.pdf rebuilt. Overnight pipeline complete.
