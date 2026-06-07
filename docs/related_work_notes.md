# Related Work — Affordance-Grasp Conditioned Spill-Free Trajectory Optimization

> Survey for the motion-planning term project. Used in both the proposal and the
> final report. Items marked **[verify]** need an author/venue/year check before
> the final report cites them.

## 1. Waiter motion planning / non-prehensile transport (no-slip, no-tip)

- **Heins & Schoellig (2023)** "Keep It Upright: Model Predictive Control for Nonprehensile Object Transportation with Obstacle Avoidance on a Mobile Manipulator", *IEEE RA-L*. — Whole-body constrained MPC balancing an ungrasped object on the EE (waiter's problem) with obstacle avoidance; plans with the minimum statically-feasible friction coefficient for robustness. *Relation:* the no-tip/no-slip transport we extend, but object rests on a tray (friction, no grasp), cost is balance/friction not slosh, and grasp choice is irrelevant.
- **Selvaggio, Garg, Ruggiero, Oriolo & Siciliano (2023)** "Non-Prehensile Object Transportation via Model Predictive Non-Sliding Manipulation Control", *IEEE TCST*. — MPC enforcing non-sliding (friction-cone) contact during fast transport. *Relation:* same friction-constraint philosophy we use for spill margin, but rigid-object sliding not fluid slosh.
- **Subburaman, Selvaggio et al. (2023)** "A Non-Prehensile Object Transportation Framework With Adaptive Tilting Based on Quadratic Programming", *IEEE RA-L* [verify authors/venue]. — Adaptive tray tilt (QP) keeping inertial + gravity forces inside the friction cone for faster transport. *Relation:* "adaptive tilt to respect a margin" is the dual of our tilt-bound-to-avoid-spill.
- **Gao, Heins et al. (2024)** "Robust Nonprehensile Object Transportation with Uncertain Inertial Parameters", *arXiv:2411.07079* [verify venue]. — Robust upright-MPC under unknown mass/inertia/CoM. *Relation:* field's move toward uncertainty robustness; complementary, distinct from grasp-conditioning.
- **Acosta-Calderon & Hu (early 2000s)** tray-carrying / "waiter" balancing with s-curve acceleration profiles [verify paper/year]. — Classic: bound peak acceleration so a loosely-placed object does not tip/slide. *Relation:* historical origin of "bound acceleration so the object survives"; we generalize survival from rigid tipping to liquid spilling. (No specific Flores-Abad waiter paper confirmed — **[verify]**.)

## 2. Liquid / fluid transport without spilling (anti-sloshing)

- **Muchacho, Laha, Figueredo & Haddadin (2022)** "A Solution to Slosh-free Robot Trajectory Optimization", *IROS 2022* (arXiv:2210.12614). — EE as a point mass on a spherical pendulum; slosh-free trajectories via pivot control, cast as a QP; on a 7-DoF Franka. *Relation:* canonical reduced-order slosh model + optimization we build on, but fixed grasp/container pose, no affordance reasoning.
- **Moriello, Biagiotti, Melchiorri & Paoli (2018)** "Manipulating liquids with robots: A sloshing-free solution", *Control Engineering Practice*. — Sloshing as vibration suppression; input-shaping / exponential filters on the reference. *Relation:* feed-forward slosh-suppression baseline; no task/grasp conditioning.
- **Di Lillo, Pham et al. (2019)** "Robotic Handling of Liquids with Spilling Avoidance: A Constraint-Based Control Approach", *IEEE RAM / ICRA* [verify authors/venue]. — Constraint-based reactive control keeping the liquid surface within container limits. *Relation:* closest "spill as a hard constraint" precedent; we instead embed spill as a soft cost/margin in CHOMP/STOMP/MPPI tied to the grasp.
- **Gandhi, Sundaralingam, Pan et al. (2024)** "Clutter-Aware Spill-Free Liquid Transport via Learned Dynamics", *arXiv:2408.00215* [verify venue]. — Learns liquid dynamics, plans spill-free motion through clutter (slosh + obstacle avoidance). *Relation:* nearest contemporary "spill-free + planning"; still grasp-agnostic.
- **Tamosiunaite, Nemec, Ude & Wörgötter (2011)** "Learning to pour with a robot arm combining goal and shape learning for DMPs", *Robotics and Autonomous Systems*. — Learns pouring by adapting DMP goal + shape from demonstration. *Relation:* DMP-for-liquid reference; demonstration-driven, not optimization-with-explicit-spill-cost.
- **"SpillNot"-style learned spill-free motion** (thenospillproject) [verify formal citation]. — Learns to move a tray-carried open container without spilling. *Relation:* learned alternative to our optimization approach; comparison point.

## 3. Trajectory optimization with task-specific cost terms

- **Zucker, Ratliff, Dragan, … & Srinivasa (2013)** "CHOMP: Covariant Hamiltonian Optimization for Motion Planning", *IJRR* 32(9–10):1164–1193. (Short: **Ratliff, Zucker, Bagnell & Srinivasa (2009)**, *ICRA*.) — Functional-gradient traj-opt trading smoothness vs SDF obstacle term via covariant gradient descent. *Relation:* our base optimizer; spill cost is an added differentiable term in the CHOMP functional. **(This is the course's Lecture 8.)**
- **Kalakrishnan, Chitta, Theodorou, Pastor & Schaal (2011)** "STOMP: Stochastic Trajectory Optimization for Motion Planning", *ICRA*. — Gradient-free sampling-based traj-opt; handles non-differentiable costs. *Relation:* ideal for a non-differentiable spill/slosh cost from a simulator; bridge to MPPI.
- **Schulman, Duan, Ho, … Goldberg & Abbeel (2014)** "Motion planning with sequential convex optimization and convex collision checking" (TrajOpt), *IJRR* 33(9):1251–1270. — SQP planner with continuous-time collision constraints; admits custom costs/constraints. *Relation:* shows how arbitrary task constraints (tilt/spill) plug into a traj-opt solver.
- **Mukadam, Dong, Yan, Dellaert & Boots (2018)** "Continuous-Time Gaussian Process Motion Planning via Probabilistic Inference" (GPMP2), *IJRR* [verify issue]. — Planning as inference on a sparse GP; factor-graph addition of arbitrary cost factors. *Relation:* spill cost = extra unary factor; relevant for a probabilistic formulation.
- **Byravan, Boots, Srinivasa & Fox (2014)** "Space-Time Functional Gradient Optimization for Motion Planning" (T-CHOMP), *ICRA*. — Extends CHOMP into space-time so velocity/time-dependent costs (dynamics) can be optimized. *Relation:* slosh is velocity/acceleration-dependent → space-time formulation directly relevant.

## 4. Tilt-/orientation-constrained motion planning

- **Berenson, Srinivasa & Kuffner (2011)** "Task Space Regions: A Framework for Pose-Constrained Manipulation Planning", *IJRR* 30(12):1435–1460. — TSRs: efficient EE pose constraints (incl. bounded tilt) sampled within CBiRRT2. *Relation:* canonical way to enforce "tilt ≤ θ" as a constraint manifold; but geometric, grasp-/dynamics-agnostic, whereas our tilt bound is a dynamic spill margin.
- **Berenson, Srinivasa, Ferguson & Kuffner (2009)** "Manipulation Planning on Constraint Manifolds" (CBiRRT), *ICRA*. — Sampling-based planner projecting samples onto constraint manifolds for EE pose constraints. *Relation:* foundational constrained-planning machinery; we add a cost-based spill margin.
- **Kingston, Moll & Kavraki (2018)** "Sampling-Based Methods for Motion Planning with Constraints", *Annual Review of Control, Robotics, and Autonomous Systems*. — Survey of constraint strategies (projection, tangent-space, atlas). *Relation:* situates our tilt constraint in the broader toolbox.
- **Stilman (2010)** "Global Manipulation Planning in Robot Joint Space with Task Constraints", *IEEE T-RO* [verify year/vol]. — Task constraints (incl. orientation, e.g. carry a glass of water level) in joint-space RRT via tangent-space projection. *Relation:* explicitly motivated by "carry a full glass upright" — close analogue, but static upright constraint vs our dynamics-driven margin.

## 5. Grasp choice / affordance → downstream manipulation feasibility (our novelty's neighbors)

- **Fontanals, Dang-Vu, Porges, Rosell & Roa (2014) / "Grasping for a Purpose"** *ICRA / arXiv:1603.04338* [verify authors]. — Selects grasps accounting for downstream task constraints so the chosen grasp keeps the whole sequence kinematically feasible. *Relation:* closest in spirit — grasp gated by downstream feasibility — but feasibility is kinematic/reachability, not a physical spill margin.
- **Dang & Allen (2012/2014)** "Semantic grasping: Planning robotic grasps functionally suitable for an object manipulation task", *IROS 2012 / Autonomous Robots 2014*. — Exemplar-based planner with a "semantic affordance map" linking geometry to task-appropriate grasps. *Relation:* establishes "grasp must suit the task" (our affordance grasps embody this), but stops at grasp selection; never feeds the grasp into transport.
- **Fang, Zhu, Garg, Kurenkov, Mehta, Fei-Fei & Savarese (2020)** "Learning Task-Oriented Grasping for Tool Manipulation from Simulated Self-Supervision", *RSS 2018 / IJRR 2020*. — Learns task-oriented grasps from self-supervised sim. *Relation:* same "grasp depends on task"; learned, grasp-centric, no traj-opt or spill coupling.
- **Song, Ek, Huebner & Kragic (2010/2015)** "Task-based robot grasp planning using probabilistic inference", *T-RO 2015* [verify pairing]. — Task constraints (task wrench space / Bayesian nets) to score grasps. *Relation:* task-wrench-space is the classical quantitative "grasp affects task forces"; ours quantifies how grasp affects achievable spill margin.
- **Toussaint, Allen, Smith & Tenenbaum (2018)** "Differentiable Physics and Stable Modes for Tool-Use and Manipulation Planning" / Logic-Geometric Programming, *RSS 2018* [verify title]. — Jointly optimizes discrete mode (incl. grasp) + continuous trajectory so grasp makes downstream motion feasible/optimal. *Relation:* strongest precedent for co-optimizing grasp + trajectory, but general tool-use/contact, not spill margin, and no learned affordance prior.

## The gap our project fills

The two halves of our pipeline are mature **but disjoint**.

- **(a) Spill-free / anti-slosh transport** (Muchacho 2022, Moriello 2018, Gandhi 2024, constraint-based liquid handling) and **waiter-problem balancing** (Heins 2023, Selvaggio 2023) treat the **grasp/container pose as a fixed boundary condition** and optimize only the trajectory; the slosh model and tilt limit are computed for a given, unchosen grasp.
- **(b) Task-oriented / affordance grasping** (Dang & Allen, Fang, Song, "Grasping for a Purpose") shows **how you grasp depends on the task and gates downstream feasibility**, but stops at grasp selection — downstream metric is kinematic reachability or task wrench space, never a *physical transport margin*, and never feeds the grasp into a slosh-aware optimizer.
- Even the closest co-optimization (LGP / differentiable physics) couples grasp and motion only through kinematics/contact stability, not fluid dynamics.

**Our contribution = the missing bridge:** an **affordance-conditioned grasp choice as an explicit input to a spill-margin cost term inside CHOMP/STOMP/MPPI**. The grasp pose on the mug determines the achievable tilt/acceleration envelope before spilling (θ_max), and that spill margin in turn scores/selects the grasp. No surveyed work closes the loop *affordance grasp → spill margin → trajectory optimization* — that coupling is the defensible novelty.

## Key sources (links)
- Keep It Upright — arXiv:2305.17484
- Non-Prehensile MPC Non-Sliding — TCST 2023
- Slosh-free Trajectory Optimization — arXiv:2210.12614 (IROS 2022)
- Sloshing-free solution — Control Engineering Practice 2018
- Clutter-Aware Spill-Free — arXiv:2408.00215
- Learning to pour with DMPs — RAS 2011
- CHOMP — IJRR 2013 / ICRA 2009 (course Lecture 8)
- STOMP — ICRA 2011 ; TrajOpt — IJRR 2014 ; GPMP2
- Task Space Regions — IJRR 2011 ; CBiRRT — ICRA 2009
- Grasping for a Purpose — arXiv:1603.04338 ; Semantic Grasping — IROS 2012 ; Task-Oriented Grasping — RSS 2018
