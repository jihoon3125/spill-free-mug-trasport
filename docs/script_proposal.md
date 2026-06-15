# Proposal Talk Script — Affordance-Grasp Conditioned Spill-Free Trajectory Optimization

**Target: 2–3 min (≈2:45) · 7 slides · English delivery**
Deck: `slides_proposal_editable.pptx`  ·  (English of the revised Korean draft `script_proposal_ko.md`)

---

## Slide 1 — Title (~12s)
> Hi, I'm Jihoon Yun, an undergraduate in Mechanical Engineering. My question today is simple: **does *how you grasp* a cup change *how safely you can carry* it?** I'll argue yes — and frame it as a motion-planning problem.

*Delivery: say the subtitle question slowly; make eye contact.*

## Slide 2 — Big picture (~30s)
> First, some background. My thesis is on **affordance- and taxonomy-aware grasp generation**: given an object and a natural-language task, it produces a suitable multi-finger grasp — same mug, but a different grasp for a different task. In this project I focus on the mug, and on the step *after* the grasp — the **carry**: producing a trajectory that moves it safely without spilling. The ultimate goal is to generate the grasp *and* the trajectory that together carry it most safely.

*Delivery: point left → right → bottom banner. Beat on "carry".*

## Slide 3 — Problem & key insight (~30s)
> The problem I solve here: given a mug grasp, plan a trajectory that **doesn't spill**, avoids obstacles, and is smooth. Whether the mug spills depends on **effective gravity** — gravity minus acceleration — so a perfectly upright cup can still spill when it accelerates. Concretely, no-spill means the **apparent-up direction, a minus g, stays inside the cup's rim cone**.

*Delivery: "a perfectly upright cup can still spill" is the hook — pause here.*

## Slide 4 — Why the grasp matters (~30s)
> So why does the grasp affect spilling? First, the grasp sets the **distance r from the end-effector to the cup**, so the *same* trajectory gives a different cup acceleration depending on the grasp. Second, different grasps have **different reachable regions**. So in this project, we generate a trajectory **for each grasp**, and then decide which grasp is best — we *measure* it, rather than assume it.

*Delivery: point to the handle / body panels. Land "for each grasp" — it sets up Slide 6.*

## Slide 5 — Related work & gap (~25s)
> Existing work either makes spill-free trajectories for a **fixed** grasp, or selects a task-appropriate grasp but **never carries it**. No prior work connects grasp selection to a **physical spill margin** — and that is what makes our approach different.

*Delivery: trace the "Gap" column with your hand; raise tone on the red sentence.*

## Slide 6 — Method (~30s)
> To plan the trajectory, we take **CHOMP** from class and add **one spill-cost term** — a differentiable penalty that activates when the apparent-up leaves the cone. We build this trajectory **for each grasp**, compute each grasp's **margin to the spill cone**, and select the safest grasp. Later, we plan to fold this spill score into the **reward function**, unifying everything into a single grasp generator.

*Delivery: top equation = "one added term"; bottom = "differentiable". Point bars → feedback arrow at the end.*

## Slide 7 — Evaluation & plan (~25s)
> For evaluation: a **three-way ablation** — linear, plain CHOMP, and ours — compared on spill ratio, smoothness, and planning time, plus a **SAPIEN particle demo** that counts particles crossing the rim. All within about a week. **Thank you.**

*Delivery: brisk. Beat before "Thank you".*

---

## Timing control
- **~2:45** as written. If time allows, slow down on Slides 3–4.
- **Trim to ~2:00:**
  - Slide 5 → one line: *"Prior work either fixes the grasp, or stops at grasp selection — nobody links grasp to a spill margin."*
  - Slide 7 → *"We validate with a 3-way ablation and a particle demo, in about a week. Thank you."*

## One-line throughline (memorize)
> The thesis picks the grasp → this project carries it → and the carry tells us which grasp was best.

## Q&A prep
- **"Real fluid sim?"** → No full CFD — a reduced slosh model (the apparent-gravity cone) plus surface particles over the rim for intuition. Muchacho 2022 validates the reduced model.
- **"Does the grasp really change the margin?"** → Yes — the grasp sets the cup's offset r, so the same arm motion gives a different cup acceleration. That is exactly what the per-grasp evaluation measures.
- **"Why CHOMP, not RRT/MPC?"** → Spill is a smooth, differentiable cost on acceleration; CHOMP's covariant gradient handles it naturally and reuses the smoothness operator. STOMP/MPPI is the gradient-free fallback.
