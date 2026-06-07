---
marp: true
theme: default
paginate: true
size: 16:9
math: katex
style: |
  section { font-family: 'Noto Sans KR', sans-serif; padding: 38px 56px; }
  h1 { color: #1a365d; }
  h2 { color: #2c3e50; border-bottom: 2px solid #ccc; padding-bottom: 4px; }
  table { font-size: 0.84em; }
  img { max-height: 410px; }
  .small { font-size: 0.8em; color: #555; }
  .hl { color: #c0392b; font-weight: bold; }
  .big { font-size: 1.13em; }
  .tag { background: #1a365d; color: #fff; padding: 2px 10px; border-radius: 3px; font-size: 0.7em; }
---

# Affordance-Grasp Conditioned<br>Spill-Free Trajectory Optimization

**Carrying a mug of liquid without spilling**

<span class="small"><b>My thesis:</b> task-conditional dexterous grasp generation — object + a natural-language task (e.g. "pour", "hand over") → <b>how a multi-finger hand should grasp it</b>.<br><b>This project (motion planning):</b> given such a grasp, plan the <b>carry</b> — and ask whether the grasp choice itself helps.</span>

<span class="big">A fast motion spills the cup <span class="hl">even when held perfectly upright</span> — liquid responds to <b>effective gravity</b> $g - a_{ee}$.</span>

<span class="tag">Motion Planning · Proposal</span> &nbsp; Jihoon Yun · 2026-06-08

---

## 1. Problem — grasp decides the spill margin

![w:760](figs/spill_cone.png)

<span class="small">No-spill ⇔ apparent-up $a_{ee}-g$ stays in the cup's <b>rim cone</b> ($\theta_{max}$). The <b>grasp</b> (taxonomy / region — handle vs rim vs body) fixes the hand→cup transform, so it sets <b>how wide that cone is</b> and how much acceleration the carry can afford.</span>

---

## 2. Related Work & Our Difference

| Line of work | What they do | Gap |
|---|---|---|
| Anti-slosh / spill-free planning <br><span class="small">Muchacho '22, Gandhi '24</span> | spill-free trajectory for a **fixed** container pose | grasp is a fixed boundary condition |
| Waiter / non-prehensile <br><span class="small">Heins '23, Selvaggio '23</span> | keep object from tipping/sliding (friction cone) | rigid object, no liquid, no grasp choice |
| Affordance / task-oriented grasping <br><span class="small">"Grasping for a Purpose" '14, Dang '12</span> | pick a task-appropriate grasp | **stops at selection** — never carries it |

<span class="hl">Nobody connects grasp choice to a physical spill margin.</span> We make the trajectory optimizer **score grasps** by how safely they can carry → the missing bridge.

---

## 3. Method — Spill-Aware CHOMP as a grasp evaluator

![w:1000](figs/grasp_evaluator.png)

$$
J(\xi) = \underbrace{J_{\text{smooth}} + J_{\text{obs}}}_{\text{course CHOMP (Lec. 8)}} + \;\lambda_s\, \underbrace{J_{\text{spill}}(\xi)}_{\text{ours, differentiable}}
\qquad
\text{\small eval: } \theta_{max}(\text{grasp}) = \text{achievable margin}
$$

<span class="small"><b>This term project:</b> spill cost in CHOMP + 3-way ablation (linear / CHOMP / +spill) on a mug + SAPIEN particle demo, and run it across grasps to show margin differs. &nbsp; <b>1 week.</b> &nbsp; <b>Future (thesis):</b> feed that margin back as a grasp-generation reward — close the loop.</span>
